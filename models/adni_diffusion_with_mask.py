import os
import re
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import find_dcm_folders
from skimage.transform import resize
import pydicom
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.io import imread

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DicomPETDataset(Dataset):
    def __init__(
        self,
        patient_dirs,
        mask_root=None,
        specific_mask_folder: str = None,
        transform=None,
        target_slices=20,
        target_size=(256, 256)
    ):
        self.patient_dirs = patient_dirs
        self.mask_root = mask_root
        self.specific_mask_folder = specific_mask_folder
        self.transform = transform
        self.target_slices, (self.target_h, self.target_w) = target_slices, target_size

        if self.mask_root:
            self.mask_folders = [
                os.path.join(self.mask_root, d)
                for d in os.listdir(self.mask_root)
                if os.path.isdir(os.path.join(self.mask_root, d))
            ]
        else:
            self.mask_folders = []

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        folder = self.patient_dirs[idx]
        files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".dcm")],
            key=lambda p: float(getattr(pydicom.dcmread(p), "SliceLocation", pydicom.dcmread(p).InstanceNumber))
        )
        slices = [pydicom.dcmread(f) for f in files]
        volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
        slope = getattr(slices[0], "RescaleSlope", 1.0)
        intercept = getattr(slices[0], "RescaleIntercept", 0.0)
        volume = (volume * slope + intercept)
        volume = (volume - volume.mean()) / (volume.std() + 1e-6)

        vol_t = torch.from_numpy(volume)[None, None]
        vol_t = F.interpolate(
            vol_t,
            size=(self.target_slices, self.target_h, self.target_w),
            mode="trilinear",
            align_corners=False
        )
        volume_tensor = vol_t.squeeze(0)

        if self.mask_folders:
            if self.specific_mask_folder:
                mask_folder = self.specific_mask_folder
            else:
                mask_folder = random.choice(self.mask_folders)

            mfiles = sorted(
                [os.path.join(mask_folder, f)
                 for f in os.listdir(mask_folder)
                 if f.lower().endswith(".png")
                    and 'slice_' in f.lower()
                    and 'mask' in f.lower()],
                key=lambda x: int(re.search(r"slice_(\d+)", x).group(1))
            )
            mask_list = []
            for mf in mfiles:
                m = imread(mf, as_gray=True).astype(np.float32)
                mask_list.append(torch.from_numpy(m))
            mask_vol = torch.stack(mask_list, dim=0)
            mask_t = mask_vol[None, None]
            mask_t = F.interpolate(
                mask_t,
                size=(self.target_slices, self.target_h, self.target_w),
                mode="trilinear",
                align_corners=False
            )
            mask = mask_t.squeeze(0).squeeze(0)
            mask = mask - mask.min()
            if mask.max() > 0:
                mask = mask / mask.max()
            mask_tensor = mask.unsqueeze(0)
        else:
            mask_tensor = torch.zeros_like(volume_tensor)

        if self.transform:
            volume_tensor = self.transform(volume_tensor)

        return volume_tensor, mask_tensor

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb_scale = torch.exp(
            torch.arange(half, device=t.device) * -np.log(10000) / (half - 1)
        )
        emb = t[:, None] * emb_scale[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class UNet3D(nn.Module):
    def __init__(
        self, in_channels=1, base_channels=32, channel_mults=(1, 2, 4), time_emb_dim=128
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )
        self.downs = nn.ModuleList()
        self.time_proj_down = nn.ModuleList()
        ch = in_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            block = nn.Sequential(
                nn.Conv3d(ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(),
            )
            self.downs.append(block)
            self.time_proj_down.append(nn.Linear(time_emb_dim, out_ch))
            ch = out_ch
        self.mid = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.ReLU(),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.ReLU(),
        )
        self.time_proj_mid = nn.Linear(time_emb_dim, ch)
        self.ups = nn.ModuleList()
        self.time_proj_up = nn.ModuleList()
        skip_channels = list(reversed([base_channels * m for m in channel_mults]))
        ch = base_channels * channel_mults[-1]
        for skip_ch in skip_channels:
            in_ch = ch + skip_ch
            out_ch = skip_ch
            block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(),
            )
            self.ups.append(block)
            self.time_proj_up.append(nn.Linear(time_emb_dim, out_ch))
            ch = out_ch
        self.final = nn.Conv3d(ch, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        features = []
        out = x
        for block, tp in zip(self.downs, self.time_proj_down):
            out = block[0](out)
            out = out + tp(t_emb)[:, :, None, None, None]
            out = block[1:](out)
            features.append(out)
            out = F.avg_pool3d(out, 2)
        out = self.mid[0](out)
        out = out + self.time_proj_mid(t_emb)[:, :, None, None, None]
        out = self.mid[1:](out)
        for block, tp in zip(self.ups, self.time_proj_up):
            skip = features.pop()
            out = F.interpolate(
                out, size=skip.shape[2:], mode="trilinear", align_corners=False
            )
            out = torch.cat([out, skip], dim=1)
            out = block[0](out)
            out = out + tp(t_emb)[:, :, None, None, None]
            out = block[1:](out)
        return self.final(out)

class Diffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, timesteps))
        alphas = 1 - self.betas
        alphas_cum = torch.cumprod(alphas, 0)
        self.register_buffer("alphas_cumprod", alphas_cum)
        self.register_buffer("sqrt_acp", torch.sqrt(alphas_cum))
        self.register_buffer("sqrt_omacp", torch.sqrt(1 - alphas_cum))

    def q_sample(self, x0, mask=None):
        batch_size = x0.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)

        sqrt_acp = self.sqrt_acp[t].view(-1, 1, 1, 1, 1)
        sqrt_omacp = self.sqrt_omacp[t].view(-1, 1, 1, 1, 1)

        x_noisy = sqrt_acp * x0 + sqrt_omacp * noise

        if mask is not None:
            if mask.shape != x0.shape:
                mask = mask.expand_as(x0)
            x_noisy = x0 * (1 - mask) + x_noisy * mask
            return x_noisy, noise, mask
        else:
            return x_noisy, noise, None

    def forward(self, x0, mask=None):
        x_noisy, noise, bern = self.q_sample(x0, mask)
        if bern is None:
            b = x0.size(0)
            t = torch.randint(0, self.timesteps, (b,), device=x0.device)
            t_in = t.float() / self.timesteps
        else:
            t_in = torch.zeros(x0.size(0), device=x0.device)
        pred = self.model(x_noisy, t_in)
        if bern is not None:
            loss = F.mse_loss(pred * bern, noise * bern)
        else:
            loss = F.mse_loss(pred, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t):
        b = x_t.size(0)
        tt = torch.full((b,), t, device=x_t.device, dtype=torch.long)
        noise_pred = self.model(x_t, tt.float() / self.timesteps)
        beta_t = self.betas[t]
        alpha_t = 1 - beta_t
        alpha_cum = self.alphas_cumprod[t]
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_cum)
        mean = coef1 * (x_t - coef2 * noise_pred)
        if t > 0:
            sigma_t = torch.sqrt(beta_t)
            noise = torch.randn_like(x_t)
            return mean + sigma_t * noise
        else:
            return mean

    @torch.no_grad()
    def reconstruct(self, x0, mask=None, steps=None):
        if steps is None:
            steps = self.timesteps
        assert steps <= self.timesteps

        t_T = steps - 1
        sqrt_acp_T   = self.sqrt_acp[t_T]
        sqrt_omacp_T = self.sqrt_omacp[t_T]
        noise = torch.randn_like(x0)

        x = sqrt_acp_T * x0 + sqrt_omacp_T * noise

        if mask is not None:
            mask = mask.expand_as(x0)
            x = x0 * (1 - mask) + x * mask

        initial_noised_x = x.clone()

        for t in reversed(range(steps)):
            x = self.p_sample(x, t)
            if mask is not None:
                x = x0 * (1 - mask) + x * mask

        return x, initial_noised_x

    @torch.no_grad()
    def sample(self, shape, steps=None):
        if steps is None:
            steps = self.timesteps
        x = torch.randn(shape, device=self.betas.device)
        for t in reversed(range(steps)):
            x = self.p_sample(x, t)
        return x

def train(root, mask_root, unet, diffusion, device, epochs=10, batch_size=2, lr=1e-4):
    paths = find_dcm_folders(root)
    ds = DicomPETDataset(paths, mask_root=mask_root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)
    for epoch in range(epochs):
        diffusion.train()
        total_loss = 0.0
        for x, mask in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            mask = mask.to(device)
            loss = diffusion(x, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg:.4f}")

    torch.save(
        {
            "unet_state_dict": unet.state_dict(),
            "diffusion_state_dict": diffusion.state_dict(),
        },
        "diffusion_model.pth",
    )
    print("Model saved to diffusion_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "example"], help="train or example")
    parser.add_argument("--dcm_root", default="dicom")
    parser.add_argument("--mask_root", default=None, help="root folder containing mask subfolders")
    parser.add_argument("--mask_folder", default=None, help="(example mode only) name of the mask subfolder to use")
    parser.add_argument("--example_folder")
    parser.add_argument("--n_slices", type=int, default=5, help="number of slices to visualize")
    args = parser.parse_args()

    unet = UNet3D().to(device)
    diffusion = Diffusion(unet).to(device)

    if args.mode == "train":
        train(args.dcm_root, args.mask_root, unet, diffusion, device)
    else:
        if not args.example_folder:
            raise ValueError("Provide --example_folder for example mode")
        example_folder = args.example_folder

        checkpoint = torch.load("diffusion_model.pth", map_location=device)
        unet.load_state_dict(checkpoint["unet_state_dict"])  
        diffusion.load_state_dict(checkpoint["diffusion_state_dict"])

        spec = None
        if args.mask_root and args.mask_folder:
            spec = os.path.join(args.mask_root, args.mask_folder)
            if not os.path.isdir(spec):
                raise ValueError(f"Mask folder {spec} doesn't exist.")

        ds = DicomPETDataset(
            [example_folder],
            mask_root=args.mask_root,
            specific_mask_folder=spec
        )
        loader = DataLoader(ds, batch_size=1)
        vol, mask = next(iter(loader))
        vol, mask = vol.to(device), mask.to(device)

        recon, initial_noised = diffusion.reconstruct(vol, mask=mask, steps=700)
        recon_np = recon.squeeze().cpu().numpy()
        initial_noised_np = initial_noised.squeeze().cpu().numpy()

        vol_np = vol.squeeze().cpu().numpy()
        diff_np = np.abs(vol_np - recon_np)
        mean_diff, std_diff = diff_np.mean(), diff_np.std()
        z_map_np = (diff_np - mean_diff) / std_diff
        threshold = 1
        anomaly_mask = (diff_np > threshold).astype(np.float32)

        D = recon_np.shape[0]
        idxs = np.linspace(0, D - 1, args.n_slices, dtype=int)
        fig, axes = plt.subplots(5, args.n_slices, figsize=(4 * args.n_slices, 11))

        mask_np = mask.squeeze().cpu().numpy()

        for i, idx in enumerate(idxs):
            axes[0, i].imshow(vol_np[idx], cmap="gray")
            axes[0, i].axis("off")
            axes[0, i].set_title(f"Original {idx}")

            axes[1, i].imshow(initial_noised_np[idx], cmap="gray")
            axes[1, i].axis("off")
            axes[1, i].set_title(f"Noised {idx}")

            axes[2, i].imshow(mask_np[idx], cmap="gray")
            axes[2, i].axis("off")
            axes[2, i].set_title(f"Noise Mask {idx}")

            axes[3, i].imshow(recon_np[idx], cmap="gray")
            axes[3, i].axis("off")
            axes[3, i].set_title(f"Reconstructed {idx}")

            axes[4, i].imshow(vol_np[idx], cmap="gray")
            axes[4, i].imshow(anomaly_mask[idx], cmap="Reds", alpha=0.5)
            axes[4, i].axis("off")
            axes[4, i].set_title(f"Anomalies {idx}")

        plt.show()