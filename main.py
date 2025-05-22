import os
import argparse
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DicomPETDataset(Dataset):
    def __init__(self, patient_dirs, transform=None, target_slices=81, target_size=(64, 64)):
        self.patient_dirs = patient_dirs
        self.transform = transform
        self.target_slices = target_slices
        self.target_h, self.target_w = target_size

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        folder = self.patient_dirs[idx]
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.dcm')]
        slices = [pydicom.dcmread(f) for f in files]
        slices.sort(key=lambda s: float(getattr(s, 'SliceLocation', s.InstanceNumber)))
        volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
        slope = getattr(slices[0], 'RescaleSlope', 1.0)
        intercept = getattr(slices[0], 'RescaleIntercept', 0.0)
        volume = volume * slope + intercept
        volume = (volume - volume.mean()) / (volume.std() + 1e-6)

        vol_t = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        vol_t = F.interpolate(vol_t, size=(self.target_slices, self.target_h, self.target_w), mode='trilinear', align_corners=False)
        volume = vol_t.squeeze(0).squeeze(0).numpy()

        if self.transform:
            volume = self.transform(volume)

        return torch.from_numpy(volume).unsqueeze(0)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb_scale = torch.exp(torch.arange(half, device=t.device) * -np.log(10000) / (half - 1))
        emb = t[:, None] * emb_scale[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, channel_mults=(1,2,4), time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        # Down
        self.downs = nn.ModuleList()
        self.time_proj_down = nn.ModuleList()
        ch = in_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            block = nn.Sequential(
                nn.Conv3d(ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.ReLU(),
                nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.ReLU()
            )
            self.downs.append(block)
            self.time_proj_down.append(nn.Linear(time_emb_dim, out_ch))
            ch = out_ch
        # Middle
        self.mid = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.ReLU(),
            nn.Conv3d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.ReLU()
        )
        self.time_proj_mid = nn.Linear(time_emb_dim, ch)
        # Up
        self.ups = nn.ModuleList()
        self.time_proj_up = nn.ModuleList()
        skip_channels = list(reversed([base_channels*m for m in channel_mults]))
        ch = base_channels * channel_mults[-1]
        for skip_ch in skip_channels:
            in_ch = ch + skip_ch
            out_ch = skip_ch
            block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.ReLU(),
                nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.ReLU()
            )
            self.ups.append(block)
            self.time_proj_up.append(nn.Linear(time_emb_dim, out_ch))
            ch = out_ch
        self.final = nn.Conv3d(ch, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        features = []
        out = x
        # Down path
        for block, tp in zip(self.downs, self.time_proj_down):
            out = block[0](out)
            out = out + tp(t_emb)[:, :, None, None, None]
            out = block[1:](out)
            features.append(out)
            out = F.avg_pool3d(out, 2)
        # Mid
        out = self.mid[0](out)
        out = out + self.time_proj_mid(t_emb)[:, :, None, None, None]
        out = self.mid[1:](out)
        # Up path
        for block, tp in zip(self.ups, self.time_proj_up):
            skip = features.pop()
            out = F.interpolate(out, size=skip.shape[2:], mode='trilinear', align_corners=False)
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
        # Beta schedule
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, timesteps))
        alphas = 1 - self.betas
        alphas_cum = torch.cumprod(alphas, 0)
        self.register_buffer('alphas_cumprod', alphas_cum)
        self.register_buffer('sqrt_acp', torch.sqrt(alphas_cum))
        self.register_buffer('sqrt_omacp', torch.sqrt(1 - alphas_cum))

    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        acp = self.sqrt_acp[t].view(-1,1,1,1,1)
        om = self.sqrt_omacp[t].view(-1,1,1,1,1)
        return acp * x0 + om * noise, noise

    def forward(self, x0):
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        x_noisy, noise = self.q_sample(x0, t)
        pred = self.model(x_noisy, t.float()/self.timesteps)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        One reverse diffusion step: x_t -> x_{t-1}
        """
        b = x_t.size(0)
        tt = torch.full((b,), t, device=x_t.device, dtype=torch.long)
        # predict noise
        noise_pred = self.model(x_t, tt.float()/self.timesteps)
        beta_t = self.betas[t]
        alpha_t = 1 - beta_t
        alpha_cum = self.alphas_cumprod[t]
        # mean of p(x_{t-1}|x_t)
        coef1 = 1/torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_cum)
        mean = coef1 * (x_t - coef2 * noise_pred)
        if t > 0:
            sigma_t = torch.sqrt(beta_t)
            noise = torch.randn_like(x_t)
            return mean + sigma_t * noise
        else:
            return mean

    @torch.no_grad()
    def reconstruct(self, x0, steps=100):
        """
        Reconstruct from clean x0 by adding noise to step t and then iteratively denoising.
        """
        # add noise to get x_t
        t_idx = steps-1
        b = x0.size(0)
        tt = torch.full((b,), t_idx, device=x0.device, dtype=torch.long)
        x, _ = self.q_sample(x0, tt)
        # iterative denoise
        for ti in reversed(range(steps)):
            x = self.p_sample(x, ti)
        return x

    @torch.no_grad()
    def sample(self, shape, steps=None):
        """
        Sample from scratch: start from pure noise.
        """
        if steps is None:
            steps = self.timesteps
        x = torch.randn(shape, device=self.betas.device)
        for t in reversed(range(steps)):
            x = self.p_sample(x, t)
        return x

def train(root, unet, diffusion, device, epochs=10, batch_size=2, lr=1e-4):
    paths = find_dcm_folders(root)
    ds = DicomPETDataset(paths)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)
    for epoch in range(epochs):
        diffusion.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            loss = diffusion(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg:.4f}")
    
    # Save model
    torch.save({
        'unet_state_dict': unet.state_dict(),
        'diffusion_state_dict': diffusion.state_dict()
    }, 'diffusion_model.pth')
    print("Model saved to diffusion_model.pth")

def example(dcm_folder, unet, diffusion, device, out_path='recon.npy', n_slices=5):
    # Load model
    checkpoint = torch.load('diffusion_model.pth', map_location=device)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
    print("Model loaded from diffusion_model.pth")

    ds = DicomPETDataset([dcm_folder])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
    vol = next(iter(loader)).to(device)

    unet.eval()
    diffusion.eval()
    recon = diffusion.reconstruct_org(vol, steps=1000)

    vol_np = vol.squeeze().cpu().numpy()
    recon_np = recon.squeeze().cpu().numpy()

    np.save(out_path, recon_np)
    print(f"Reconstruction saved to {out_path}")

    D = vol_np.shape[0]
    indices = np.linspace(0, D-1, n_slices, dtype=int)

    fig, axes = plt.subplots(2, n_slices, figsize=(3*n_slices, 6))
    for i, idx in enumerate(indices):
        axes[0, i].imshow(vol_np[idx], cmap='gray')
        axes[0, i].set_title(f"Orig slice {idx}")
        axes[0, i].axis('off')

        axes[1, i].imshow(recon_np[idx], cmap='gray')
        axes[1, i].set_title(f"Recon slice {idx}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train','example'], help="train or example")
    parser.add_argument('--dcm_root', default='dicom')
    parser.add_argument('--example_folder')
    parser.add_argument('--n_slices', type=int, default=5)
    args = parser.parse_args()

    unet = UNet3D().to(device)
    diffusion = Diffusion(unet).to(device)

    if args.mode=='train':
        train(args.dcm_root, unet, diffusion, device)
    else:
        if not args.example_folder:
            raise ValueError("Provide --example_folder for example mode")
        example_folder = args.example_folder
        # load model
        checkpoint = torch.load('diffusion_model.pth', map_location=device)
        unet.load_state_dict(checkpoint['unet_state_dict'])
        diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        # load data
        ds = DicomPETDataset([example_folder])
        loader = DataLoader(ds, batch_size=1)
        vol = next(iter(loader)).to(device)
        # reconstruct
        recon = diffusion.reconstruct(vol, steps=200)
        # or sample from noise:
        # recon = diffusion.sample(vol.shape, steps=100)
        recon_np = recon.squeeze().cpu().numpy()
        np.save('recon.npy', recon_np)
        # visualize
        D = recon_np.shape[0]
        idxs = np.linspace(0,D-1,args.n_slices, dtype=int)
        fig,axes = plt.subplots(2,args.n_slices, figsize=(3*args.n_slices,6))
        vol_np = vol.squeeze().cpu().numpy()
        for i,idx in enumerate(idxs):
            axes[0,i].imshow(vol_np[idx], cmap='gray'); axes[0,i].axis('off'); axes[0,i].set_title(f"Orig {idx}")
            axes[1,i].imshow(recon_np[idx], cmap='gray'); axes[1,i].axis('off'); axes[1,i].set_title(f"Recon {idx}")
        plt.tight_layout(); plt.show()
