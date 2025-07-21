import os
import re
import random
import argparse
import datetime
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from skimage.io import imread
from tqdm import tqdm

from utils import find_pt_lung_cancer_folders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NiftiLungPETDataset(Dataset):
    def __init__(
        self,
        file_paths,
        mask_dirs=None,
        transform=None,
        target_slices=32,
        target_size=(128, 128),
        random_mask: bool = False,
    ):
        self.file_paths = file_paths
        self.mask_dirs = mask_dirs or []
        self.transform = transform
        self.target_slices = target_slices
        self.target_h, self.target_w = target_size
        self.random_mask = random_mask

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        nii_path = self.file_paths[idx]
        img_nii = nib.load(nii_path)
        vol = img_nii.get_fdata(dtype=np.float32)

        if vol.ndim == 3:
            vol = np.moveaxis(vol, 2, 0)
        else:
            raise ValueError(f"Expected 3D volume, got shape {vol.shape}")

        if self.mask_dirs:
            if self.random_mask:
                mask_folder = random.choice(self.mask_dirs)
            else:
                mask_folder = self.mask_dirs[idx]

            mfiles = sorted(
                [os.path.join(mask_folder, f)
                 for f in os.listdir(mask_folder)
                 if f.lower().endswith(".png")
                 and 'slice_' in f.lower()
                 and 'mask' in f.lower()],
                key=lambda x: int(re.search(r"slice_(\d+)", os.path.basename(x)).group(1))
            )
            mask_slices = [imread(mf, as_gray=True).astype(np.float32) for mf in mfiles]
            mask = np.stack(mask_slices, axis=0)
            mask -= mask.min()
            if mask.max() > 0:
                mask /= mask.max()
        else:
            mask = np.zeros_like(vol, dtype=np.float32)

        vol = (vol - vol.mean()) / (vol.std() + 1e-6)

        if self.transform:
            vol = self.transform(vol)
            mask = self.transform(mask)

        vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

        vol_t = F.interpolate(
            vol_t,
            size=(self.target_slices, self.target_h, self.target_w),
            mode="trilinear",
            align_corners=False,
        )
        mask_t = F.interpolate(
            mask_t,
            size=(self.target_slices, self.target_h, self.target_w),
            mode="nearest",
        )
        mask_t = (mask_t > 0.5).float()

        x = vol_t.squeeze(0)
        m = mask_t.squeeze(0)
        return x, m


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
        self,
        in_channels=2,
        base_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
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
        skip_chs = list(reversed([base_channels * m for m in channel_mults]))
        ch = base_channels * channel_mults[-1]
        for skip_ch in skip_chs:
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

        self.final = nn.Conv3d(ch, 1, 1)

    def forward(self, x, t, mask=None):
        if mask is not None:
            x = torch.cat([x, mask], dim=1)
        t_emb = self.time_mlp(t)
        feats = []
        out = x

        for blk, tp in zip(self.downs, self.time_proj_down):
            out = blk[0](out)
            out = out + tp(t_emb)[:, :, None, None, None]
            out = blk[1:](out)
            feats.append(out)
            out = F.avg_pool3d(out, 2)

        out = self.mid[0](out)
        out = out + self.time_proj_mid(t_emb)[:, :, None, None, None]
        out = self.mid[1:](out)

        for blk, tp in zip(self.ups, self.time_proj_up):
            skip = feats.pop()
            out = F.interpolate(out, size=skip.shape[2:], mode="trilinear", align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = blk[0](out)
            out = out + tp(t_emb)[:, :, None, None, None]
            out = blk[1:](out)

        return self.final(out)


class Diffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.register_buf_