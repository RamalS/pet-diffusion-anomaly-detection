import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import nibabel as nib
import numpy as np

class NiftiLungPETDataset(Dataset):
    def __init__(
        self,
        file_paths,
        labels,
        transform=None,
        target_slices=32,
        target_size=(128, 128)
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.target_slices = target_slices
        self.target_h, self.target_w = target_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        nii = nib.load(path)
        volume = nii.get_fdata(dtype=np.float32)
        if volume.shape[-1] == volume.shape[0] or volume.shape[-1] == volume.shape[1]:
            volume = np.moveaxis(volume, -1, 0)
        volume = (volume - volume.mean()) / (volume.std() + 1e-6)
        if self.transform:
            volume = self.transform(volume)
        vol_t = torch.from_numpy(volume)
        vol_t = vol_t.unsqueeze(0).unsqueeze(0)
        vol_t = F.interpolate(
            vol_t,
            size=(self.target_slices, self.target_h, self.target_w),
            mode="trilinear",
            align_corners=False,
        )
        x = vol_t.squeeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
