import os
import argparse

import torch
import torch.nn.functional as F
import pydicom
import numpy as np

from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose

class SmallAlzheimer3DCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_p=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1,  8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Dropout3d(p=dropout_p),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def load_volume_from_folder(folder, target_slices=20, target_size=(64,64)):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith('.dcm')
    ]
    if not files:
        raise ValueError(f"No DICOM files found in {folder!r}")
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda s: float(getattr(s, 'SliceLocation', s.InstanceNumber)))
    volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
    slope = getattr(slices[0], 'RescaleSlope', 1.0)
    intercept = getattr(slices[0], 'RescaleIntercept', 0.0)
    volume = volume * slope + intercept
    volume = (volume - volume.mean()) / (volume.std() + 1e-6)
    vol_t = torch.from_numpy(volume)
    vol_t = vol_t.unsqueeze(0).unsqueeze(0)
    vol_t = F.interpolate(
        vol_t,
        size=(target_slices, *target_size),
        mode='trilinear',
        align_corners=False
    )
    return vol_t

def predict(model, volume_tensor, device):
    model.eval()
    volume_tensor = volume_tensor.to(device, dtype=torch.float32, non_blocking=True)
    with torch.no_grad():
        logits = model(volume_tensor)
        pred = logits.argmax(dim=1).item()
    return pred

def main():
    parser = argparse.ArgumentParser(
        description="Predict Alzheimer vs Normal from a DICOM folder"
    )
    parser.add_argument(
        "dicom_folder",
        type=str,
        help="Path to folder containing the DICOM series"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="alzheimer_3dcnn.pth",
        help="Path to the saved model state_dict"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device, e.g. 'cuda' or 'cpu'. Defaults to GPU if available."
    )
    args = parser.parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    model = SmallAlzheimer3DCNN(num_classes=2, dropout_p=0.2)
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).to(memory_format=torch.channels_last_3d)
    volume = load_volume_from_folder(args.dicom_folder)
    pred_class = predict(model, volume, device)
    class_names = {0: "Normal", 1: "Alzheimer"}
    print(f"Predicted class: {pred_class} → {class_names.get(pred_class, 'Unknown')}")

if __name__ == "__main__":
    main()
