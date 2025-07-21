import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import pydicom
import numpy as np
from tqdm import tqdm

from utils import find_dcm_folders
from collections import Counter

def print_class_ratios(labels, split_name="Split"):
    counter = Counter(labels)
    total = len(labels)
    print(f"\n{split_name} set (total samples = {total}):")
    for cls in sorted(counter):
        cnt = counter[cls]
        ratio = cnt / total * 100
        print(f"  Class {cls}: {cnt} / {total} ({ratio:.2f}%)")

def random_3d_augment(volume: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        volume = np.flip(volume, axis=2).copy()
    k = random.randint(0, 3)
    volume = np.rot90(volume, k, axes=(1, 2)).copy()
    return volume

class DicomPETDataset(Dataset):
    def __init__(
        self,
        patient_dirs,
        labels,
        transform=None,
        target_slices=20,
        target_size=(64, 64),
    ):
        self.patient_dirs = patient_dirs
        self.labels = labels
        self.transform = transform
        self.target_slices = target_slices
        self.target_h, self.target_w = target_size

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        folder = self.patient_dirs[idx]
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".dcm")
        ]
        slices = [pydicom.dcmread(f) for f in files]
        slices.sort(key=lambda s: float(getattr(s, "SliceLocation", s.InstanceNumber)))
        volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
        slope = getattr(slices[0], "RescaleSlope", 1.0)
        intercept = getattr(slices[0], "RescaleIntercept", 0.0)
        volume = volume * slope + intercept
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

class Alzheimer3DCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_p=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Dropout3d(p=dropout_p),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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

def train_epoch(model, loader, criterion, optimizer, device, scaler, epoch=None):
    model.train()
    running_loss = correct = total = 0
    loop = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for inputs, labels in loop:
        inputs = inputs.to(device, non_blocking=True).to(
            memory_format=torch.channels_last_3d
        )
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=running_loss/total, acc=correct/total)
    return running_loss / total, correct / total

def eval_epoch(model, loader, criterion, device, scaler=None, epoch=None):
    model.eval()
    running_loss = correct = total = 0
    loop = tqdm(loader, desc=f"Val Epoch {epoch}", leave=False)
    with torch.no_grad():
        for inputs, labels in loop:
            inputs = inputs.to(device, non_blocking=True).to(
                memory_format=torch.channels_last_3d
            )
            labels = labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=running_loss/total, acc=correct/total)
    return running_loss / total, correct / total

def main(data_root, batch_size=8, epochs=100, lr=1e-3, val_split=0.2):
    cudnn.benchmark = True
    healthy_dirs = find_dcm_folders(os.path.join(data_root, "CN_larger"))
    alz_dirs     = find_dcm_folders(os.path.join(data_root, "PET_AD_meshup"))
    patient_dirs = healthy_dirs + alz_dirs
    labels       = [0] * len(healthy_dirs) + [1] * len(alz_dirs)
    num_samples = len(patient_dirs)
    indices     = torch.randperm(num_samples).tolist()
    split_idx   = int(num_samples * (1 - val_split))
    train_idx   = indices[:split_idx]
    val_idx     = indices[split_idx:]
    train_lbls_all = [labels[i] for i in train_idx]
    class0_idx = [i for i, lbl in zip(train_idx, train_lbls_all) if lbl == 0]
    class1_idx = [i for i, lbl in zip(train_idx, train_lbls_all) if lbl == 1]
    n1 = len(class1_idx)
    class0_undersampled = random.sample(class0_idx, n1)
    train_idx = class1_idx + class0_undersampled
    random.shuffle(train_idx)
    train_dirs = [patient_dirs[i] for i in train_idx]
    train_lbls = [labels[i]       for i in train_idx]
    val_dirs   = [patient_dirs[i] for i in val_idx]
    val_lbls   = [labels[i]       for i in val_idx]
    train_ds = DicomPETDataset(
        train_dirs,
        train_lbls,
        transform=random_3d_augment,
        target_slices=20,
        target_size=(64,64),
    )
    val_ds = DicomPETDataset(
        val_dirs,
        val_lbls,
        transform=None,
        target_slices=20,
        target_size=(64,64),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    print_class_ratios(train_lbls, "Training")
    print_class_ratios(val_lbls,   "Validation")
    model = SmallAlzheimer3DCNN(dropout_p=0.2) \
            .to(device) \
            .to(memory_format=torch.channels_last_3d)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = GradScaler()
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, device, scaler, epoch
        )
        scheduler.step()
        print(
            f"[{epoch:2d}/{epochs}] "
            f"Train: loss={train_loss:.3f}, acc={train_acc:.3f} | "
            f" Val: loss={val_loss:.3f}, acc={val_acc:.3f} | "
            f" lr={scheduler.get_last_lr()[0]:.2e}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"best_model_epoch_{epoch:02d}_acc_{val_acc:.3f}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"New best model (acc={val_acc:.3f}), saved to {save_path}")
    torch.save(model.state_dict(), "alzheimer_3dcnn_final.pth")
    print("Training complete. Final model saved to alzheimer_3dcnn_final.pth")

if __name__ == "__main__":
    data_root = "../"
    main(data_root)
