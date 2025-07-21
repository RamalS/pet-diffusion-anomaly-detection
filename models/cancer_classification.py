import os
import random
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import nibabel as nib
import numpy as np
from tqdm import tqdm

from collections import Counter

from utils import find_pt_lung_cancer_folders
from dataset.nifti_lung_pet import NiftiLungPETDataset

def print_class_ratios(labels, split_name="Split"):
    counter = Counter(labels)
    total = len(labels)
    print(f"\n{split_name} set (total = {total}):")
    for cls in sorted(counter):
        cnt = counter[cls]
        pct = cnt / total * 100
        print(f"  Class {cls}: {cnt}/{total} ({pct:.2f}%)")

def random_3d_augment(volume: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        volume = np.flip(volume, axis=2).copy()
    k = random.randint(0, 3)
    volume = np.rot90(volume, k, axes=(1, 2)).copy()
    return volume

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

class SmallLungCancer3DCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_p=0.3):
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
            nn.Dropout3d(p=dropout_p),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_epoch(model, loader, criterion, optimizer, device, scaler, epoch=None):
    model.train()
    running_loss = correct = total = 0
    loop = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for inputs, labels in loop:
        inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last_3d)
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
    return running_loss/total, correct/total

def eval_epoch(model, loader, criterion, device, scaler=None, epoch=None):
    model.eval()
    running_loss = correct = total = 0
    loop = tqdm(loader, desc=f"Val Epoch {epoch}", leave=False)
    with torch.no_grad():
        for inputs, labels in loop:
            inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last_3d)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=running_loss/total, acc=correct/total)
    return running_loss/total, correct/total

def main(data_root, batch_size=4, epochs=50, lr=1e-4, val_split=0.2):
    cudnn.benchmark = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("model_results", f"lung_cancer_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving all outputs to: {results_dir}")

    metrics_csv = os.path.join(results_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    csv_path = "./fb_metadata.csv"
    healthy_paths = find_pt_lung_cancer_folders(
        csv_path, data_root,
        diagnosis="NEGATIVE",
        include_file_name="PET_segmented.nii.gz"
    )
    cancer_paths = find_pt_lung_cancer_folders(
        csv_path, data_root,
        diagnosis="LUNG_CANCER",
        include_file_name="PET_segmented.nii.gz"
    )

    file_paths = healthy_paths + cancer_paths
    labels     = [0] * len(healthy_paths) + [1] * len(cancer_paths)

    num  = len(file_paths)
    perm = torch.randperm(num).tolist()
    split = int(num * (1 - val_split))
    train_idx, val_idx = perm[:split], perm[split:]

    train_lbls = [labels[i] for i in train_idx]
    idx0 = [i for i, l in zip(train_idx, train_lbls) if l == 0]
    idx1 = [i for i, l in zip(train_idx, train_lbls) if l == 1]
    idx0_und = random.sample(idx0, len(idx1))
    train_idx = idx1 + idx0_und
    random.shuffle(train_idx)

    train_paths = [file_paths[i] for i in train_idx]
    train_lbls  = [labels[i]     for i in train_idx]
    val_paths   = [file_paths[i] for i in val_idx]
    val_lbls    = [labels[i]     for i in val_idx]

    print_class_ratios(train_lbls, "Train")
    print_class_ratios(val_lbls,   "Val")

    train_ds = NiftiLungPETDataset(
        train_paths, train_lbls,
        transform=random_3d_augment,
        target_slices=80, target_size=(64, 64)
    )
    val_ds = NiftiLungPETDataset(
        val_paths, val_lbls,
        transform=None,
        target_slices=50, target_size=(64, 64)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    model = SmallLungCancer3DCNN(num_classes=2, dropout_p=0.3)
    model = model.to(device).to(memory_format=torch.channels_last_3d)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = GradScaler()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader,
            criterion, optimizer,
            device, scaler, epoch
        )
        val_loss, val_acc = eval_epoch(
            model, val_loader,
            criterion, device,
            scaler, epoch
        )
        scheduler.step()

        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        print(
            f"[{epoch}/{epochs}] "
            f"Train: loss={train_loss:.3f}, acc={train_acc:.3f} | "
            f"Val:   loss={val_loss:.3f}, acc={val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_name = f"best_epoch{epoch:02d}_acc{val_acc:.3f}.pth"
            ckpt_path = os.path.join(results_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best model to {ckpt_path}")

    final_path = os.path.join(results_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")

if __name__ == "__main__":
    root_folder = r"F:\PET_FULL_BODY_DATASET\nifti\FDG-PET-CT-Lesions"
    main(data_root=root_folder)