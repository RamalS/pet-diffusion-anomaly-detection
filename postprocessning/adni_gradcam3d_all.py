import os
from glob import glob

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from utils import find_dcm_folders
from models.adni_classification import SmallAlzheimer3DCNN

def load_dicom_volume(folder, target_slices=20, target_size=(64,64)):
    files = sorted(glob(os.path.join(folder, '*.dcm')))
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda s: float(getattr(s, 'SliceLocation', s.InstanceNumber)))
    vol = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
    slope = getattr(slices[0], 'RescaleSlope', 1.0)
    intercept = getattr(slices[0], 'RescaleIntercept', 0.0)
    vol = vol * slope + intercept
    vol = (vol - vol.mean()) / (vol.std() + 1e-6)
    vt = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    vt = F.interpolate(vt,
                       size=(target_slices, *target_size),
                       mode='trilinear',
                       align_corners=False)
    return vt.squeeze(0), vol

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(
            lambda _, __, output: setattr(self, 'activations', output.detach())
        )
        target_layer.register_full_backward_hook(
            lambda _, __, grad_out: setattr(self, 'gradients', grad_out[0].detach())
        )

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[0, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        grads = self.gradients
        weights = grads.mean(dim=[2,3,4], keepdim=True)
        cams = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cams)
        cam = F.interpolate(cam,
                            size=x.shape[2:],
                            mode='trilinear',
                            align_corners=False)[0,0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam.cpu().numpy()

PIXELS = 128

def save_heatmaps(volume, cam, output_dir, slice_indices=None,
                  threshold_value=0.5, threshold_method='fixed'):
    os.makedirs(output_dir, exist_ok=True)
    D = volume.shape[0]
    idxs = slice_indices or list(range(D))
    for i in idxs:
        img = volume[i]
        hm = cam[i]
        if threshold_method == 'percentile':
            thr = np.percentile(hm, threshold_value)
        else:
            thr = threshold_value
        mask = (hm > thr).astype(np.uint8) * 255

        fig_int, ax_int = plt.subplots(figsize=(1, 1), dpi=PIXELS)
        ax_int.imshow(hm, cmap='gray')
        ax_int.axis('off')
        fig_int.tight_layout(pad=0)
        fig_int.savefig(
            os.path.join(output_dir, f'slice_{i:03d}_intensity.png'),
            dpi=PIXELS,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig_int)

        fig_ov, ax_ov = plt.subplots(figsize=(1, 1), dpi=PIXELS)
        ax_ov.imshow(img, cmap='gray')
        ax_ov.imshow(hm, cmap='jet', alpha=0.4)
        ax_ov.axis('off')
        fig_ov.tight_layout(pad=0)
        fig_ov.savefig(
            os.path.join(output_dir, f'slice_{i:03d}_overlay.png'),
            dpi=PIXELS,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig_ov)

        fig_mask, ax_mask = plt.subplots(figsize=(1, 1), dpi=PIXELS)
        ax_mask.imshow(mask, cmap='gray')
        ax_mask.axis('off')
        fig_mask.tight_layout(pad=0)
        fig_mask.savefig(
            os.path.join(output_dir, f'slice_{i:03d}_mask.png'),
            dpi=PIXELS,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig_mask)

if __name__ == '__main__':
    DICOM_FOLDERS   = find_dcm_folders("../PET_AD")
    CHECKPOINT      = 'best_model_epoch_28_acc_0.837.pth'
    BASE_OUTPUT_DIR = './results/gradcam_outputs'
    TARGET_SLICES   = 20
    TARGET_SIZE     = (64, 64)
    SLICE_INDICES   = None
    THRESHOLD_METHOD= 'percentile'
    THRESHOLD_VALUE = 95

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    model = SmallAlzheimer3DCNN(num_classes=2, dropout_p=0.2)
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(ckpt)
    target_layer = model.features[8]
    gradcam = GradCAM3D(model, target_layer)

    for folder in DICOM_FOLDERS:
        patient_id = os.path.basename(folder.rstrip('/'))
        out_dir = os.path.join(BASE_OUTPUT_DIR, patient_id)
        vt, raw_vol = load_dicom_volume(folder, TARGET_SLICES, TARGET_SIZE)
        vol_resized = vt.squeeze(0).squeeze(0).numpy()
        x = vt.unsqueeze(0)
        cam3d = gradcam(x)
        save_heatmaps(vol_resized, cam3d, out_dir, SLICE_INDICES,
                      THRESHOLD_VALUE, THRESHOLD_METHOD)
        count = len(SLICE_INDICES or list(range(vol_resized.shape[0])))
        print(f"Processed {patient_id}: saved {count} overlays and masks to {out_dir}")
