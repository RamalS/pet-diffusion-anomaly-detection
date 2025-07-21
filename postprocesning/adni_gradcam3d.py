import os
import argparse
from glob import glob

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pydicom

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
    return vt.squeeze(0)

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
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(img, cmap='gray')
        ax.imshow(hm, cmap='jet', alpha=0.4)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'slice_{i:03d}_overlay.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        mask = (hm > thr).astype(np.uint8) * 255
        fig2, ax2 = plt.subplots(figsize=(4,4))
        ax2.imshow(mask, cmap='gray')
        ax2.axis('off')
        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, f'slice_{i:03d}_mask.png'),
                     dpi=150, bbox_inches='tight')
        plt.close(fig2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dcm_folder', required=True,
                   help="Path to folder of .dcm slices")
    p.add_argument('--checkpoint', required=True,
                   help=".pth model checkpoint")
    p.add_argument('--output_dir', default='gradcam_out',
                   help="Where to save output images")
    p.add_argument('--slice_indices', type=int, nargs='+',
                   help="List of slice indices to save (default: all)")
    p.add_argument('--threshold_method', choices=['fixed','percentile'],
                   default='percentile',
                   help="How to compute binary threshold")
    p.add_argument('--threshold_value', type=float,
                   default=95,
                   help=("If fixed: threshold in [0,1]; "
                         "if percentile: percentile in [0,100]"))
    args = p.parse_args()

    model = SmallAlzheimer3DCNN(num_classes=2, dropout_p=0.2)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt)
    target_layer = model.features[8]

    x = load_dicom_volume(args.dcm_folder)
    raw = x[0].numpy()
    x = x.unsqueeze(0)

    gradcam = GradCAM3D(model, target_layer)
    cam3d = gradcam(x)

    save_heatmaps(raw, cam3d, args.output_dir, args.slice_indices,
                  args.threshold_value, args.threshold_method)
    count = len(args.slice_indices or list(range(raw.shape[0])))
    print(f"Saved {count} overlays and masks (method={args.threshold_method}) to {args.output_dir}")

if __name__ == '__main__':
    main()