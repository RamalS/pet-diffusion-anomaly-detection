import os
import argparse
from glob import glob

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from models.cancer_classification import SmallLungCancer3DCNN

def load_nifti_volume(nii_path, target_slices=80, target_size=(64,64)):
    img = nib.load(nii_path)
    vol = img.get_fdata().astype(np.float32)
    vol = np.transpose(vol, (2, 0, 1))
    vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-6)
    vt = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    vt = F.interpolate(vt,
                       size=(target_slices, *target_size),
                       mode='trilinear',
                       align_corners=False)
    return vt.squeeze(0)

def preview_slices(vol3d, n=5):
    D, H, W = vol3d.shape
    idxs = np.linspace(0, D-1, n, dtype=int)
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for ax, z in zip(axes, idxs):
        ax.imshow(vol3d[z], cmap='gray')
        ax.set_title(f"z={z}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

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

def save_heatmaps(volume, cam, output_dir,
                  threshold_value=0.5, threshold_method='fixed'):
    os.makedirs(output_dir, exist_ok=True)
    D = volume.shape[0]
    idxs = list(range(D))
    for i in idxs:
        img = volume[i]
        hm = cam[i]

        if threshold_method == 'percentile':
            thr = np.percentile(hm, threshold_value)
        else:
            thr = threshold_value

        fig_orig, ax_orig = plt.subplots(figsize=(4,4))
        ax_orig.imshow(img, cmap='gray')
        ax_orig.axis('off')
        fig_orig.tight_layout()
        fig_orig.savefig(os.path.join(output_dir, f'slice_{i:03d}_original.png'),
                         dpi=150, bbox_inches='tight')
        plt.close(fig_orig)

        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(img, cmap='gray')
        ax.imshow(hm, cmap='jet', alpha=0.4)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'slice_{i:03d}_overlay.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        mask = (hm > thr).astype(np.uint8) * 255
        fig_mask, ax_mask = plt.subplots(figsize=(4,4))
        ax_mask.imshow(mask, cmap='gray')
        ax_mask.axis('off')
        fig_mask.tight_layout()
        fig_mask.savefig(os.path.join(output_dir, f'slice_{i:03d}_mask.png'),
                         dpi=150, bbox_inches='tight')
        plt.close(fig_mask)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nii_file', required=True,
                   help="Path to .nii or .nii.gz file")
    p.add_argument('--checkpoint', required=True,
                   help=".pth model checkpoint")
    p.add_argument('--output_dir', default='gradcam_out',
                   help="Where to save output images")
    p.add_argument('--threshold_method', choices=['fixed','percentile'],
                   default='percentile',
                   help="How to compute binary threshold")
    p.add_argument('--threshold_value', type=float,
                   default=95,
                   help=("If fixed: threshold in [0,1]; "
                         "if percentile: percentile in [0,100]"))
    args = p.parse_args()

    model = SmallLungCancer3DCNN(num_classes=2, dropout_p=0.2)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt)
    target_layer = model.features[8]

    vt = load_nifti_volume(args.nii_file)
    raw = vt[0].numpy()

    preview_slices(raw, n=5)

    x = vt.unsqueeze(0)
    gradcam = GradCAM3D(model, target_layer)
    cam3d = gradcam(x)

    save_heatmaps(raw, cam3d, args.output_dir,
                  args.threshold_value, args.threshold_method)
    count = raw.shape[0]
    print(f"Saved {count} originals, overlays, and masks "
          f"(method={args.threshold_method}) to {args.output_dir}")

if __name__ == '__main__':
    main()