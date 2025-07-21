import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from models.cancer_classification import SmallLungCancer3DCNN
from utils import find_pt_lung_cancer_folders

def load_nifti_volume(nii_path, target_slices=80, target_size=(64,64)):
    img = nib.load(nii_path)
    vol = img.get_fdata().astype(np.float32)
    vol = np.transpose(vol, (2,0,1))
    vol = (vol - vol.mean()) / (vol.std() + 1e-6)
    vt = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    vt = F.interpolate(vt,
                       size=(target_slices, *target_size),
                       mode='trilinear', align_corners=False)
    return vt.squeeze(0), vol

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(
            lambda _, __, out: setattr(self, 'activations', out.detach())
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
                            mode='trilinear', align_corners=False)[0,0]
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
        hm  = cam[i]
        thr = (np.percentile(hm, threshold_value)
               if threshold_method=='percentile'
               else threshold_value)
        mask = (hm > thr).astype(np.uint8) * 255

        fig, ax = plt.subplots(figsize=(1,1), dpi=PIXELS)
        ax.imshow(hm, cmap='gray'); ax.axis('off')
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(output_dir,f'slice_{i:03d}_intensity.png'),
                    dpi=PIXELS, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(1,1), dpi=PIXELS)
        ax.imshow(img, cmap='gray')
        ax.imshow(hm,  cmap='jet', alpha=0.4)
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(output_dir,f'slice_{i:03d}_overlay.png'),
                    dpi=PIXELS, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(1,1), dpi=PIXELS)
        ax.imshow(mask, cmap='gray'); ax.axis('off')
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(output_dir,f'slice_{i:03d}_mask.png'),
                    dpi=PIXELS, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

if __name__ == '__main__':
    csv_path = './fb_metadata.csv'
    data_root = r"F:\PET_FULL_BODY_DATASET\nifti\FDG-PET-CT-Lesions"
    cancer_paths = find_pt_lung_cancer_folders(
        csv_path, data_root,
        diagnosis="LUNG_CANCER",
        include_file_name="PET_segmented.nii.gz"
    )
    print(f"Data count {len(cancer_paths)}")
    NII_FILES        = cancer_paths
    CHECKPOINT       = r'F:\Obrada slike\pet-scan-diffusion-anomaly-detection\model_results\lung_cancer_20250719_220641\best_epoch27_acc0.720.pth'
    BASE_OUTPUT_DIR  = './gradcam_out'
    TARGET_SLICES    = 80
    TARGET_SIZE      = (64,64)
    SLICE_INDICES    = None
    THRESHOLD_METHOD = 'percentile'
    THRESHOLD_VALUE  = 95

    model = SmallLungCancer3DCNN(num_classes=2, dropout_p=0.2)
    ckpt  = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(ckpt)
    target_layer = model.features[8]
    gradcam = GradCAM3D(model, target_layer)

    it = 0
    for nii_path in NII_FILES:
        it += 1
        case_id = os.path.splitext(os.path.basename(nii_path))[0].replace(".nii.gz", "") + f"_{it}"
        out_dir = os.path.join(BASE_OUTPUT_DIR, case_id)

        vt, raw_vol = load_nifti_volume(nii_path,
                                        target_slices=TARGET_SLICES,
                                        target_size=TARGET_SIZE)
        vol_resized = vt.squeeze(0).squeeze(0).numpy()
        x = vt.unsqueeze(0)

        cam3d = gradcam(x)
        save_heatmaps(vol_resized, cam3d, out_dir,
                      SLICE_INDICES, THRESHOLD_VALUE, THRESHOLD_METHOD)

        count = len(SLICE_INDICES or list(range(vol_resized.shape[0])))
        print(f"[{case_id}] saved {count} maps → {out_dir}")