import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def visualize_slices(nii_path, num_slices=5):
    img = nib.load(nii_path)
    vol = img.get_fdata().astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    depth = vol.shape[2]
    indices = np.linspace(0, depth-1, num_slices, dtype=int)
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 3))
    for ax, idx in zip(axes, indices):
        ax.imshow(vol[:, :, idx], cmap='gray')
        ax.set_title(f"Slice {idx}")
        ax.axis('off')
    fig.suptitle(f"{nii_path} — {num_slices} slices", fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    p = argparse.ArgumentParser(description="Quick preview of a NIfTI volume")
    p.add_argument('nii_file', help="Path to .nii or .nii.gz file")
    p.add_argument('--slices', '-n', type=int, default=5,
                   help="Number of evenly spaced slices to show")
    args = p.parse_args()
    visualize_slices(args.nii_file, args.slices)

if __name__ == '__main__':
    main()
