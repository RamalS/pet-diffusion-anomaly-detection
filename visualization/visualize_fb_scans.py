import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse

def visualize_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()

    axial_idx = data.shape[2] // 2
    sagittal_idx = data.shape[0] // 2
    coronal_idx = data.shape[1] // 2

    axial = data[:, :, axial_idx]
    sagittal = data[sagittal_idx, :, :]
    coronal = data[:, coronal_idx, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(np.rot90(axial), cmap='gray')
    axes[0].set_title('Axial Slice')
    axes[0].axis('off')
    
    axes[1].imshow(np.rot90(sagittal), cmap='gray')
    axes[1].set_title('Sagittal Slice')
    axes[1].axis('off')
    
    axes[2].imshow(np.rot90(coronal), cmap='gray')
    axes[2].set_title('Coronal Slice')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize NIfTI file slices.")
    parser.add_argument("path", type=str, help="Path to the NIfTI file (.nii or .nii.gz)")
    args = parser.parse_args()

    visualize_nifti(args.path)
