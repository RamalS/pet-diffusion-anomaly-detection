import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata(), img

def resample_to_target(data, source_img, target_img):
    src_shape = data.shape
    tgt_shape = target_img.shape
    factors = [tgt_shape[i] / src_shape[i] for i in range(3)]
    data_resampled = ndi.zoom(data, factors, order=0)
    return data_resampled

def main(ct_path=r"F:\PET_FULL_BODY_DATASET\nifti\FDG-PET-CT-Lesions\PETCT_0117d7f11f\09-13-2001-NA-PET-CT Ganzkoerper  primaer mit KM-68547\PET_segmented.nii.gz", seg_path=r"F:\PET_FULL_BODY_DATASET\nifti\FDG-PET-CT-Lesions\PETCT_0117d7f11f\09-13-2001-NA-PET-CT Ganzkoerper  primaer mit KM-68547\SEG_segmented.nii.gz"):
    ct, ct_nii = load_nifti(ct_path)
    seg, seg_nii = load_nifti(seg_path)

    if seg.shape != ct.shape:
        seg = resample_to_target(seg, seg_nii, ct_nii)
        print(f"Resampled SEG from {seg_nii.shape} to {ct.shape}")

    n_slices = ct.shape[2]
    init_slice = n_slices // 2

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ct_img = ax.imshow(ct[:, :, init_slice].T, cmap='gray', origin='lower')
    mask = np.ma.masked_where(seg[:, :, init_slice].T == 0, seg[:, :, init_slice].T)
    seg_img = ax.imshow(mask, cmap='autumn', alpha=0.5, origin='lower')
    ax.set_title(f'Axial slice {init_slice}')
    ax.axis('off')

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', valmin=0, valmax=n_slices-1, valinit=init_slice, valstep=1, valfmt='%0.0f')

    def update(val):
        idx = int(slider.val)
        ct_img.set_data(ct[:, :, idx].T)
        new_mask = np.ma.masked_where(seg[:, :, idx].T == 0, seg[:, :, idx].T)
        seg_img.set_data(new_mask)
        ax.set_title(f'Axial slice {idx}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

if __name__ == '__main__':
    main()
