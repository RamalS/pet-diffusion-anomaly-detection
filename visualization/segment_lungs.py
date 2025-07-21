import argparse
import nibabel as nib
import numpy as np


def crop_lung_slices(
    input_nii: str,
    output_nii: str,
    hu_lower: int = -1000,
    hu_upper: int = -400,
    min_pixels: int = 5000,
) -> None:
    img = nib.load(input_nii)
    arr = img.get_fdata().astype(np.float32)
    affine = img.affine

    hdr = img.header
    slope = hdr.get('slope', 1.0)
    intercept = hdr.get('inter', 0.0)
    arr = arr * slope + intercept

    lung_mask = (arr >= hu_lower) & (arr <= hu_upper)

    slice_counts = lung_mask.sum(axis=(1, 2))
    z_idxs = np.where(slice_counts > min_pixels)[0]
    if z_idxs.size == 0:
        raise RuntimeError("No lung slices found. Try adjusting thresholds or checking volume orientation.")
    z_min, z_max = z_idxs.min(), z_idxs.max()
    print(f"Lung slices found between Z = {z_min} and {z_max} (inclusive)")

    submask = lung_mask[z_min : z_max + 1]
    proj_mask = submask.any(axis=0)
    ys, xs = np.where(proj_mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    print(f"Cropping in-plane bounding box: Y[{y_min}:{y_max}], X[{x_min}:{x_max}]")

    cropped = arr[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]

    out_img = nib.Nifti1Image(cropped, affine)
    nib.save(out_img, output_nii)
    print(f"Saved cropped volume to {output_nii}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop full-body CT to lung region using simple HU thresholding"
    )
    parser.add_argument("input", help="Input NIfTI file (.nii or .nii.gz)")
    parser.add_argument("output", help="Output cropped NIfTI file")
    parser.add_argument(
        "--hu_lower", type=int, default=-1000, help="Lower HU threshold (default: -1000)"
    )
    parser.add_argument(
        "--hu_upper", type=int, default=-400, help="Upper HU threshold (default: -400)"
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=5000,
        help="Minimum mask pixels per slice to consider lung (default: 5000)",
    )
    args = parser.parse_args()
    crop_lung_slices(
        args.input, args.output, args.hu_lower, args.hu_upper, args.min_pixels
    )