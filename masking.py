import numpy as np
import nibabel as nib

def auto_mask_and_render(fieldmap_path):
    img = nib.load(fieldmap_path)
    data = img.get_fdata()

    # Normalize and threshold
    norm_data = data / np.max(data)
    mask = norm_data > 0.2  # Simple threshold



    # Select the rightmost axial slice (e.g. slice with max brain area)
    slice_sums = mask.sum(axis=(0, 1))
    best_slice = np.argmax(slice_sums)
    slice2d = mask[:, :, best_slice].astype(np.uint8)

    return slice2d