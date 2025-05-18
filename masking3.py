import numpy as np
import nibabel as nib
#import time
from scipy.ndimage import (
    gaussian_filter, label, binary_fill_holes,
    binary_erosion, binary_dilation, generate_binary_structure
)

def run_masking(fieldmap_path, treshold = 80):


    img = nib.load(fieldmap_path)
    data = img.get_fdata()

    # Step 1: Normalize and smooth
    data = data / np.max(data)
    smoothed = gaussian_filter(data, sigma=1)

    # Step 2: Adaptive threshold - aggressive
    thresh = np.percentile(smoothed, treshold)  
    binary = smoothed > thresh

    # Step 3: 3D connected component analysis
    struct = generate_binary_structure(3, 2)  
    labeled, num = label(binary, structure=struct)
    if num == 0:
        raise RuntimeError("No foreground found.")

    # Step 4: Select largest component
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = sizes.argmax()
    mask = (labeled == largest_label)

    # Step 5: Fill holes slice-by-slice
    for i in range(mask.shape[2]):
        mask[:, :, i] = binary_fill_holes(mask[:, :, i])

    # Step 6: Morphological cleanup
    mask = binary_erosion(mask, structure=struct, iterations=1)
    mask = binary_dilation(mask, structure=struct, iterations=2)

    # Finalize and save
    out_img = nib.Nifti1Image(mask.astype(np.uint8), affine=img.affine)

    out_path = "mask.nii"
  
    nib.save(out_img, out_path)






    return out_path
