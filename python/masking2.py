import numpy as np
import nibabel as nib
from scipy.ndimage import label, binary_fill_holes

def run_masking(fieldmap_path):
    img = nib.load(fieldmap_path)
    data = img.get_fdata()
    norm_data = data / np.max(data)

    # Step 1: Loose threshold to get possible brain + connected junk
    loose_mask = norm_data > 0.1

    # Step 2: Connected component labeling
    labeled, num_features = label(loose_mask)
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # background
    largest_label = component_sizes.argmax()
    largest_component = labeled == largest_label

    # Step 3: Fill holes in the largest component
    filled = binary_fill_holes(largest_component)

    # Step 4: Apply stricter threshold within this mask to refine
    refined_mask = (filled & (norm_data > 0.4)).astype(np.uint8)

    out_img = nib.Nifti1Image(refined_mask, affine=img.affine)
    out_path = "rts_output.nii"
    nib.save(out_img, out_path)

    return out_path
