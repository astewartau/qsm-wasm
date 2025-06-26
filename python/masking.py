import numpy as np
import nibabel as nib

def run_masking(fieldmap_path):
    img = nib.load(fieldmap_path)
    data = img.get_fdata()

    # Normalize and threshold
    norm_data = data / np.max(data)

    #Treshholding
    #Cast to int, because bool array is not supported in NIfTI
    mask = (norm_data > 0.3).astype(np.uint8)  



    # Create new NIfTI image with the same affine as input
    out_img = nib.Nifti1Image(mask, affine=img.affine)
    out_path = "mask.nii"
    nib.save(out_img, out_path)

    return out_path  # ✅ now JS can load the file from Pyodide FS