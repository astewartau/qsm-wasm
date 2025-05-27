import nibabel as nib
import numpy as np

# --- INPUT FILE PATHS ---
img_path = 'image.nii'       # Input MRI image
mask_path = 'mask.nii'       # Binary mask image (same shape)

# --- LOAD FILES ---
img_nii = nib.load(img_path)
mask_nii = nib.load(mask_path)

img = img_nii.get_fdata()
mask = mask_nii.get_fdata().astype(bool)  # Convert to boolean mask

# --- APPLY MASK ---
masked_img = np.where(mask, img, 0)  # Zero out values outside the mask

# --- SAVE RESULT ---
masked_nii = nib.Nifti1Image(masked_img, affine=img_nii.affine, header=img_nii.header)
nib.save(masked_nii, 'masked_image.nii')

print("Masked image saved as 'masked_image.nii'")