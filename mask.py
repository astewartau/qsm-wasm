import nibabel as nib
import numpy as np


img_nii = nib.load("./reference/chimap.nii")
mask_nii = nib.load("./reference/mask.nii")

img = img_nii.get_fdata()
mask = mask_nii.get_fdata().astype(bool) 


masked_img = np.where(mask, img, 0) 

masked_nii = nib.Nifti1Image(masked_img, affine=img_nii.affine, header=img_nii.header)
nib.save(masked_nii, './reference/Chimap_masked.nii')


