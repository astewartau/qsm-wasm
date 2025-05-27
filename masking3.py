import numpy as np
import nibabel as nib
#import time
from scipy.ndimage import (
    gaussian_filter, label, binary_fill_holes,
    binary_erosion, binary_dilation, generate_binary_structure
)

def run_masking(fieldmap_path, treshold = 80):

    """
    Brain extraction/masking function using magnitude image
    Input: treshold value (default 80)

    4 step approach:
    1. Gaussian smoothing
    2. Find largest connected component
    3. Fill 2D axial slices
    3. Erode and dilate the mask to remove small disturbances around the edges



    """

    img = nib.load(fieldmap_path)
    data = img.get_fdata()

    #Normalize the image values to [0, 1]
    data = data / np.max(data)

    #Gaussian filter to make the image smoother -> mask at end will be smoother
    # 0.5 was chosen after experimenting with different values, this can be changed
    # Could eventually be a parameter of the function
    smoothed = gaussian_filter(data, sigma=0.5)

    # Adaptive threshold for constintency (in terms of threshold value) across different datasets
    # I see this as implicit "normalisation" of the threshold value
    thresh = np.percentile(smoothed, treshold)  
    binary = smoothed > thresh


    #Select largest component from 3D connected component analysis
    struct = generate_binary_structure(3, 2)  
    labeled, num = label(binary, structure=struct)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = sizes.argmax()
    mask = (labeled == largest_label)

    #Fill holes slice-by-slice (2D) in axial (XY) plane
    #(Gives better results than 3D filling, by experimenting)
    for i in range(mask.shape[2]):
        mask[:, :, i] = binary_fill_holes(mask[:, :, i])


    # remove small disturbances around the edges
    mask = binary_erosion(mask, structure=struct, iterations=1)
    # Smoothen edges
    mask = binary_dilation(mask, structure=struct, iterations=2)

    # -> Binary mask with values 0 and 1





    out_img = nib.Nifti1Image(mask.astype(np.uint8), affine=img.affine)
    out_path = "mask.nii"
    nib.save(out_img, out_path)

    return out_path
