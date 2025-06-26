import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure, gaussian_filter
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import nibabel as nib



def sphere_kernel(matrix_size, radius):
    # Create 3D cube from -radius to +radius
    rng = np.arange(-radius, radius + 1)
    a, b, c = np.meshgrid(rng, rng, rng, indexing='ij')

    # Create binary sphere inside the cube
    sphere = (a**2 / radius**2 + b**2 / radius**2 + c**2 / radius**2) <= 1

    # Normalize and adjust center value
    sphere = sphere.astype(np.float32) #cast to float before division
    sphere = -sphere / np.sum(sphere)
    sphere[radius, radius, radius] = 1 + sphere[radius, radius, radius]

    # Embed into full-size volume
    kernel = np.zeros(matrix_size, dtype=np.float32)
    x_start = matrix_size[0] // 2 - radius
    x_end   = matrix_size[0] // 2 + radius + 1
    y_start = matrix_size[1] // 2 - radius
    y_end   = matrix_size[1] // 2 + radius + 1
    z_start = matrix_size[2] // 2 - radius
    z_end   = matrix_size[2] // 2 + radius + 1

    kernel[x_start:x_end, y_start:y_end, z_start:z_end] = sphere

    # Return spherical kernel in k-space
    return fftn(fftshift(kernel))



def erode_mask(mask, radius):
    erode_size = int(radius * 2 + 1)
    structure = generate_binary_structure(3, 1)
    eroded = binary_erosion(mask, structure=structure, iterations=erode_size)
    return eroded.astype(np.float32)

def run_bgremoval(total_field_path, mask_path):

    img_field = nib.load(total_field_path)
    total_field = img_field.get_fdata()

    img_mask = nib.load(mask_path)
    mask = img_mask.get_fdata()

    # Parse radius input
    radius_list = [5, 4, 3, 2, 1]

    # Zero pad field and mask
    total_field = np.pad(total_field, pad_width=1, mode='constant', constant_values=0)
    mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    matrix_size = total_field.shape

    k_total_field = fftn(total_field)

    RDF = np.zeros(matrix_size, dtype=np.complex64)
    num_radii = len(radius_list)
    
    DiffMask = np.zeros(matrix_size + (num_radii,), dtype=bool)
    Mask_Sharp = np.zeros(matrix_size + (num_radii,), dtype=bool)
    Del_Sharp = np.zeros(matrix_size + (num_radii,), dtype=np.complex64)

    for k, r in enumerate(radius_list):
        #k = index
        #r = radius
        # Generate sphere kernel
        kernel = sphere_kernel(matrix_size, r)

        # Erode mask to avoid edge artifacts
        eroded_mask = erode_mask(mask, r)
        Mask_Sharp[..., k] = eroded_mask > 0

        Del_Sharp[..., k] = kernel

        if k == 0:
            DiffMask[..., k] = Mask_Sharp[..., k]
        else:
            DiffMask[..., k] = np.logical_and(Mask_Sharp[..., k], ~Mask_Sharp[..., k - 1])

        # Apply kernel in k-space and accumulate
        filtered = ifftn(Del_Sharp[..., k] * k_total_field)
        RDF = RDF + DiffMask[..., k] * filtered

    final_mask = Mask_Sharp[..., -1]
    RDF = np.real(RDF * final_mask)

    # Remove padding
    RDF = RDF[1:-1, 1:-1, 1:-1]
    final_mask = final_mask[1:-1, 1:-1, 1:-1]

    RDF = gaussian_filter(RDF, sigma=0.3)
 


    # Finalize and save
    out_img = nib.Nifti1Image(RDF.astype(np.float32), affine=img_field.affine)

    out_path = f"rts_output_bg.nii"
    nib.save(out_img, out_path)
    return out_path
