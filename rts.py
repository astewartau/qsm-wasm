import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load NIfTI image
nii = nib.load("fieldmap-local.nii")

# Extract the image data (as NumPy array)
fieldmap = nii.get_fdata(dtype=np.float32)  #instead of the standard float64 for speed. Will have to compare the two for speed vs quality

#Extract the mask image 
nii_mask = nib.load("mask.nii")
mask = nii_mask.get_fdata() # make sure it's boolean

# Define voxel size and B0 direction
vsz = nii.header.get_zooms()[:3]   # dx, dy, dz (voxel sizes in mm) from fieldmap header

bdir = (0, 0, 1) #if BIDS, you can obtain this info from the json metadata. Currently only support for NiFti



#Input: fieldmap, mask, vsz, bdir

def dipole_kernel(shape, vsz, bdir=(0, 0, 1)):
    """
    Generate dipole kernel in k-space.
    
    Parameters:
        shape: tuple of ints (nx, ny, nz)
        vsz: tuple of floats (dx, dy, dz) in mm
        bdir: tuple of floats (bx, by, bz), default is (0, 0, 1)
    
    Returns:
        dipole_kernel: np.ndarray, complex128, shape = shape
    """
    nx, ny, nz = shape
    dx, dy, dz = vsz

    # Compute physical FOV in mm
    FOVx = nx * dx
    FOVy = ny * dy
    FOVz = nz * dz

    # Construct k-space grids (centered around 0)
    kx = np.arange(-np.ceil((nx-1)/2.0), np.floor((nx-1)/2.0)+1) * 1.0 / FOVx
    ky = np.arange(-np.ceil((ny-1)/2.0), np.floor((ny-1)/2.0)+1) * 1.0 / FOVy
    kz = np.arange(-np.ceil((nz-1)/2.0), np.floor((nz-1)/2.0)+1) * 1.0 / FOVz

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')  # match Julia's indexing

    # Normalize B-field direction
    bdir = np.asarray(bdir, dtype=np.float64)
    bdir = bdir / np.linalg.norm(bdir)

    # Dot product (k · bdir) and magnitude
    k_dot_b = KX * bdir[0] + KY * bdir[1] + KZ * bdir[2]
    k2 = KX**2 + KY**2 + KZ**2

    # Dipole formula
    D = (k_dot_b**2 / (k2 + 1e-8)) - 1/3

    # Optional: Shift center of k-space to (0,0,0)
    D = np.fft.ifftshift(D)

    # Set DC term to zero
    D[0, 0, 0] = 0

    return D.astype(np.complex128)


"""
def dipole_kernel2(shape, vsz, bdir=(0, 0, 1)):
    nx, ny, nz = shape
    dx, dy, dz = vsz
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    kz = np.fft.fftfreq(nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    
    k2 = KX**2 + KY**2 + KZ**2
    dot = KX*bdir[0] + KY*bdir[1] + KZ*bdir[2]
    D = (dot**2 / (k2 + 1e-8)) - 1/3
    D[k2 == 0] = 0
    return D
"""




def rts(fieldmap, mask, vsz=(1.0, 1.0, 1.0), bdir=(0, 0, 1), delta=0.15):
    shape = fieldmap.shape
    D = dipole_kernel(shape, vsz, bdir)
    
    F_field = np.fft.fftn(fieldmap * mask)
    D_inv = np.zeros_like(D)

    #invert the well-conditioned voxels and set the rest on zero
    D_inv[np.abs(D) > delta] = 1.0 / D[np.abs(D) > delta] 
    
    #calculate the susceptibility map (in k-space)
    chi_k = F_field * D_inv

    #inverse fourier to get the final susceptibility map
    chi = np.fft.ifftn(chi_k).real
    return chi * mask




# --- Run inversion
chi = rts(fieldmap, mask, vsz=vsz, bdir=bdir)

# --- Save output
nii_out = nib.Nifti1Image(chi.astype(np.float32), affine=nii.affine)
nib.save(nii_out, "rts_output_step2.nii")