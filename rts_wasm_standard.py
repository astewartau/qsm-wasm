import numpy as np
import nibabel as nib
import time

def dipole_kernel(shape, vsz, bdir=(0, 0, 1)):
    nx, ny, nz = shape
    dx, dy, dz = vsz
    FOVx = nx * dx
    FOVy = ny * dy
    FOVz = nz * dz

    kx = np.arange(-np.ceil((nx - 1) / 2), np.floor((nx - 1) / 2) + 1) / FOVx
    ky = np.arange(-np.ceil((ny - 1) / 2), np.floor((ny - 1) / 2) + 1) / FOVy
    kz = np.arange(-np.ceil((nz - 1) / 2), np.floor((nz - 1) / 2) + 1) / FOVz
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    bdir = np.asarray(bdir, dtype=np.float64)
    bdir /= np.linalg.norm(bdir)

    k_dot_b = KX * bdir[0] + KY * bdir[1] + KZ * bdir[2]
    k2 = KX**2 + KY**2 + KZ**2

    D = (k_dot_b**2 / (k2 + 1e-8)) - 1/3
    D = np.fft.ifftshift(D)
    D[0, 0, 0] = 0
    return D.astype(np.complex128)

def run_rts(fieldmap_path, mask_path, output_path="rts_output.nii", vsz=None, bdir=(0, 0, 1), delta=0.15):


    start_time = time.time()
    print("Dipole inversion started...")


    #load files
    field_nii = nib.load(fieldmap_path)
    mask_nii = nib.load(mask_path)

    #extract fieldmap
    fieldmap = field_nii.get_fdata()
    #extract mask
    mask = mask_nii.get_fdata()

    affine = field_nii.affine
    if vsz is None:
        vsz = field_nii.header.get_zooms()[:3]

    shape = fieldmap.shape

    #create dipole kernel
    D = dipole_kernel(shape, vsz, bdir)

    #convert to k-space
    F_field = np.fft.fftn(fieldmap)

    D_inv = np.zeros_like(D)
    D_inv[np.abs(D) > delta] = 1.0 / D[np.abs(D) > delta]


    
    chi_k = F_field * D_inv
    chi = np.fft.ifftn(chi_k).real

    chi_out = -chi * mask
    nii_out = nib.Nifti1Image(chi_out.astype(np.float32), affine)
    nib.save(nii_out, output_path)


    elapsed = time.time() - start_time
    print(f"Dipole inversion completed in {elapsed:.3f} seconds")

    return output_path
