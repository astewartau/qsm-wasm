import numpy as np
import nibabel as nib
from scipy.fft import fftn, ifftn, fftshift
from scipy.sparse.linalg import lsmr, LinearOperator
import time

# EXPERIMENTAL FILE: using LSMR
# Notes for thesis---------------------------------------------
"""
Using the same threshold as the currently used rts algo gives a super noisy result.
Super low thresholds give a visually good result, but very blurry.
Max_iter doesn't seem to have a big effect.

The currently used rts algo shows artifacts on low threshold (as exptected), but using a higher threshold 
gives a good result. This does essentially cancel out certain useful frequencies/information though...
Try filtering out only the 0 and look at the result...
"""


def dipole_kernel(shape, voxel_size, B0_dir=(0, 0, 1)):
    nx, ny, nz = shape
    dx, dy, dz = voxel_size
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    kz = np.fft.fftfreq(nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    B0_dir = np.asarray(B0_dir)
    B0_dir = B0_dir / np.linalg.norm(B0_dir)

    k_dot_b = KX * B0_dir[0] + KY * B0_dir[1] + KZ * B0_dir[2]
    k2 = KX**2 + KY**2 + KZ**2
    k2[k2 == 0] = np.inf  # avoid divide by zero

    D = (k_dot_b**2 / k2) - 1/3
    D = fftshift(D)
    D[0, 0, 0] = 0
    return D

# ---------------------------------------------
# LinearOperator for LSMR (acts like diag(D))
# ---------------------------------------------
def create_lsmr_operator(D_vals):
    def matvec(x):
        return D_vals * x

    def rmatvec(y):
        return np.conj(D_vals) * y

    n = D_vals.size
    return LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec, dtype=np.complex128)


def run_rts(fieldmap_path, mask_path, output_path="rts_lsmr_output.nii",
                 delta=0.0001, max_iter=30, bdir=(0, 0, 1)):
    
    start_time = time.time()
    print("RTS LSMR: Starting dipole inversion...")

    # Load inputs
    field_nii = nib.load(fieldmap_path)
    mask_nii = nib.load(mask_path)
    fieldmap = field_nii.get_fdata()
    mask = mask_nii.get_fdata()
    affine = field_nii.affine
    voxel_size = field_nii.header.get_zooms()[:3]
    shape = fieldmap.shape

    # Step 1: Build dipole kernel
    D = dipole_kernel(shape, voxel_size, bdir)

    # Step 2: Forward FFT of the fieldmap
    F_field = fftn(fieldmap)

    # Step 3: Mask for well-conditioned region in k-space
    well_mask = np.abs(D) > delta

    # Step 4: Build LSMR operator for masked region
    A_op = create_lsmr_operator(D[well_mask])
    b = F_field[well_mask]

    # Step 5: Solve using LSMR with early stopping
    result = lsmr(A_op, b, maxiter=max_iter)
    x_solution = result[0]

    # Step 6: Fill k-space with solution in well-conditioned region
    chi_k = np.zeros_like(F_field, dtype=np.complex128)
    chi_k[well_mask] = x_solution

    # Step 7: Inverse FFT to get susceptibility map
    chi = ifftn(chi_k).real
    chi_masked = chi * mask  # optional masking in image space

    # Step 8: Save result
    out_img = nib.Nifti1Image(chi_masked.astype(np.float32), affine)
    nib.save(out_img, output_path)

    print(f"RTS LSMR completed in {time.time() - start_time:.2f} seconds")
    return output_path