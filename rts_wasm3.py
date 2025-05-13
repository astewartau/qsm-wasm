import numpy as np
import nibabel as nib



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

def grad(x, voxel_size):
    dx = np.diff(x, axis=0, append=x[-1:,:,:]) / voxel_size[0]
    dy = np.diff(x, axis=1, append=x[:,-1:,:]) / voxel_size[1]
    dz = np.diff(x, axis=2, append=x[:,:,-1:]) / voxel_size[2]
    return dx, dy, dz

def div(dx, dy, dz, voxel_size):
    dxx = np.diff(dx, axis=0, prepend=dx[0:1,:,:]) / voxel_size[0]
    dyy = np.diff(dy, axis=1, prepend=dy[:,0:1,:]) / voxel_size[1]
    dzz = np.diff(dz, axis=2, prepend=dz[:,:,0:1]) / voxel_size[2]
    return dxx + dyy + dzz

def soft_shrink(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

def run_rts_nib(fieldmap_path, mask_path, output_path="rts_output.nii", vsz=None, bdir=(0, 0, 1), delta=0.15,
                mu=1e5, rho=10, tol=1e-2, maxit=20):

    # Load files
    field_nii = nib.load(fieldmap_path)
    mask_nii = nib.load(mask_path)

    fieldmap = field_nii.get_fdata()
    mask = mask_nii.get_fdata()
    affine = field_nii.affine

    if vsz is None:
        vsz = field_nii.header.get_zooms()[:3]

    shape = fieldmap.shape

    # Create dipole kernel
    D = dipole_kernel(shape, vsz, bdir)

    #Well-conditioned inversion (simple division)
    F_field = np.fft.fftn(fieldmap)
    D_inv = np.zeros_like(D)
    D_inv[np.abs(D) > delta] = 1.0 / D[np.abs(D) > delta]
    chi_k = F_field * D_inv
    chi = np.fft.ifftn(chi_k).real


    #NOW SECOND STEP OF RTS

    #Compute residual
    F_chi = np.fft.fftn(chi)
    field_simulated = np.fft.ifftn(D * F_chi).real
    residual = (fieldmap - field_simulated) * mask

    #ADMM solve for residual (TV regularization)
    chi_residual = np.zeros_like(residual)
    px = np.zeros_like(residual)
    py = np.zeros_like(residual)
    pz = np.zeros_like(residual)

    for it in range(maxit):
        dx, dy, dz = grad(chi_residual, vsz)
        dx = soft_shrink(dx + px, 1/rho)
        dy = soft_shrink(dy + py, 1/rho)
        dz = soft_shrink(dz + pz, 1/rho)

        px += grad(chi_residual, vsz)[0] - dx
        py += grad(chi_residual, vsz)[1] - dy
        pz += grad(chi_residual, vsz)[2] - dz

        div_p = div(dx - px, dy - py, dz - pz, vsz)

        rhs = residual + (mu/rho) * div_p
        rhs_k = np.fft.fftn(rhs)
        denom = D * np.conj(D) + mu/rho
        chi_residual = np.real(np.fft.ifftn(rhs_k / (denom + 1e-8))) * mask

        if np.linalg.norm(div_p) / (np.linalg.norm(chi_residual) + 1e-8) < tol:
            break

    # Step 4: Combine
    chi_final = (chi + chi_residual) * mask

    # Save result
    nii_out = nib.Nifti1Image(chi_final.astype(np.float32), affine)
    nib.save(nii_out, output_path)

    return output_path
