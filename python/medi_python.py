"""
MEDI (Morphology Enabled Dipole Inversion) and SMV Background Removal
for QSM-WASM

Based on: Liu et al. MRM 2012;68(4):1125-37
"""

import numpy as np

# =============================================================================
# SMV (Spherical Mean Value) Background Removal
# =============================================================================

def create_sphere_kernel_kspace(shape, voxel_size, radius):
    """
    Create spherical mean value kernel in k-space (for convolution via FFT).

    Parameters
    ----------
    shape : tuple
        (nx, ny, nz) image dimensions
    voxel_size : tuple
        (dx, dy, dz) voxel dimensions in mm
    radius : float
        Sphere radius in mm

    Returns
    -------
    S : ndarray
        Sphere kernel in k-space (real, normalized)
    """
    nx, ny, nz = shape
    dx, dy, dz = voxel_size
    r2 = radius * radius

    # Create coordinate grids in real space
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)

    # Wrap coordinates around center (for FFT convention)
    x = np.where(x <= nx // 2, x, x - nx) * dx
    y = np.where(y <= ny // 2, y, y - ny) * dy
    z = np.where(z <= nz // 2, z, z - nz) * dz

    # Create 3D grids
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Sphere mask
    sphere = (X*X + Y*Y + Z*Z <= r2).astype(np.float64)

    # Normalize so sum = 1
    sphere_sum = np.sum(sphere)
    if sphere_sum > 0:
        sphere /= sphere_sum

    # FFT to get k-space kernel
    S = np.real(np.fft.fftn(sphere))

    return S


def smv_background_removal(fieldmap, mask, voxel_size, radius=5.0,
                           progress_callback=None):
    """
    SMV (Spherical Mean Value) background field removal.

    Applies a high-pass filter using spherical mean value convolution
    to remove large-scale background fields.

    Parameters
    ----------
    fieldmap : ndarray
        Input field map (e.g., B0 map in Hz)
    mask : ndarray
        Binary brain mask
    voxel_size : tuple
        Voxel dimensions in mm
    radius : float
        Sphere radius in mm (default: 5.0)
    progress_callback : callable, optional
        Progress callback function(step, total)

    Returns
    -------
    local_field : ndarray
        Background-removed local field
    eroded_mask : ndarray
        Eroded mask after SMV
    """
    shape = fieldmap.shape

    if progress_callback:
        progress_callback(1, 3)

    print(f"SMV background removal with radius={radius:.1f}mm")

    # Create sphere kernel in k-space
    S = create_sphere_kernel_kspace(shape, voxel_size, radius)

    if progress_callback:
        progress_callback(2, 3)

    # High-pass filter: HP = 1 - S (removes low frequencies)
    HP = 1.0 - S

    # Apply high-pass filter to field
    F = np.fft.fftn(fieldmap)
    local_field = np.real(np.fft.ifftn(HP * F))

    # Erode mask: keep voxels where SMV(mask) > threshold
    # This removes edge voxels where the sphere extends outside the mask
    M = np.fft.fftn(mask.astype(np.float64))
    eroded = np.real(np.fft.ifftn(S * M))
    eroded_mask = eroded > 0.999

    # Apply eroded mask
    local_field = local_field * eroded_mask

    if progress_callback:
        progress_callback(3, 3)

    mask_coverage = np.sum(eroded_mask) / eroded_mask.size * 100
    print(f"SMV complete. Eroded mask: {np.sum(eroded_mask)} voxels ({mask_coverage:.1f}%)")

    return local_field, eroded_mask


# =============================================================================
# MEDI Helper Functions
# =============================================================================

def fgrad(chi, voxel_size):
    """
    Forward gradient operator with Neumann boundary conditions.

    Parameters
    ----------
    chi : ndarray
        3D scalar field
    voxel_size : tuple
        (dx, dy, dz) voxel dimensions

    Returns
    -------
    grad : ndarray
        4D gradient field (nx, ny, nz, 3) with x, y, z components
    """
    dx, dy, dz = voxel_size
    nx, ny, nz = chi.shape

    grad = np.zeros((*chi.shape, 3), dtype=chi.dtype)

    # Forward differences with Neumann BC (replicate at boundary)
    # X gradient
    grad[:-1, :, :, 0] = (chi[1:, :, :] - chi[:-1, :, :]) / dx
    grad[-1, :, :, 0] = 0  # Neumann: zero derivative at boundary

    # Y gradient
    grad[:, :-1, :, 1] = (chi[:, 1:, :] - chi[:, :-1, :]) / dy
    grad[:, -1, :, 1] = 0

    # Z gradient
    grad[:, :, :-1, 2] = (chi[:, :, 1:] - chi[:, :, :-1]) / dz
    grad[:, :, -1, 2] = 0

    return grad


def bdiv(grad_field, voxel_size):
    """
    Backward divergence operator (negative adjoint of fgrad).

    Parameters
    ----------
    grad_field : ndarray
        4D gradient field (nx, ny, nz, 3)
    voxel_size : tuple
        (dx, dy, dz) voxel dimensions

    Returns
    -------
    div : ndarray
        3D divergence field
    """
    dx, dy, dz = voxel_size
    gx = grad_field[:, :, :, 0]
    gy = grad_field[:, :, :, 1]
    gz = grad_field[:, :, :, 2]

    nx, ny, nz = gx.shape

    # Backward differences with Dirichlet BC (zero at boundary)
    div = np.zeros_like(gx)

    # X component
    div[0, :, :] = gx[0, :, :] / dx
    div[1:-1, :, :] += (gx[1:-1, :, :] - gx[:-2, :, :]) / dx
    div[-1, :, :] += -gx[-2, :, :] / dx

    # Y component
    div[:, 0, :] += gy[:, 0, :] / dy
    div[:, 1:-1, :] += (gy[:, 1:-1, :] - gy[:, :-2, :]) / dy
    div[:, -1, :] += -gy[:, -2, :] / dy

    # Z component
    div[:, :, 0] += gz[:, :, 0] / dz
    div[:, :, 1:-1] += (gz[:, :, 1:-1] - gz[:, :, :-2]) / dz
    div[:, :, -1] += -gz[:, :, -2] / dz

    return div


def create_dipole_kernel(shape, voxel_size, bdir=(0, 0, 1)):
    """
    Create dipole kernel in k-space.

    D(k) = 1/3 - (k·B)² / |k|²

    Parameters
    ----------
    shape : tuple
        Image dimensions
    voxel_size : tuple
        Voxel dimensions in mm
    bdir : tuple
        B0 field direction (will be normalized)

    Returns
    -------
    D : ndarray
        Dipole kernel in k-space
    """
    nx, ny, nz = shape
    dx, dy, dz = voxel_size

    # K-space frequency grids
    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kz = np.fft.fftfreq(nz, dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    # Normalize B field direction
    bdir = np.array(bdir, dtype=np.float64)
    bdir = bdir / np.linalg.norm(bdir)

    # k dot B
    k_dot_b = KX * bdir[0] + KY * bdir[1] + KZ * bdir[2]

    # |k|²
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1e-12  # Avoid division by zero at DC

    # Dipole kernel: D = 1/3 - (k·B)²/|k|²
    D = 1.0/3.0 - (k_dot_b**2) / k2

    # Set DC to zero
    D[0, 0, 0] = 0

    return D


def gradient_mask(magnitude, mask, percentage=0.9):
    """
    Create anatomy-aware edge weighting mask.

    Identifies edges in the magnitude image where TV regularization
    should be reduced to preserve anatomical boundaries.

    Parameters
    ----------
    magnitude : ndarray
        Anatomical magnitude image
    mask : ndarray
        Binary brain mask
    percentage : float
        Fraction of voxels to consider as non-edge (0-1)
        Higher values = more edges preserved

    Returns
    -------
    wG : ndarray
        Edge weighting mask (0 at edges, 1 elsewhere)
    """
    # Compute gradient magnitude of anatomical image
    gy, gx, gz = np.gradient(magnitude)
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # Get values inside mask
    masked_grad = grad_mag[mask > 0]

    if len(masked_grad) == 0:
        return np.ones_like(magnitude)

    # Find threshold at given percentile
    threshold = np.percentile(masked_grad, percentage * 100)

    # Create edge mask: 0 at strong edges, 1 elsewhere
    # This reduces regularization at anatomical boundaries
    wG = np.ones_like(magnitude)
    wG[grad_mag > threshold] = 0

    # Only apply within mask
    wG = wG * mask

    return wG


def cg_solve(A_operator, b, x0=None, tol=0.01, max_iter=100,
             progress_callback=None):
    """
    Conjugate Gradient solver for symmetric positive-definite systems.

    Solves A @ x = b where A is given as an operator (function).

    Parameters
    ----------
    A_operator : callable
        Function that computes A @ x
    b : ndarray
        Right-hand side
    x0 : ndarray, optional
        Initial guess (default: zeros)
    tol : float
        Convergence tolerance on residual norm
    max_iter : int
        Maximum iterations
    progress_callback : callable, optional
        Progress callback function(iteration, max_iter)

    Returns
    -------
    x : ndarray
        Solution
    """
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    r = b - A_operator(x)
    p = r.copy()
    rsold = np.sum(r * r)

    b_norm = np.sqrt(np.sum(b * b))
    if b_norm < 1e-12:
        return x

    for i in range(max_iter):
        if progress_callback and i % 10 == 0:
            progress_callback(i, max_iter)

        Ap = A_operator(p)
        pAp = np.sum(p * Ap)

        if np.abs(pAp) < 1e-12:
            break

        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = np.sum(r * r)

        # Check convergence
        if np.sqrt(rsnew) / b_norm < tol:
            break

        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x


# =============================================================================
# MEDI Algorithm
# =============================================================================

def medi_l1(local_field, mask, magnitude, voxel_size,
            lambda_=1000, bdir=(0, 0, 1),
            max_iter=10, cg_max_iter=100, cg_tol=0.01,
            tol_norm_ratio=0.1, edge_percentage=0.9,
            merit=False, progress_callback=None):
    """
    MEDI (Morphology Enabled Dipole Inversion) L1 algorithm.

    Solves the regularized dipole inversion problem:
        min ||W * (exp(i*phi_measured) - exp(i*phi_forward))||² + lambda * TV(chi)

    using iterative Gauss-Newton optimization with anatomy-aware TV regularization.

    Parameters
    ----------
    local_field : ndarray
        Local field map after background removal (in Hz or rad)
    mask : ndarray
        Binary brain mask
    magnitude : ndarray
        Magnitude image for edge-aware regularization
    voxel_size : tuple
        Voxel dimensions in mm
    lambda_ : float
        Regularization weight (default: 1000)
    bdir : tuple
        B0 field direction (default: (0, 0, 1) for axial)
    max_iter : int
        Maximum Gauss-Newton iterations (default: 10)
    cg_max_iter : int
        Maximum CG iterations per GN step (default: 100)
    cg_tol : float
        CG convergence tolerance (default: 0.01)
    tol_norm_ratio : float
        GN stopping criterion: ||dx|| / ||chi|| (default: 0.1)
    edge_percentage : float
        Percentile for edge detection (default: 0.9)
    merit : bool
        Enable merit function for iterative noise reweighting (default: False)
    progress_callback : callable, optional
        Progress callback function(iteration, max_iter)

    Returns
    -------
    chi : ndarray
        Susceptibility map
    """
    shape = local_field.shape

    print(f"MEDI L1: lambda={lambda_}, max_iter={max_iter}, edge%={edge_percentage}")

    # Create dipole kernel
    D = create_dipole_kernel(shape, voxel_size, bdir)
    D_conj = np.conj(D)
    D2 = D * D_conj  # |D|²

    # Forward dipole convolution operator
    def Dconv(x):
        return np.real(np.fft.ifftn(D * np.fft.fftn(x)))

    # Adjoint dipole convolution (D* = D for real symmetric kernel)
    def Dconv_adj(x):
        return np.real(np.fft.ifftn(D_conj * np.fft.fftn(x)))

    # Data weighting (uniform for simplicity, could use N_std)
    m = mask.astype(np.float64)

    # Initial phase term
    b0 = m * np.exp(1j * local_field)

    # Anatomy-aware gradient weighting
    wG = gradient_mask(magnitude, mask, edge_percentage)
    wG_expanded = wG[:, :, :, np.newaxis]  # For broadcasting with gradient

    # Initialize susceptibility
    chi = np.zeros(shape, dtype=np.float64)

    eps = 1e-6  # Small constant for numerical stability

    print("Starting Gauss-Newton iterations...")

    for outer_iter in range(max_iter):
        if progress_callback:
            progress_callback(outer_iter + 1, max_iter)

        # =====================================================================
        # Step 1: Compute TV edge-aware weighting Vr
        # =====================================================================
        grad_chi = fgrad(chi, voxel_size)
        # Weighted gradient magnitude
        wG_grad = wG_expanded * grad_chi
        grad_mag_sq = np.sum(wG_grad**2, axis=-1)
        Vr = 1.0 / np.sqrt(grad_mag_sq + eps)

        # =====================================================================
        # Step 2: Phase residual
        # =====================================================================
        phi_forward = Dconv(chi)
        w = m * np.exp(1j * phi_forward)

        # =====================================================================
        # Step 3: Define operators for CG
        # =====================================================================

        # Regularization operator: reg0(dx) = div(wG * Vr * wG * grad(dx))
        def reg0(dx):
            g = fgrad(dx, voxel_size)
            weighted_g = wG_expanded * Vr[:, :, :, np.newaxis] * wG_expanded * g
            return bdiv(weighted_g, voxel_size)

        # Data fidelity operator: fidelity(dx) = D' * (|w|² * D * dx)
        def fidelity(dx):
            Ddx = Dconv(dx)
            w_sq = np.abs(w)**2
            return Dconv_adj(w_sq * Ddx)

        # Combined system matrix: A = reg0 + 2*lambda*fidelity
        def A_operator(dx):
            return reg0(dx) + 2 * lambda_ * fidelity(dx)

        # =====================================================================
        # Step 4: Compute RHS
        # =====================================================================
        # RHS = reg0(chi) + 2*lambda*D'*Im(conj(w)*(w - b0))
        reg0_chi = reg0(chi)

        # Phase difference term
        residual = w - b0
        # Im(conj(w) * residual) = Im(conj(w) * w - conj(w) * b0)
        #                        = Im(|w|² - conj(w)*b0)
        #                        = -Im(conj(w) * b0)
        phase_term = np.imag(np.conj(w) * residual)

        rhs = reg0_chi + 2 * lambda_ * Dconv_adj(phase_term)

        # =====================================================================
        # Step 5: Solve CG and update
        # =====================================================================
        dx = cg_solve(A_operator, -rhs, tol=cg_tol, max_iter=cg_max_iter)

        chi = chi + dx

        # =====================================================================
        # Step 6: Check convergence
        # =====================================================================
        dx_norm = np.linalg.norm(dx)
        chi_norm = np.linalg.norm(chi)

        rel_change = dx_norm / (chi_norm + eps)

        # Compute costs for monitoring
        wres = m * np.exp(1j * Dconv(chi)) - b0
        cost_data = np.linalg.norm(wres)
        cost_reg = np.sum(np.abs(wG_expanded * fgrad(chi, voxel_size)))

        print(f"  Iter {outer_iter + 1}/{max_iter}: "
              f"rel_change={rel_change:.4f}, "
              f"data_cost={cost_data:.2f}, reg_cost={cost_reg:.2f}")

        if rel_change < tol_norm_ratio:
            print(f"  Converged at iteration {outer_iter + 1}")
            break

        # =====================================================================
        # Step 7: Merit function (optional iterative reweighting)
        # =====================================================================
        if merit:
            # Reweight by residual magnitude
            wres_centered = wres - np.mean(wres[mask > 0])
            wres_std = np.std(np.abs(wres_centered[mask > 0]))
            factor = 6 * wres_std

            if factor > eps:
                wres_norm = np.abs(wres_centered) / factor
                wres_norm = np.maximum(wres_norm, 1.0)

                # Update data weighting
                m = mask.astype(np.float64) / wres_norm
                m[~(mask > 0)] = 0
                b0 = m * np.exp(1j * local_field)

    # Apply mask to final result
    chi = chi * mask

    print(f"MEDI complete. Chi range: [{chi[mask > 0].min():.4f}, {chi[mask > 0].max():.4f}]")

    return chi


# =============================================================================
# Entry Point for Worker
# =============================================================================

def run_medi(local_field, mask, magnitude, voxel_size,
             lambda_=1000, max_iter=10, cg_max_iter=100, cg_tol=0.01,
             edge_percentage=0.9, merit=False, bdir=(0, 0, 1),
             progress_callback=None):
    """
    Main entry point for MEDI dipole inversion.

    This is called from the JavaScript worker.
    """
    return medi_l1(
        local_field=local_field,
        mask=mask,
        magnitude=magnitude,
        voxel_size=voxel_size,
        lambda_=lambda_,
        bdir=bdir,
        max_iter=max_iter,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
        edge_percentage=edge_percentage,
        merit=merit,
        progress_callback=progress_callback
    )
