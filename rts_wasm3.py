# Rapid two-step dipole inversion with sparsity priors (Python translation)
# Based on: Kames et al. (2018), Neuroimage

import numpy as np
from numpy.fft import fftn, ifftn
from scipy.sparse.linalg import LinearOperator, lsmr
import nibabel as nib


def run_rts(f, mask, vsz=None,
        pad=(0, 0, 0),
        Dkernel='k',
        bdir=(0, 0, 1),
        lstol=4,
        delta=0.15,
        mu=1e5,
        rho=10,
        tol=1e-2,
        maxit=20,
        verbose=False,
        tau=0.7,
        gamma=5):
    affine = nib.load(f).affine
    f = nib.load(f).get_fdata()
    mask = nib.load(mask).get_fdata()

    if f.ndim not in (3, 4):
        raise ValueError(f"arrays must be 3d or 4d, got {f.ndim}d")

    if vsz is None:
        vsz = (1.0, 1.0, 1.0)  # default voxel size

    x = np.zeros_like(f)
    return _rts_(x, f, mask, vsz, pad, Dkernel, bdir, lstol, delta,
                mu, rho, tol, maxit, verbose, tau, gamma,affine )


def _grad3d(x, vsz):
    dx = np.zeros_like(x)
    dy = np.zeros_like(x)
    dz = np.zeros_like(x)
    dx[:-1, :, :] = (x[1:, :, :] - x[:-1, :, :]) / vsz[0]
    dy[:, :-1, :] = (x[:, 1:, :] - x[:, :-1, :]) / vsz[1]
    dz[:, :, :-1] = (x[:, :, 1:] - x[:, :, :-1]) / vsz[2]
    return dx, dy, dz


def _grad3d_adj(dx, dy, dz, vsz):
    gx = np.zeros_like(dx)
    gy = np.zeros_like(dy)
    gz = np.zeros_like(dz)
    gx[1:, :, :] = (dx[1:, :, :] - dx[:-1, :, :]) / vsz[0]
    gy[:, 1:, :] = (dy[:, 1:, :] - dy[:, :-1, :]) / vsz[1]
    gz[:, :, 1:] = (dz[:, :, 1:] - dz[:, :, :-1]) / vsz[2]
    return gx + gy + gz


def _shrink(p, lambd):
    mag = np.abs(p)
    return np.where(mag > lambd, p - lambd * np.sign(p), 0.0)


def _rts_(x, f, mask, vsz, pad, Dkernel, bdir, lstol, delta,
          mu, rho, tol, maxit, verbose, tau, gamma, affine):

    T = f.dtype.type
    zeroT = T(0)
    eps = T(tol)
    i_gamma = T(1) / gamma
    delta, mu, rho, gamma, tau = map(T, (delta, mu, rho, gamma, tau))

    sz0 = mask.shape
    sz = tuple(np.array(sz0) + np.array(pad))
    sz_ = sz

    def padfast(x, shape):
        pad_width = [(0, s - x.shape[i]) for i, s in enumerate(shape)]
        return np.pad(x, pad_width, mode='constant')

    def unpad(x, shape):
        slices = tuple(slice(0, s) for s in shape)
        return x[slices]

    D = np.ones(sz_, dtype=T)
    L = np.ones(sz_, dtype=T)
    M = np.where(np.abs(D) > delta, mu, zeroT)
    iA = np.zeros(sz_, dtype=T)

    n_echos = f.shape[3] if f.ndim == 4 else 1

    for t in range(n_echos):
        if verbose and n_echos > 1:
            print(f"Echo: {t + 1}/{n_echos}")

        ft = f[..., t] if f.ndim == 4 else f
        xp = padfast(ft, sz)
        m = padfast(mask.astype(T), sz)

        x0 = np.empty_like(xp)

        def matvec(v):
            return D.flatten() * v

        A = LinearOperator((D.size, D.size), matvec=matvec, rmatvec=matvec, dtype=np.complex64)

        B_hat = fftn(xp)
        res = lsmr(A, B_hat.flatten(), atol=0, btol=0, maxiter=lstol)
        X_hat = res[0].reshape(sz)
        xp = ifftn(X_hat, s=sz).real
        xp *= m

        X_hat = fftn(xp)

        F_hat = np.zeros_like(X_hat)
        for idx in np.ndindex(sz_):
            a = rho * L[idx] + M[idx]
            if a != 0:
                ia = 1. / a
                F_hat[idx] = ia * M[idx] * X_hat[idx]
                iA[idx] = rho * ia

        px = np.zeros_like(xp)
        py = np.zeros_like(xp)
        pz = np.zeros_like(xp)

        vx = np.zeros_like(xp)
        vy = np.zeros_like(xp)
        vz = np.zeros_like(xp)

        nr = np.finfo(T).max

        for i in range(maxit):
            x0[...] = xp.copy()

            dx, dy, dz = _grad3d(xp, vsz)

            yx = _shrink(dx + px, 1. / rho)
            yy = _shrink(dy + py, 1. / rho)
            yz = _shrink(dz + pz, 1. / rho)

            px += dx - yx
            py += dy - yy
            pz += dz - yz

            vx = yx - px
            vy = yy - py
            vz = yz - pz

            g_adj = _grad3d_adj(vx, vy, vz, vsz)

            X_hat = fftn(g_adj)
            X_hat = iA * X_hat + F_hat
            xp = ifftn(X_hat, s=sz)

            ndx = np.linalg.norm(xp - x0)
            nx = np.linalg.norm(xp)

            if verbose:
                print(f"{i + 1}/{maxit}\t    {ndx / nx:.4e}")

            if ndx < eps * nx:
                break

            nr0 = nr
            res_norm = np.linalg.norm(np.stack([dx - yx, dy - yy, dz - yz]))
            nr = res_norm

            if nr > tau * nr0:
                rho_old = rho
                rho *= gamma
                iA *= rho / rho_old
                px *= i_gamma
                py *= i_gamma
                pz *= i_gamma

        if f.ndim == 4:
            x[..., t] = unpad(xp, sz0)
        else:
            x[...] = unpad(xp, sz0)

    
    out_img = nib.Nifti1Image(x.astype(np.float32), affine)
    nib.save(out_img, "output.nii")
