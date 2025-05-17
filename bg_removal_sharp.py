import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter
from scipy.special import gamma, jn


def getFourierDomain(x):
    return np.fft.fftn(np.fft.ifftshift(x))


def getImgDomain(x_f):
    return np.fft.fftshift(np.fft.ifftn(x_f))


def convolve(img, kernel_f):
    return np.real(getImgDomain(kernel_f * getFourierDomain(img)))


def inverseTransform(kImg, FOV):
    return float(kImg.size) / np.prod(FOV) * np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kImg)))


def phantomGrid3D(shape, FOV):
    sx, sy, sz = shape
    FOVx, FOVy, FOVz = FOV
    kx, ky, kz = np.mgrid[
        -np.floor(sx / 2.0):-np.floor(sx / 2.0) + sx,
        -np.floor(sy / 2.0):-np.floor(sy / 2.0) + sy,
        -np.floor(sz / 2.0):-np.floor(sz / 2.0) + sz
    ]
    return kx / FOVx, ky / FOVy, kz / FOVz


def sincn(n, r):
    r = np.asarray(r)
    result = np.empty_like(r, dtype=np.float64)
    result[r == 0] = 0.5
    nonzero = r != 0
    result[nonzero] = (
        0.5 * gamma(1 + n / 2.0) * jn(n / 2.0, r[nonzero]) /
        (r[nonzero] / 2.0) ** (n / 2.0)
    )
    return result


def sincnForImgSpaceUnitCircle(n, radMap):
    return sincn(n, radMap * np.pi) * np.pi / n


def scSincn(n, radMap, scale):
    return scale ** n * sincnForImgSpaceUnitCircle(n, scale * radMap)


def kSphere(kx, ky, kz, radius, amplitude=1):
    radMap = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    return amplitude * scSincn(3, radMap, 2 * radius)


def sharp_background_removal(img, mask, FOV, sphereDiam=5.0, threshold=0.05):
    kx, ky, kz = phantomGrid3D(img.shape, FOV)
    vol = 4.0 / 3 * np.pi * (sphereDiam / 2) ** 3
    signal = 1.0 / vol * np.prod(FOV) / np.prod(img.shape)

    sk = getFourierDomain(np.real(inverseTransform(
        kSphere(kx, ky, kz, radius=sphereDiam / 2, amplitude=signal), FOV)))

    SMV = 1.0 - sk
    SMV[np.abs(SMV) < threshold] = 0

    one_over_SMV = np.zeros_like(SMV)
    one_over_SMV[SMV != 0] = 1.0 / SMV[SMV != 0]

    mask = mask.astype(bool)
    img_masked = img * mask

    Bint_corrupted = convolve(img_masked, SMV)
    mask_erode = convolve((~mask).astype(float), sk) < 0.005
    Bint = convolve(Bint_corrupted * mask_erode, one_over_SMV) * mask_erode

    return Bint


def run_bgremoval(data_path, mask_path, output_path="rts_output.nii"):
    img_nii = nib.load(data_path)
    mask_nii = nib.load(mask_path)

    img = img_nii.get_fdata()
    mask = mask_nii.get_fdata().astype(bool)

    shape = img.shape
    voxel_size = img_nii.header.get_zooms()[:3]
    FOV = [shape[i] * voxel_size[i] for i in range(3)]

    result = sharp_background_removal(img, mask, FOV)

    out_img = nib.Nifti1Image(result.astype(np.float32), affine=img_nii.affine)
    nib.save(out_img, output_path)

    return output_path
