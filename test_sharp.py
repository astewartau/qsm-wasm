# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:28:26 2016

@author: Alan

Rough code for doing sharp in fourier domain.
"""

from scipy.special import gamma, jn
import numpy as np
from multiprocessing import Pool
import pyfftw
from scipy.ndimage import binary_dilation, binary_fill_hoales, gaussian_filter
import nibabel as nib
import os

nthreads = 16
np.fft.fftn = pyfftw.interfaces.numpy_fft.fftn
np.fft.ifftn = pyfftw.interfaces.numpy_fft.ifftn

def getFourierDomain(x):
    return np.fft.fftn(np.fft.ifftshift(x), threads=nthreads)

def getImgDomain(x_f):
    return np.fft.fftshift(np.fft.ifftn(x_f, threads=nthreads))

def convolve(img, kernel_f, returnFourier=False, imgInFourier=False):
    if imgInFourier:
        if returnFourier:
            return kernel_f * img
        else:
            return np.real(getImgDomain(kernel_f * img))
    else:
        if returnFourier:
            return kernel_f * getFourierDomain(img)
        else:
            return np.real(getImgDomain(kernel_f * getFourierDomain(img)))

def inverseTransform(kImg, FOV):
    if kImg.ndim != len(FOV):
        raise ValueError("Number of kImg dimensions must match number of dimensions in FOV!")
    return float(kImg.size) / np.array(FOV).prod() * np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kImg)))

def forwardTransform(img, FOV):
    if img.ndim != len(FOV):
        raise ValueError("Number of img dimensions must match number of dimensions in FOV!")
    return np.array(FOV).prod() / float(img.size) * np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))

def phantomGrid3D(numXSamples, FOVx, numYSamples, FOVy, numZSamples, FOVz):
    FOVx, FOVy, FOVz = float(FOVx), float(FOVy), float(FOVz)
    kx, ky, kz = np.mgrid[
        -np.floor(numXSamples/2.0):-np.floor(numXSamples/2.0)+numXSamples,
        -np.floor(numYSamples/2.0):-np.floor(numYSamples/2.0)+numYSamples,
        -np.floor(numZSamples/2.0):-np.floor(numZSamples/2.0)+numZSamples
    ]
    return kx/FOVx, ky/FOVy, kz/FOVz

def kSpaceShift(kimg, k, shift):
    if np.sum(np.abs(np.array(shift))) != 0:
        phase = -(1j * 2 * np.pi * sum(k[i] * shift[i] for i in range(len(k))))
        kimg *= np.exp(phase)
    return kimg

def sincn(n, r):
    if r == 0:
        return 0.5
    else:
        return 0.5 * gamma(1 + n/2.0) * jn(n/2.0, r) / (r/2.0)**(n/2.0)

sincn = np.vectorize(sincn)

def sincnForImgSpaceUnitCircle(n, radMap):
    return sincn(n, radMap * np.pi) * np.pi / n

def scSincn(n, radMap, scale):
    scale = float(scale)
    return scale**n * sincnForImgSpaceUnitCircle(n, scale * radMap)

def _kSphereHelper(args):      
    i, radMapPart, ksphererad = args
    return scSincn(3, radMapPart, 2 * ksphererad)

def kSphere(kx, ky, kz, radius, amplitude=1, xShift=0, yShift=0, zShift=0):
    poolsz = 8
    chunksize = int(np.ceil(float(kz.shape[2]) / poolsz))

    radMap = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    args = [
        (i, radMap[:, :, i * chunksize:min((i + 1) * chunksize, radMap.shape[2])], radius)
        for i in range(poolsz)
    ]

    with Pool(poolsz) as p:
        vals = p.map(_kSphereHelper, args)

    vals = np.concatenate(vals, axis=2)
    return amplitude * kSpaceShift(vals, [kx, ky, kz], [xShift, yShift, zShift])


class SpherePhantom():
    def __init__(self, phantomDiam, phantomIntensity=1, shift=[0.0, 0.0, 0.0]):
        self.phantomRad = phantomDiam / 2.0
        self.phantomIntensity = phantomIntensity
        self.shift = shift

    def sample(self, kx, ky, kz):
        return kSphere(kx, ky, kz, radius=self.phantomRad, amplitude=self.phantomIntensity,
                       xShift=self.shift[0], yShift=self.shift[1], zShift=self.shift[2])

def sharp(img, FOV, mask, sphereDiam=5.0, threshold=0.05):
    FOVx, FOVy, FOVz = FOV
    acquisitionX, acquisitionY, acquisitionZ = img.shape

    kx, ky, kz = phantomGrid3D(acquisitionX, FOVx, acquisitionY, FOVy, acquisitionZ, FOVz)

    vol = 4.0/3 * np.pi * (sphereDiam/2)**3
    signal = 1.0/vol * float(FOVx * FOVy * FOVz) / float(acquisitionX * acquisitionY * acquisitionZ)
    sphere = SpherePhantom(sphereDiam, phantomIntensity=signal)
    sphereKData = sphere.sample(kx, ky, kz)
    sphereImgData = np.real(inverseTransform(sphereKData, (FOVx, FOVy, FOVz)))
    sk = getFourierDomain(sphereImgData)

    SMV = (1.0 - sk)
    SMV[np.abs(SMV) < threshold] = 0
    one_over_SMV = np.zeros_like(SMV)
    one_over_SMV[SMV != 0] = 1.0 / SMV[SMV != 0]

    global img_reliable, mask_reliable
    img_reliable = img * mask
    mask_reliable = mask

    Bint_corrupted = convolve(img_reliable, SMV)
    mask_erode = convolve(np.invert(mask_reliable), sk) < .005
    Bint_corrupted *= mask_erode
    Bint = convolve(Bint_corrupted, one_over_SMV) * mask_erode

    return Bint

if __name__ == "__main__":
    data_loc = os.path.abspath('./test_algo/unwrapped.nii')
    mask_loc = os.path.abspath('./test_algo/mask.nii')

    imgObj = nib.load(data_loc)
    img = imgObj.get_fdata()

    mask = nib.load(mask_loc).get_fdata()
    mask = mask.astype('bool')

    acquisitionX, acquisitionY, acquisitionZ = imgObj.header['dim'][1:4]
    szX, szY, szZ = imgObj.header['pixdim'][1:4]
    FOVx, FOVy, FOVz = szX * acquisitionX, szY * acquisitionY, szZ * acquisitionZ

    lfs = sharp(img, (FOVx, FOVy, FOVz), mask)
    affine = imgObj.affine
    nii_out = nib.Nifti1Image(lfs.astype(np.float32), affine)
    nib.save(nii_out, "unwrapped_sharp.nii")
