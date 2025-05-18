import numpy as np
from scipy.fftpack import dctn, idctn
import nibabel as nib
import time

#https://github.com/korbinian90/MriResearchTools.jl/blob/master/src/laplacianunwrapping.jl

"""Laplacian phase unwrapping (Schofield & Zhu 2003)"""

def pqterm(shape):
    """
    Compute the Laplacian eigenvalue term matrix (p^2 + q^2 + r^2 + ...) 
    It is an array of the same shape as the fieldmap, where each voxel holds the sum of squared indices
    starting from 1
    """

    # Step 1: Create 1D coordinate range arrays for each axis.
    # For example, if shape = (3, 4, 2), we'll get:
    #   [array([1., 2., 3.]), array([1., 2., 3., 4.]), array([1., 2.])]
    # As the input shape is 3D, there will be a total of three arrays after this step
    coord_arrays = []
    for dim_size in shape:
        axis_coords = np.arange(1, dim_size + 1, dtype=np.float64)  # Starting from 1
        coord_arrays.append(axis_coords)

    # Step 2: Create 3D grids of coordinate indices using meshgrid
    # For shape = (3, 4, 2), this creates three 3×4×2 arrays:
    #   - grids[0]: Contains the x-axis indices (1 to 3), repeated along y and z
    #   - grids[1]: Contains the y-axis indices (1 to 4), repeated along x and z
    #   - grids[2]: Contains the z-axis indices (1 to 2), repeated along x and y
    grids = np.meshgrid(*coord_arrays, indexing='ij')  

    # Step 3: Sum the squares of each coordinate
    # This results in the Laplacian eigenvalue term at each voxel
    pq = np.zeros(shape, dtype=np.float64)
    for g in grids:
        pq += g**2

    # result: pq[i, j, k] = grids[0][i,j,k]**2 + grids[1][i,j,k]**2 + grids[2][i,j,k]**2

    return pq


def dct_laplacian(x):
    """second derivative using DCT"""
    shape = x.shape
    pq = pqterm(shape)
    return - (2 * np.pi)**x.ndim / np.prod(shape) * idctn(pq * dctn(x, type=2, norm='ortho'), type=2, norm='ortho')

def dct_laplacian_inverse(x):
    """inverse second derivative using DCT"""
    shape = x.shape
    pq = pqterm(shape)
    pq[pq == 0] = np.inf  # avoid divide by zero
    return - np.prod(shape) / (2 * np.pi)**x.ndim * idctn(dctn(x, type=2, norm='ortho') / pq, type=2, norm='ortho')

def run_unwrap(file_path):

    start_time = time.time()
    print("Unwrapping started...")


    #Load the files (given the file path)
    field_nii = nib.load(file_path)
    phi_wrapped = field_nii.get_fdata()

    affine = field_nii.affine




    #Convert np.memmap (from the field data) to np.ndarray
    phi_wrapped = np.asarray(phi_wrapped, dtype=np.float64)

    
    lap = dct_laplacian(phi_wrapped)
    lap_nw = np.cos(phi_wrapped) * dct_laplacian(np.sin(phi_wrapped)) - np.sin(phi_wrapped) * dct_laplacian(np.cos(phi_wrapped))


    #Poisson
    k = dct_laplacian_inverse(lap_nw - lap) / (2 * np.pi)


    # phase correction
    phi_unwrapped = phi_wrapped + 2 * np.pi * k

    nii_out = nib.Nifti1Image(phi_unwrapped.astype(np.float32), affine)
    nib.save(nii_out, "unwrapped.nii")

    elapsed = time.time() - start_time
    print(f"Unwrapping completed in {elapsed:.3f} seconds")

    return "unwrapped.nii"