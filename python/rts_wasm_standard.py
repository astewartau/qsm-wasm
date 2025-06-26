import numpy as np
import nibabel as nib
import time





def dipole_kernel(shape, vsz, bdir=(0, 0, 1)):
    #This function is taken from https://github.com/AlanKuurstra/pyqsm

    nx, ny, nz = shape
    dx, dy, dz = vsz
    FOVx = nx * dx
    FOVy = ny * dy
    FOVz = nz * dz

    kx=np.arange(-np.ceil((nx-1)/2.0),np.floor((nx-1)/2.0)+1)*1.0/FOVx
    ky=np.arange(-np.ceil((ny-1)/2.0),np.floor((ny-1)/2.0)+1)*1.0/FOVy
    kz=np.arange(-np.ceil((nz-1)/2.0),np.floor((nz-1)/2.0)+1)*1.0/FOVz
    
    KX,KY,KZ=np.meshgrid(kx,ky,kz)
    KX=KX.transpose(1,0,2)
    KY=KY.transpose(1,0,2)
    KZ=KZ.transpose(1,0,2)
    
    K2=KX**2+KY**2+KZ**2
    
    dipole_f=1.0/3-KZ**2/K2
    dipole_f=np.fft.ifftshift(dipole_f) 
    dipole_f[0,0,0]=0
    dipole_f=dipole_f.astype('complex')
    return dipole_f

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


    print("INPUT TO RTS - min: ", np.min(fieldmap),"max: ", np.max(fieldmap))

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

    chi_out = chi * mask
    nii_out = nib.Nifti1Image(chi_out.astype(np.float32), affine)
    nib.save(nii_out, output_path)


    elapsed = time.time() - start_time
    print(f"Dipole inversion completed in {elapsed:.3f} seconds")

    return output_path
