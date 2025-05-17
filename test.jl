import Pkg
Pkg.activate("env_julia")

using QSM
using NIfTI

# Load your background-corrected field map (units: ppm)
fl = NIfTI.niread("./test_algo/sharp.nii")
mask = NIfTI.niread("./test_algo/mask.nii").raw  .> 0.5# Convert to Bool if needed



# Metadata
vsz  = (1.0, 1.0, 1.0)          # voxel size in mm
bdir = (0.0, 0.0, 1.0)          # direction of B0 field (usually z-axis)

x = rts(fl, mask, vsz, bdir=bdir)

volume = NIfTI.NIVolume(fl.header, x)
# Save result
NIfTI.niwrite("qsm_output.nii", volume)