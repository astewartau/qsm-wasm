import Pkg
Pkg.activate("env_julia")

using QSM
using NIfTI

# Load your background-corrected field map (units: ppm)
fl = NIfTI.niread("./test_algo/unwrapped.nii")
mask = NIfTI.niread("./mask.nii").raw  .> 0.5# Convert to Bool if needed



# Metadata
vsz  = (1.0, 1.0, 1.0)          # voxel size in mm
bdir = (0.0, 0.0, 1.0)          # direction of B0 field (usually z-axis)

#x = rts(fl, mask, vsz, bdir=bdir)
x = sharp(fl, mask, vsz, r=6.0)

volume = NIfTI.NIVolume(fl.header, x[1])
# Save result
NIfTI.niwrite("julia_sharp6.nii", volume)