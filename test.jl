import Pkg
Pkg.activate("env_julia")

#using QSM
using NIfTI
#using MriResearchTools
using ROMEO

# Load your background-corrected field map (units: ppm)
fl = NIfTI.niread("./phase.nii").raw
mask = NIfTI.niread("./mask.nii").raw  .> 0.5# Convert to Bool if needed

mag = NIfTI.niread("./magnitude.nii").raw



# Metadata
vsz  = (1.0, 1.0, 1.0)          # voxel size in mm
bdir = (0.0, 0.0, 1.0)          # direction of B0 field (usually z-axis)

#x = rts(fl, mask, vsz, bdir=bdir)
x = unwrapped = unwrap(fl,mag=mag)
volume = NIfTI.NIVolume(fl.header, x)
#volume = NIfTI.NIVolume(fl.header, x[1]) #->for vsharp
# Save result
NIfTI.niwrite("./romeo.nii", volume)