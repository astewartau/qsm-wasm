import Pkg
Pkg.activate("env_julia")

using QSM
using NIfTI
#using MriResearchTools
#using ROMEO

# Load your background-corrected field map (units: ppm)
fl = NIfTI.niread("./reference/fieldmap-local.nii")
mask = NIfTI.niread("./reference/mask.nii").raw  .> 0.5# Convert to Bool if needed


# Metadata
vsz  = (1.0, 1.0, 1.0)          # voxel size in mm
bdir = (0.0, 0.0, 1.0)          # direction of B0 field (usually z-axis)

x = rts(fl, mask, vsz, bdir=bdir)

volume = NIfTI.NIVolume(fl.header, x)
#volume = NIfTI.NIVolume(fl.header, x[1]) #->for vsharp
# Save result
NIfTI.niwrite("./Benchmark/bg_removal/rts_julia.nii", volume)