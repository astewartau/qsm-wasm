import Pkg
Pkg.activate("qsm_env")
using QSM
using NIfTI

# Load smaller test field map
fl = randn(64, 64, 64) * 0.01    # synthetic test data
mask = trues(64, 64, 64)
vsz = (1.0, 1.0, 1.0)
bdir = (0.0, 0.0, 1.0)

@time x = rts(fl, mask, vsz, bdir=bdir)