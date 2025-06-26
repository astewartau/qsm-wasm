import Pkg
Pkg.activate("qsm_env")
using QSM
function run_rts(fieldmap, mask)
    vsz = (1.0, 1.0, 1.0)
    bdir = (0.0, 0.0, 1.0)
    return QSM.rts(fieldmap, mask, vsz, bdir)
end