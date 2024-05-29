from devito.ir.equations import OpMatVec, OpRHS
from devito.petsc.iet.nodes import MatVecAction, RHSLinearSystem

# Mapping special Eq operations to their corresponding IET Expression subclass types.
# These operations correspond to subclasses of Eq utilised within PETScSolve.
iet_mapper = {OpMatVec: MatVecAction,
              OpRHS: RHSLinearSystem}