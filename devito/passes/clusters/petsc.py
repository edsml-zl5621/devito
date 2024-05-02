from devito.tools import timed_pass
from devito.types import LinearSolveExpr

__all__ = ['petsc_lift']


@timed_pass()
def petsc_lift(clusters):
    """
    Lift the iteration space associated with each PETSc equation.
    TODO: Potentially only need to lift the PETSc equations required
    by the callback functions.
    """
    processed = []
    for c in clusters:

        if isinstance(c.exprs[0].rhs, LinearSolveExpr):
            ispace = c.ispace.lift(c.exprs[0].rhs.target.dimensions)
            processed.append(c.rebuild(ispace=ispace))
        else:
            processed.append(c)

    return processed
