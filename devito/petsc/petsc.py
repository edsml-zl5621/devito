from devito.tools import timed_pass
from devito.petsc import LinearSolveExpr

__all__ = ['petsc_lift']


@timed_pass()
def petsc_lift(clusters):
    """
    Lift the iteration space surrounding each PETSc equation to create
    distinct iteration loops. This simplifys the movement of the loops
    into specific callback functions generated at the IET level.
    TODO: Potentially only need to lift the PETSc equations required
    by the callback functions, not the ones that stay inside the main kernel.
    """
    processed = []
    for c in clusters:

        if isinstance(c.exprs[0].rhs, LinearSolveExpr):
            ispace = c.ispace.lift(c.exprs[0].rhs.target.dimensions)
            processed.append(c.rebuild(ispace=ispace))
        else:
            processed.append(c)

    return processed
