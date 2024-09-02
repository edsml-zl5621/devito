from devito.tools import timed_pass
from devito.petsc.types import LinearSolveExpr, CallbackExpr


@timed_pass()
def petsc_lift(clusters):
    """
    Lift the iteration space surrounding each PETSc equation to create
    distinct iteration loops.
    """
    processed = []
    for c in clusters:
        if isinstance(c.exprs[0].rhs, LinearSolveExpr):
            ispace = c.ispace.lift(c.exprs[0].rhs.target.space_dimensions)
            processed.append(c.rebuild(ispace=ispace))
        else:
            processed.append(c)

    return processed


@timed_pass()
def petsc_project(clusters):
    """
    Drop time loop for clusters which appear in PETSc callback functions.
    """
    processed = []
    for c in clusters:
        if isinstance(c.exprs[0].rhs, CallbackExpr):
            time_dims = [d for d in c.ispace.intervals.dimensions if d.is_Time]
            ispace = c.ispace.project(lambda d: d not in time_dims)
            processed.append(c.rebuild(ispace=ispace))
        else:
            processed.append(c)

    return processed
