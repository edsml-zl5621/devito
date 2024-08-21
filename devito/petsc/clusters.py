from devito.tools import timed_pass
from devito.petsc.types import LinearSolveExpr, CallbackExpr


@timed_pass()
def petsc_lift(clusters):
    """
    Lift the iteration space surrounding each PETSc equation to create
    distinct iteration loops.
    # TODO: Can probably remove this now due to recursive compilation, but
    # leaving it for now.
    """
    processed = []
    # from IPython import embed; embed()
    for c in clusters:
        # from IPython import embed; embed()
        if isinstance(c.exprs[0].rhs, LinearSolveExpr):
            ispace = c.ispace.lift(c.exprs[0].rhs.target.space_dimensions)
            processed.append(c.rebuild(ispace=ispace))
        elif isinstance(c.exprs[0].rhs, CallbackExpr):
            # from IPython import embed; embed()
            # ispace = c.ispace.project([c.exprs[0].ispace.dimensions[1:]])
            # tmp = c.exprs[0].ispace.intervals.drop(c.exprs[0].ispace.dimensions[0])
            # ispace = c.ispace.rebuild(intervals=tmp)
            # from IPython import embed; embed()
            processed.append(c)
        else:
            processed.append(c)

    return processed
