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
        if isinstance(c.exprs[0].rhs, LinearSolveExpr):
            ispace = c.ispace.lift(c.exprs[0].rhs.target.space_dimensions)
            processed.append(c.rebuild(ispace=ispace))
        elif isinstance(c.exprs[0].rhs, CallbackExpr):
            time_dims = [d for d in c.ispace.intervals.dimensions if d.is_Time]
            ispace = c.ispace.project(lambda d: d not in time_dims)
            processed.append(c.rebuild(ispace=ispace))
            # processed.append(c)
        else:
            processed.append(c)

    return processed
    

def attach_parent_modulo_dims(e, mds):
    def rebuild_items(items):
        return [i._rebuild(rhs=i.rhs._rebuild(parent_modulo_dims=mds)) for i in items]

    linsolveexpr = e.rhs
    if isinstance(linsolveexpr, LinearSolveExpr):
        linsolveexpr._matvecs = rebuild_items(linsolveexpr.matvecs)
        linsolveexpr._formfuncs = rebuild_items(linsolveexpr.formfuncs)
        linsolveexpr._formrhs = rebuild_items(linsolveexpr.formrhs)

    return e

