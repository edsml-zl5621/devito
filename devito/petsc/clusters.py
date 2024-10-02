from devito.tools import timed_pass
from devito.petsc.types import LinearSolveExpr, CallbackExpr
from devito.symbolics import xreplace_indices
from devito.ir.support import IterationSpace


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

        elif isinstance(c.exprs[0].rhs, CallbackExpr):
            exprs = c.exprs
            timeee_mapper = exprs[0].rhs.target
            new_dict = {inner_key: inner_value for outer_dict in timeee_mapper.values() for inner_key, inner_value in outer_dict.items()}
            mod_dimss = [dim for dim in exprs[0].dimensions if dim.is_Modulo]
            temp_mapper = {dim: new_dict[dim.origin] for dim in mod_dimss}
            exprs_new = [xreplace_indices(exprs[0], temp_mapper)]
            ispace = IterationSpace(c.ispace.intervals, c.ispace.sub_iterators,
                                    c.ispace.directions)
            processed.append(c.rebuild(exprs=exprs_new, ispace=ispace))
            
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
