from devito.tools import timed_pass
from devito.petsc.types import LinearSolveExpr, CallbackExpr
import sympy


@timed_pass()
def petsc_lift(clusters):
    """
    - Lift the iteration space surrounding each PETSc equation to create
    distinct iteration loops.
    - Drop time-loop for expressions which appear in PETSc callback functions.
    """
    processed = []
    for c in clusters:
        if isinstance(c.exprs[0].rhs, LinearSolveExpr):
            ispace = c.ispace.lift(c.exprs[0].rhs.target.space_dimensions)
            processed.append(c.rebuild(ispace=ispace))

        # Drop time-loop for expressions that appear in PETSc callback functions
        elif isinstance(c.exprs[0].rhs, CallbackExpr):
            time_dims = [d for d in c.ispace.intervals.dimensions if d.is_Time]
            ispace = c.ispace.project(lambda d: d not in time_dims)
            processed.append(c.rebuild(ispace=ispace))

        else:
            processed.append(c)

    return processed

# class MySpecialFunction(sympy.Function):
#     @classmethod
#     def eval(cls, true_rhs, t_plus_1, t):
#         # Custom evaluation rules (optional)
#         return None 
    

def override_modulo_dims(e, mds):
    # from IPython import embed; embed()

    def rebuild_items(items, t_plus_1, t):
        new_items = []
        for i in items:
            original_callback = i.rhs
            rebuilt = original_callback._rebuild(expr=(MySpecialFunction(original_callback.expr, t_plus_1, t)))
            new_items.append(i._rebuild(rhs=rebuilt))
        return new_items
        # return [i._rebuild(rhs=i.rhs._rebuild(i.rhs.expr)) for i in items]

    linsolveexpr = e.rhs
    if isinstance(linsolveexpr, LinearSolveExpr):
        iters = list(list(mds[2].values())[0])
        # from IPython import embed; embed()
        t_plus_1 = iters[0]
        t = iters[1]
        linsolveexpr._matvecs = rebuild_items(linsolveexpr.matvecs, t_plus_1, t)
        # linsolveexpr._formfuncs = rebuild_items(linsolveexpr.formfuncs)
        # linsolveexpr._formrhs = rebuild_items(linsolveexpr.formrhs)

    return e
