from devito.passes.iet.engine import iet_pass
from devito.ir.iet import FindNodes, Expression, PointerCast, Call
from devito.types import Vec
from devito.symbolics import Byref

__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):

    # TODO: This is a placeholder for the actual PETSc lowering.

    # TODO: Drop the LinearSolveExpr's using .args[0] so that _rebuild doesn't
    # appear in ccode

    exprs = FindNodes(Expression).visit(iet)

    vec = Vec('local_xvec')
    call = Call('VecGetArray', [vec, Byref(exprs[0].expr.lhs.function._C_name)])

    # from IPython import embed; embed()
    tmp = PointerCast(exprs[0].expr.lhs.base.function)

    # from IPython import embed; embed()

    body = iet.body._rebuild(body=(tuple([call]) + tuple([tmp]) + iet.body.body))
    iet = iet._rebuild(body=body)

    return iet, {}
