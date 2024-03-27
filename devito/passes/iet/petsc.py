from devito.passes.iet.engine import iet_pass


__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):

    # TODO: This is a placeholder for the actual PETSc lowering.
    # action_expr = FindNodes(MatVecAction).visit(iet)
    # rhs_expr = FindNodes(RHSExpr).visit(iet)
    return iet, {}
