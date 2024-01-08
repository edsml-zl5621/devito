from devito.passes.iet.engine import iet_pass


__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):

    # Then I can access the is_action expression via:
    # tmp = FindNodes(Expression).visit(iet)
    # action = [i for i in tmp if i.expr.is_action]

    return iet, {'includes': ['petscksp.h', 'petscdmda.h']}
