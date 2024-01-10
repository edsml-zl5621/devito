from devito.passes.iet.engine import iet_pass


__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):

    # Then I can access the action expression via:
    # tmp = FindNodes(Expression).visit(iet)
    # action = [i for i in tmp if getattr(i, 'operation', None) \
    #           and getattr(i.operation, 'name', None) == 'action']

    return iet, {}
