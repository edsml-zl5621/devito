from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (FindNodes, LinSolveMock, retrieve_iteration_tree,
                           filter_iterations, Transformer)

__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):

    # TODO: This is a placeholder for the actual PETSc lowering.

    return iet, {}


def drop_mocks(iet):
    """
    Drop the spatial iteration loop containing each LinSolveMock.
    """

    mapper = {}

    for tree in retrieve_iteration_tree(iet):

        # Eventually, when using implicit dims etc do not want to drop
        # the time loop.
        root = filter_iterations(tree, key=lambda i: i.dim.is_Space)
        mock = FindNodes(LinSolveMock).visit(root)

        if mock:
            mapper.update({root[0]: None})

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet
