from devito.passes.iet.engine import iet_pass

__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):

    # TODO: This is a placeholder for the actual PETSc lowering.

    return iet, {}
