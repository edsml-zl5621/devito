from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (FindNodes, Call, LinearSolverExpression,
                           Transformer)
from devito.types import PetscMPIInt
from devito.symbolics import Byref

__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):

    # TODO: Drop the LinearSolveExpr's using .args[0] so that _rebuild doesn't
    # appear in ccode

    # Determine if there is was a PETScSolve
    is_petsc = FindNodes(LinearSolverExpression).visit(iet)

    if is_petsc:

        # Collect all solution fields we're solving for
        targets = [i.expr.rhs.target for i in is_petsc]

        # Initialize PETSc i.e generate the one off PETSc calls.
        init_setup = petsc_setup(targets, **kwargs)

        # TODO: Insert code that utilises the metadata attached to each LinSolveExpr
        # that appears in the RHS of each LinearSolverExpression.

        # Remove the LinSolveExpr that was utilised above to carry metadata.
        mapper = {expr:
                  expr._rebuild(expr=expr.expr._rebuild(rhs=expr.expr.rhs.expr))
                  for expr in is_petsc}

        iet = Transformer(mapper).visit(iet)

        body = iet.body._rebuild(body=(tuple(init_setup) + iet.body.body))
        iet = iet._rebuild(body=body)

    return iet, {}


def petsc_setup(targets, **kwargs):

    # Assumption: all targets are generated from the same Grid.
    if kwargs['options']['mpi']:
        communicator = targets[-1].grid.distributor._obj_comm
    else:
        communicator = 'PETSC_COMM_SELF'

    size = PetscMPIInt(name='size')

    # Initialize PETSc
    initialize = Call('PetscCall', [Call('PetscInitialize',
                                         arguments=['NULL', 'NULL',
                                                    'NULL', 'NULL'])])

    call_mpi = Call('PetscCallMPI', [Call('MPI_Comm_size',
                                          arguments=[communicator,
                                                     Byref(size)])])

    return [initialize, call_mpi]
