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

        # Initalize PETSc
        init = init_petsc(**kwargs)

        # MPI
        call_mpi = mpi_petsc(targets, **kwargs)

        # TODO: Insert code that utilises the metadata attached to each LinSolveExpr
        # that appears in the RHS of each LinearSolverExpression.

        # Remove the LinSolveExpr that was utilised above to carry metadata.
        mapper = {expr:
                  expr._rebuild(expr=expr.expr._rebuild(rhs=expr.expr.rhs.expr))
                  for expr in is_petsc}

        iet = Transformer(mapper).visit(iet)

        body = iet.body._rebuild(init=init, body=call_mpi + iet.body.body)
        iet = iet._rebuild(body=body)

    return iet, {}


def init_petsc(**kwargs):

    # Initialize PETSc -> for now, assuming all solver options have to be 
    # specifed via the parameters dict in PETScSolve.
    # NOTE: Are users going to be able to use PETSc command line arguments?
    # In firedrake, they have an options_prefix for each solver, enabling the use
    # of command line options.
    initialize = Call('PetscCall', [Call('PetscInitialize',
                                         arguments=['NULL', 'NULL',
                                                    'NULL', 'NULL'])])

    return tuple([initialize])


def mpi_petsc(targets, **kwargs):

    # Assumption: all targets are generated from the same Grid.
    if kwargs['options']['mpi']:
        communicator = targets[-1].grid.distributor._obj_comm
    else:
        communicator = 'PETSC_COMM_SELF'

    size = PetscMPIInt(name='size')

    call_mpi = Call('PetscCallMPI', [Call('MPI_Comm_size',
                                          arguments=[communicator,
                                                     Byref(size)])])

    return tuple([call_mpi])
