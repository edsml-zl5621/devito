from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (FindNodes, MatVecAction, Call, LinearSolverExpression,
                           Expression, Transformer)
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

        includes = []
        # TODO: Obviously it won't be like this.
        kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/include')
        path = '/home/zl5621/petsc/arch-linux-c-debug/include'
        kwargs['compiler'].add_include_dirs(path)
        kwargs['compiler'].add_libraries('petsc')
        libdir = '/home/zl5621/petsc/arch-linux-c-debug/lib'
        kwargs['compiler'].add_library_dirs(libdir)
        kwargs['compiler'].add_ldflags('-Wl,-rpath,%s' % libdir)

        includes.extend(['petscksp.h', 'petscdmda.h'])

        return iet, {'includes': includes}

    return iet, {}


def petsc_setup(targets, **kwargs):


    size = PetscMPIInt(name='size')

    # Initialize PETSc
    initialize = Call('PetscCall', [Call('PetscInitialize',
                                         arguments=['NULL', 'NULL',
                                                    'NULL', 'NULL'])])
    
    call_mpi = Call('PetscCallMPI', [Call('MPI_Comm_size',
                                          arguments=['PETSC_COMM_WORLD',
                                                     Byref(size)])])

    return [initialize, call_mpi]
