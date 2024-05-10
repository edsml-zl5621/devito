from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (FindNodes, Call, MatVecAction,
                           Transformer, FindSymbols, LinearSolverExpression)
from devito.types import PetscMPIInt, PETScStruct, DMDALocalInfo, DM
from devito.symbolics import Byref

__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):

    # TODO: Drop the LinearSolveExpr's using .args[0] so that _rebuild doesn't
    # appear in ccode

    # Check if PETScSolve was used and count occurrences. Each PETScSolve
    # will have a unique MatVecAction.
    is_petsc = FindNodes(MatVecAction).visit(iet)

    if is_petsc:

        # Collect all solution fields we're solving for
        targets = [i.expr.rhs.target for i in is_petsc]

        # Initalize PETSc
        init = init_petsc(**kwargs)

        # Create context data struct.
        struct = build_struct(iet)

        objs = build_core_objects(targets[-1], struct, **kwargs)

        # Create core PETSc calls required by general linear solves,
        # which only need to be generated once e.g create DMDA.
        core = core_petsc(targets[-1], objs, **kwargs)

        # TODO: Insert code that utilises the metadata attached to each LinSolveExpr
        # that appears in the RHS of each LinearSolverExpression.

        # Remove the LinSolveExpr that was utilised above to carry metadata.
        mapper = {expr:
                  expr._rebuild(expr=expr.expr._rebuild(rhs=expr.expr.rhs.expr))
                  for expr in FindNodes(LinearSolverExpression).visit(iet)}

        iet = Transformer(mapper).visit(iet)

        body = iet.body._rebuild(init=init, body=core + iet.body.body)
        iet = iet._rebuild(body=body)

    return iet, {}


def init_petsc(**kwargs):

    # Initialize PETSc -> for now, assuming all solver options have to be
    # specifed via the parameters dict in PETScSolve.
    # NOTE: Are users going to be able to use PETSc command line arguments?
    # In firedrake, they have an options_prefix for each solver, enabling the use
    # of command line options.
    initialize = Call('PetscCall', [Call('PetscInitialize',
                                         arguments=['NULL', 'NULL',
                                                    'NULL', 'NULL'])])

    return tuple([initialize])


def build_struct(iet):
    # Place all context data required by the shell routines
    # into a PETScStruct.
    usr_ctx = []

    basics = FindSymbols('basics').visit(iet)
    avoid = FindSymbols('dimensions|indexedbases').visit(iet)
    usr_ctx.extend(data for data in basics if data not in avoid)

    return PETScStruct('ctx', usr_ctx)


def core_petsc(target, objs, **kwargs):
    # Assumption: all targets are generated from the same Grid,
    # so we can use any target.

    # MPI
    call_mpi = Call('PetscCallMPI', [Call('MPI_Comm_size',
                                          arguments=[objs['comm'],
                                                     Byref(objs['size'])])])

    # Create DMDA
    dmda = create_dmda(target, objs)
    dm_setup = Call('PetscCall', [Call('DMSetUp', arguments=[objs['da']])])
    dm_app_ctx = Call('PetscCall', [Call('DMSetApplicationContext',
                                         arguments=[objs['da'], objs['struct']])])
    dm_mat_type = Call('PetscCall', [Call('DMSetMatType',
                                          arguments=[objs['da'], 'MATSHELL'])])
    dm_local_info = Call('PetscCall', [Call('DMDAGetLocalInfo',
                                            arguments=[objs['da'], Byref(objs['info'])])])

    return tuple([call_mpi, dmda, dm_setup, dm_app_ctx, dm_mat_type, dm_local_info])


def build_core_objects(target, struct, **kwargs):

    if kwargs['options']['mpi']:
        communicator = target.grid.distributor._obj_comm
    else:
        communicator = 'PETSC_COMM_SELF'

    return {'da': DM(name='da'),
            'size': PetscMPIInt(name='size'),
            'info': DMDALocalInfo(name='info'),
            'comm': communicator,
            'struct': struct}


def create_dmda(target, objs):

    args = [objs['comm']]

    args += ['DM_BOUNDARY_GHOSTED' for _ in range(len(target.space_dimensions))]

    # stencil type
    if len(target.space_dimensions) > 1:
        args += ['DMDA_STENCIL_BOX']

    # global dimensions
    args += list(target.shape_global)[::-1]

    # no.of processors in each dimension
    if len(target.space_dimensions) > 1:
        args += list(target.grid.distributor.topology)[::-1]

    args += [1, target.space_order]

    args += ['NULL' for _ in range(len(target.space_dimensions))]

    args += [Byref(objs['da'])]

    dmda = Call('PetscCall', [Call(f'DMDACreate{len(target.space_dimensions)}d',
                                   arguments=args)])

    return dmda
