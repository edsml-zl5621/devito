from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (FindNodes, Call, MatVecAction,
                           Transformer, FindSymbols, LinearSolverExpression,
                           MapNodes, Iteration, Callable, Callback, List, Uxreplace,
                           Definition)
from devito.types import (PetscMPIInt, PETScStruct, DMDALocalInfo, DM, Mat,
                          Vec, KSP, PC, SNES, PetscErrorCode)
from devito.symbolics import Byref, Macro, FieldFromPointer
import cgen as c

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

        matvec_mapper = MapNodes(Iteration, MatVecAction, 'groupby').visit(iet)

        main_mapper = {}

        setup = []
        efuncs = []
        for target in unique_targets:

            solver_objs = build_solver_objs(target)

            matvec_callback_body_iters = []

            solver_setup = False

            for iter, (matvec,) in matvec_mapper.items():

                if matvec.expr.rhs.target == target:
                    if not solver_setup:
                        solver = generate_solver_calls(solver_objs, objs, matvec)
                        setup.extend(solver)
                        solver_setup = True

                    matvec_body = matvec_body_list._rebuild(body=[
                        matvec_body_list.body, iter[0]])
                    matvec_body_list = matvec_body_list._rebuild(body=matvec_body)

                    main_mapper.update({iter[0]: None})

            matvec_callback, matvec_op = create_matvec_callback(
                target, matvec_callback_body_iters, solver_objs, objs, struct)

            setup.append(matvec_op)
            setup.append(c.Line())

            efuncs.append(matvec_callback)

        # Remove the LinSolveExpr from iet and efuncs that were used to carry
        # metadata e.g solver_parameters
        main_mapper.update(rebuild_expr_mapper(iet))
        efunc_mapper = {efunc: rebuild_expr_mapper(efunc) for efunc in efuncs}

        iet = Transformer(main_mapper).visit(iet)
        efuncs = [Transformer(efunc_mapper[efunc]).visit(efunc) for efunc in efuncs]

        # Replace symbols appearing in each efunc with a pointer to the struct
        efuncs = transform_efuncs(efuncs, struct)

        body = iet.body._rebuild(body=(tuple(init_setup) + iet.body.body))
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
    call_mpi = Call(petsc_call_mpi, [Call('MPI_Comm_size',
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
    args += ['DMDA_STENCIL_BOX']

    # global dimensions
    args += list(target.shape_global)[::-1]

    # no.of processors in each dimension
    args += list(target.grid.distributor.topology)[::-1]

    args += [1, target.space_order]

    args += ['NULL' for _ in range(len(target.space_dimensions))]

    args += [Byref(objs['da'])]

    dmda = Call(f'DMDACreate{len(target.space_dimensions)}d', arguments=args)

    return dmda


def build_solver_objs(target):

    return {'Jac': Mat(name='J_'+str(target.name)),
            'x_global': Vec(name='x_global_'+str(target.name)),
            'x_local': Vec(name='x_local_'+str(target.name), liveness='eager'),
            'b_global': Vec(name='b_global_'+str(target.name)),
            'b_local': Vec(name='b_local_'+str(target.name), liveness='eager'),
            'ksp': KSP(name='ksp_'+str(target.name)),
            'pc': PC(name='pc_'+str(target.name)),
            'snes': SNES(name='snes_'+str(target.name)),
            'x': Vec(name='x_'+str(target.name)),
            'y': Vec(name='y_'+str(target.name))}


def generate_solver_calls(solver_objs, objs, matvec):

    snes_create = Call('PetscCall', [Call('SNESCreate', arguments=[
        objs['comm'], Byref(solver_objs['snes'])])])

    snes_set_dm = Call('PetscCall', [Call('SNESSetDM', arguments=[
        solver_objs['snes'], objs['da']])])

    create_matrix = Call('PetscCall', [Call('DMCreateMatrix', arguments=[
        objs['da'], Byref(solver_objs['Jac'])])])

    # NOTE: Assumming all solves are linear for now.
    snes_set_type = Call('PetscCall', [Call('SNESSetType', arguments=[
        solver_objs['snes'], 'SNESKSPONLY'])])

    global_x = Call('PetscCall', [Call('DMCreateGlobalVector', arguments=[
        objs['da'], Byref(solver_objs['x_global'])])])

    local_x = Call('PetscCall', [Call('DMCreateLocalVector', arguments=[
        objs['da'], Byref(solver_objs['x_local'])])])

    global_b = Call('PetscCall', [Call('DMCreateGlobalVector', arguments=[
        objs['da'], Byref(solver_objs['b_global'])])])

    local_b = Call('PetscCall', [Call('DMCreateLocalVector', arguments=[
        objs['da'], Byref(solver_objs['b_local'])])])

    snes_get_ksp = Call('PetscCall', [Call('SNESGetKSP', arguments=[
        solver_objs['snes'], Byref(solver_objs['ksp'])])])

    return tuple([snes_create, snes_set_dm, create_matrix, snes_set_type,
                  global_x, local_x, global_b, local_b, snes_get_ksp])


def create_matvec_callback(target, matvec_callback_body_iters,
                           solver_objs, objs, struct):

    # Struct needs to be defined explicitly here since CompositeObjects
    # do not have 'liveness'
    defn_struct = Definition(struct)

    get_context = Call('PetscCall', [Call('MatShellGetContext',
                                          arguments=[solver_objs['Jac'],
                                                     Byref(struct)])])

    body = List(body=[defn_struct,
                      get_context,
                      matvec_callback_body_iters])

    matvec_callback = Callable('MyMatShellMult_'+str(target.name),
                               matvec_body,
                               retval=objs['err'],
                               parameters=(solver_objs['Jac'],
                                           solver_objs['x'],
                                           solver_objs['y']))

    matvec_operation = Call('PetscCall', [
        Call('MatShellSetOperation', arguments=[solver_objs['Jac'],
                                                'MATOP_MULT',
                                                Callback(matvec_callback.name,
                                                         Void, Void)])])

    return matvec_callback, matvec_operation


def rebuild_expr_mapper(iet):

    return {expr: expr._rebuild(
        expr=expr.expr._rebuild(rhs=expr.expr.rhs.expr)) for
        expr in FindNodes(LinearSolverExpression).visit(iet)}


def transform_efuncs(efuncs, struct):

    efuncs_new = []
    for efunc in efuncs:
        new_body = efunc.body
        for i in struct.usr_ctx:
            new_body = Uxreplace({i: FieldFromPointer(i, struct)}).visit(new_body)
        efunc_with_new_body = efunc._rebuild(body=new_body)
        efuncs_new.append(efunc_with_new_body)

    return efuncs_new


Null = Macro('NULL')
Void = Macro('void')

petsc_call = String('PetscCall')
petsc_call_mpi = String('PetscCallMPI')
petsc_function_begin_user = c.Line('PetscFunctionBeginUser;')

linear_solver_mapper = {
    'gmres': 'KSPGMRES',
    'jacobi': 'PCJACOBI',
    None: 'PCNONE'
}
