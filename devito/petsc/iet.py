from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (FindNodes, Call,
                           Transformer, FindSymbols,
                           MapNodes, Iteration, Callable, Callback, List, Uxreplace,
                           Definition, BlankLine, PointerCast)
from devito.petsc.types import (PetscMPIInt, PETScStruct, DM, Mat,
                          Vec, KSP, PC, SNES, PetscErrorCode, PETScArray)
from devito.symbolics import Byref, Macro, FieldFromPointer
import cgen as c
from devito.petsc.nodes import MatVecAction, RHSLinearSystem, LinearSolverExpression


@iet_pass
def lower_petsc(iet, **kwargs):

    # Check if PETScSolve was used.
    petsc_nodes = FindNodes(MatVecAction).visit(iet)

    if not petsc_nodes:
        return iet, {}

    else:
        # Collect all petsc solution fields
        unique_targets = list(set([i.expr.rhs.target for i in petsc_nodes]))

        # Initalize PETSc
        init = init_petsc(**kwargs)

        # Create context data struct
        struct = build_struct(iet)

        objs = build_core_objects(unique_targets[-1], **kwargs)

        # Create core PETSc calls (not specific to each PETScSolve)
        core = core_petsc(unique_targets[-1], struct, objs, **kwargs)

        matvec_mapper = MapNodes(Iteration, MatVecAction, 'groupby').visit(iet)

        main_mapper = {}

        setup = []
        efuncs = []

        for target in unique_targets:

            solver_objs = build_solver_objs(target)

            matvec_body_list = List()

            solver_setup = False

            for iter, (matvec,) in matvec_mapper.items():

                # Skip the MatVecAction if it is not associated with the target
                # There will be more than one MatVecAction associated with the target
                # e.g interior matvec + BC matvecs
                if matvec.expr.rhs.target != target:
                    continue

                # Only need to generate solver setup once per target
                if not solver_setup:
                    solver = generate_solver_calls(solver_objs, objs, matvec, target)
                    setup.extend(solver)
                    solver_setup = True

                matvec_body = matvec_body_list._rebuild(body=[
                    matvec_body_list.body, iter[0]])
                matvec_body_list = matvec_body_list._rebuild(body=matvec_body)

                main_mapper.update({iter[0]: None})

            # Create the matvec callback and operation for each target
            matvec_callback, matvec_op = create_matvec_callback(
                target, matvec_body_list, solver_objs, objs,
                struct)

            setup.append(matvec_op)
            setup.append(BlankLine)
            efuncs.append(matvec_callback)

        # Remove the LinSolveExpr from iet and efuncs that were used to carry
        # metadata e.g solver_parameters
        main_mapper.update(rebuild_expr_mapper(iet))
        efunc_mapper = {efunc: rebuild_expr_mapper(efunc) for efunc in efuncs}

        iet = Transformer(main_mapper).visit(iet)
        efuncs = [Transformer(efunc_mapper[efunc]).visit(efunc) for efunc in efuncs]

        # Replace symbols appearing in each efunc with a pointer to the PETScStruct
        efuncs = transform_efuncs(efuncs, struct)

        body = iet.body._rebuild(init=init, body=core + tuple(setup) + iet.body.body)
        iet = iet._rebuild(body=body)

        return iet, {'efuncs': efuncs}


def init_petsc(**kwargs):

    # Initialize PETSc -> for now, assuming all solver options have to be
    # specifed via the parameters dict in PETScSolve
    # TODO: Are users going to be able to use PETSc command line arguments?
    # In firedrake, they have an options_prefix for each solver, enabling the use
    # of command line options
    initialize = Call(petsc_call, [
        Call('PetscInitialize', arguments=[Null, Null, Null, Null])])

    return tuple([petsc_func_begin_user, initialize])


def build_struct(iet):
    # Place all context data required by the shell routines
    # into a PETScStruct
    usr_ctx = []

    basics = FindSymbols('basics').visit(iet)
    avoid = FindSymbols('dimensions|indexedbases').visit(iet)
    usr_ctx.extend(data for data in basics if data not in avoid)

    return PETScStruct('ctx', usr_ctx)


def core_petsc(target, struct, objs, **kwargs):

    # MPI
    call_mpi = Call(petsc_call_mpi, [Call('MPI_Comm_size',
                                          arguments=[objs['comm'],
                                                     Byref(objs['size'])])])
    # Create DMDA
    dmda = create_dmda(target, objs)
    dm_setup = Call(petsc_call, [
        Call('DMSetUp', arguments=[objs['da']])])
    dm_app_ctx = Call(petsc_call, [
        Call('DMSetApplicationContext', arguments=[objs['da'], struct])])
    dm_mat_type = Call(petsc_call, [
        Call('DMSetMatType', arguments=[objs['da'], 'MATSHELL'])])

    return tuple([petsc_func_begin_user, call_mpi, dmda, dm_setup,
                  dm_app_ctx, dm_mat_type, BlankLine])


def build_core_objects(target, **kwargs):

    if kwargs['options']['mpi']:
        communicator = target.grid.distributor._obj_comm
    else:
        communicator = 'PETSC_COMM_SELF'

    return {'da': DM(name='da', liveness='eager'),
            'size': PetscMPIInt(name='size'),
            'comm': communicator,
            'err': PetscErrorCode(name='err')}


def create_dmda(target, objs):

    args = [objs['comm']]

    args += ['DM_BOUNDARY_GHOSTED' for _ in range(len(target.space_dimensions))]

    # Stencil type
    if len(target.space_dimensions) > 1:
        args += ['DMDA_STENCIL_BOX']

    # Global dimensions
    args += list(target.shape_global)[::-1]

    # No.of processors in each dimension
    if len(target.space_dimensions) > 1:
        args += list(target.grid.distributor.topology)[::-1]

    args += [1, target.space_order]

    args += [Null for _ in range(len(target.space_dimensions))]

    args += [Byref(objs['da'])]

    dmda = Call(petsc_call, [
        Call(f'DMDACreate{len(target.space_dimensions)}d', arguments=args)])

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
            'X_global': Vec(name='X_global_'+str(target.name)),
            'Y_global': Vec(name='Y_global_'+str(target.name)),
            'X_local': Vec(name='X_local_'+str(target.name), liveness='eager'),
            'Y_local': Vec(name='Y_local_'+str(target.name), liveness='eager')
            }


def generate_solver_calls(solver_objs, objs, matvec, target):

    solver_params = matvec.expr.rhs.solver_parameters

    snes_create = Call(petsc_call, [Call('SNESCreate', arguments=[
        objs['comm'], Byref(solver_objs['snes'])])])

    snes_set_dm = Call(petsc_call, [Call('SNESSetDM', arguments=[
        solver_objs['snes'], objs['da']])])

    create_matrix = Call(petsc_call, [Call('DMCreateMatrix', arguments=[
        objs['da'], Byref(solver_objs['Jac'])])])

    # NOTE: Assumming all solves are linear for now.
    snes_set_type = Call(petsc_call, [Call('SNESSetType', arguments=[
        solver_objs['snes'], 'SNESKSPONLY'])])

    global_x = Call(petsc_call, [Call('DMCreateGlobalVector', arguments=[
        objs['da'], Byref(solver_objs['x_global'])])])

    local_x = Call(petsc_call, [Call('DMCreateLocalVector', arguments=[
        objs['da'], Byref(solver_objs['x_local'])])])

    global_b = Call(petsc_call, [Call('DMCreateGlobalVector', arguments=[
        objs['da'], Byref(solver_objs['b_global'])])])

    local_b = Call(petsc_call, [Call('DMCreateLocalVector', arguments=[
        objs['da'], Byref(solver_objs['b_local'])])])

    snes_get_ksp = Call(petsc_call, [Call('SNESGetKSP', arguments=[
        solver_objs['snes'], Byref(solver_objs['ksp'])])])

    vec_replace_array = Call(petsc_call, [
        Call('VecReplaceArray', arguments=[solver_objs[
            'x_local'], FieldFromPointer(target._C_field_data, target._C_symbol)])])

    ksp_set_tols = Call(petsc_call, [Call('KSPSetTolerances', arguments=[
        solver_objs['ksp'], solver_params['ksp_rtol'], solver_params['ksp_atol'],
        solver_params['ksp_divtol'], solver_params['ksp_max_it']])])

    ksp_set_type = Call(petsc_call, [Call('KSPSetType', arguments=[
        solver_objs['ksp'], linear_solver_mapper[solver_params['ksp_type']]])])

    ksp_get_pc = Call(petsc_call, [Call('KSPGetPC', arguments=[
        solver_objs['ksp'], Byref(solver_objs['pc'])])])

    pc_set_type = Call(petsc_call, [Call('PCSetType', arguments=[
        solver_objs['pc'], linear_solver_mapper[solver_params['pc_type']]])])

    ksp_set_from_ops = Call(petsc_call, [Call('KSPSetFromOptions', arguments=[
        solver_objs['ksp']])])

    return tuple([snes_create, snes_set_dm, create_matrix, snes_set_type,
                  global_x, local_x, global_b, local_b, snes_get_ksp,
                  vec_replace_array, ksp_set_tols, ksp_set_type, ksp_get_pc,
                  pc_set_type, ksp_set_from_ops])


def create_matvec_callback(target, body, solver_objs, objs, struct):

    # There will be 2 PETScArrays within the body
    petsc_arrays = [i.function for i in FindSymbols('indexedbases').visit(body)
                    if isinstance(i.function, PETScArray)]

    # There will only be one PETScArray that is written to within this body and
    # one PETScArray which corresponds to the 'seed' vector
    petsc_arr_write, = FindSymbols('writes').visit(body)
    petsc_arr_seed, = [i for i in petsc_arrays if i.function != petsc_arr_write.function]
    
    # Struct needs to be defined explicitly here since CompositeObjects
    # do not have 'liveness'
    defn_struct = Definition(struct)

    mat_get_dm = Call(petsc_call, [Call('MatGetDM', arguments=[
        solver_objs['Jac'], Byref(objs['da'])])])

    dm_get_app_context = Call(petsc_call, [Call('DMGetApplicationContext', arguments=[
        objs['da'], Byref(struct._C_symbol)])])

    dm_get_local_xvec = Call(petsc_call, [Call('DMGetLocalVector', arguments=[
        objs['da'], Byref(solver_objs['X_local'])])])

    global_to_local_begin = Call(petsc_call, [Call('DMGlobalToLocalBegin', arguments=[
        objs['da'], solver_objs['X_global'], 'INSERT_VALUES', solver_objs['X_local']])])

    global_to_local_end = Call(petsc_call, [Call('DMGlobalToLocalEnd', arguments=[
        objs['da'], solver_objs['X_global'], 'INSERT_VALUES', solver_objs['X_local']])])

    dm_get_local_yvec = Call(petsc_call, [Call('DMGetLocalVector', arguments=[
        objs['da'], Byref(solver_objs['Y_local'])])])

    vec_get_array_y = Call(petsc_call, [Call('VecGetArray', arguments=[
        solver_objs['Y_local'], Byref(petsc_arr_write.function)])])

    vec_get_array_x = Call(petsc_call, [Call('VecGetArray', arguments=[
        solver_objs['X_local'], Byref(petsc_arrays[0])])])

    dm_get_local_info = Call(petsc_call, [Call('DMDAGetLocalInfo', arguments=[
        objs['da'], Byref(petsc_arrays[0].function.dmda_info)])])

    casts = [PointerCast(i.function) for i in petsc_arrays]

    vec_restore_array_y = Call(petsc_call, [Call('VecRestoreArray', arguments=[
        solver_objs['Y_local'], Byref(petsc_arr_write[0])])])

    vec_restore_array_x = Call(petsc_call, [Call('VecRestoreArray', arguments=[
        solver_objs['X_local'], Byref(petsc_arr_seed)])])

    dm_local_to_global_begin = Call(petsc_call, [Call('DMLocalToGlobalBegin', arguments=[
        objs['da'], solver_objs['Y_local'], 'INSERT_VALUES', solver_objs['Y_global']])])

    dm_local_to_global_end = Call(petsc_call, [Call('DMLocalToGlobalEnd', arguments=[
        objs['da'], solver_objs['Y_local'], 'INSERT_VALUES', solver_objs['Y_global']])])

    func_return = Call('PetscFunctionReturn', arguments=[0])

    matvec_body = List(body=[
        petsc_func_begin_user, defn_struct, mat_get_dm, dm_get_app_context,
        dm_get_local_xvec, global_to_local_begin, global_to_local_end,
        dm_get_local_yvec, vec_get_array_y, vec_get_array_x, dm_get_local_info,
        casts, BlankLine, body, vec_restore_array_y, vec_restore_array_x,
        dm_local_to_global_begin, dm_local_to_global_end, func_return])

    matvec_callback = Callable(
        'MyMatShellMult_'+str(target.name), matvec_body, retval=objs['err'],
        parameters=(solver_objs['Jac'], solver_objs['X_global'], solver_objs['Y_global']))

    matvec_operation = Call(petsc_call, [
        Call('MatShellSetOperation', arguments=[
            solver_objs['Jac'], 'MATOP_MULT', Callback(matvec_callback.name,
                                                       Void, Void)])])

    return matvec_callback, matvec_operation


def rebuild_expr_mapper(callable):

    return {expr: expr._rebuild(
        expr=expr.expr._rebuild(rhs=expr.expr.rhs.expr)) for
        expr in FindNodes(LinearSolverExpression).visit(callable)}


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
Void = 'void'

petsc_call = 'PetscCall'
petsc_call_mpi = 'PetscCallMPI'
# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')

linear_solver_mapper = {
    'gmres': 'KSPGMRES',
    'jacobi': 'PCJACOBI',
    None: 'PCNONE'
}
