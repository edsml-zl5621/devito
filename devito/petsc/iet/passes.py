from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (FindNodes, Call,
                           Transformer, FindSymbols,
                           MapNodes, Iteration, Callable, Callback, List, Uxreplace,
                           Definition, BlankLine, PointerCast)
from devito.petsc.types import (PetscMPIInt, PETScStruct, DM, Mat,
                                Vec, KSP, PC, SNES, PetscErrorCode, PETScArray)
from devito.symbolics import Byref, Macro, FieldFromPointer
from devito.petsc.iet.nodes import MatVecAction, LinearSolverExpression
from devito.petsc.utils import (solver_mapper, core_metadata,
                                petsc_call, petsc_call_mpi)
import cgen as c


@iet_pass
def lower_petsc(iet, **kwargs):

    # Check if PETScSolve was used
    petsc_nodes = FindNodes(MatVecAction).visit(iet)

    if not petsc_nodes:
        return iet, {}

    # Collect all petsc solution fields
    unique_targets = list(set([i.expr.rhs.target for i in petsc_nodes]))

    init = init_petsc(**kwargs)

    # Create context data struct
    struct = build_struct(iet)

    objs = build_core_objects(unique_targets[-1], **kwargs)

    # Create core PETSc calls (not specific to each PETScSolve)
    core = make_core_petsc_calls(unique_targets[-1], struct, objs, **kwargs)

    matvec_mapper = MapNodes(Iteration, MatVecAction, 'groupby').visit(iet)

    subs = {}

    setup = []
    efuncs = []

    for target in unique_targets:
        solver_objs = build_solver_objs(target)
        matvec_body_list = List()
        solver_setup = False

        for iter, (matvec,) in matvec_mapper.items():

            # Skip the MatVecAction if it is not associated with the target
            # There will most likely be more than one MatVecAction
            # associated with the target e.g interior matvec + BC matvecs
            if matvec.expr.rhs.target != target:
                continue

            # Only need to generate solver setup once per target
            if not solver_setup:
                solver = generate_solver_calls(solver_objs, objs, matvec, target)
                setup.extend(solver)
                solver_setup = True

            # Make the body of the matrix-vector callback for this target
            matvec_body = matvec_body_list._rebuild(body=[
                matvec_body_list.body, iter[0]])
            matvec_body_list = matvec_body_list._rebuild(body=matvec_body)

            # Remove the iteration loop from the main kernel encapsulating
            # the matvec equations since they are moved to the callback
            subs.update({iter[0]: None})

        # Create the matvec callback and operation for each target
        matvec_callback, matvec_op = create_matvec_callback(
            target, matvec_body_list, solver_objs, objs,
            struct)

        setup.extend([matvec_op, BlankLine])
        efuncs.append(matvec_callback)

    # Remove the LinSolveExpr from iet and efuncs that were used to carry
    # metadata e.g solver_parameters
    subs.update(rebuild_expr_mapper(iet))
    for efunc in efuncs:
        subs.update({efunc: rebuild_expr_mapper(efunc)})

    iet = Transformer(subs).visit(iet)
    efuncs = [Transformer(subs[efunc]).visit(efunc) for efunc in efuncs]

    # Replace symbols appearing in each efunc with a pointer to the PETScStruct
    efuncs = uxreplace_efuncs(efuncs, struct)

    body = iet.body._rebuild(init=init, body=core + tuple(setup) + iet.body.body)
    iet = iet._rebuild(body=body)

    # metadata = core_metadata()
    # metadata.update({'efuncs': efuncs})

    return iet, {'efuncs': efuncs}


def init_petsc(**kwargs):

    # Initialize PETSc -> for now, assuming all solver options have to be
    # specifed via the parameters dict in PETScSolve
    # TODO: Are users going to be able to use PETSc command line arguments?
    # In firedrake, they have an options_prefix for each solver, enabling the use
    # of command line options
    initialize = petsc_call('PetscInitialize', [Null, Null, Null, Null])

    return tuple([petsc_func_begin_user, initialize])


def build_struct(iet):
    # Place all context data required by the shell routines
    # into a PETScStruct
    usr_ctx = []

    basics = FindSymbols('basics').visit(iet)
    avoid = FindSymbols('dimensions|indexedbases').visit(iet)
    usr_ctx.extend(data for data in basics if data not in avoid)

    return PETScStruct('ctx', usr_ctx)


def make_core_petsc_calls(target, struct, objs, **kwargs):
    # MPI
    call_mpi = petsc_call_mpi('MPI_Comm_size', [objs['comm'], Byref(objs['size'])])

    # Create DMDA
    dmda = create_dmda(target, objs)

    dm_setup = petsc_call('DMSetUp', [objs['da']])
    dm_app_ctx = petsc_call('DMSetApplicationContext', [objs['da'], struct])
    dm_mat_type = petsc_call('DMSetMatType', [objs['da'], 'MATSHELL'])

    return (
        petsc_func_begin_user,
        call_mpi,
        dmda,
        dm_setup,
        dm_app_ctx,
        dm_mat_type,
        BlankLine
    )


def build_core_objects(target, **kwargs):

    if kwargs['options']['mpi']:
        communicator = target.grid.distributor._obj_comm
    else:
        communicator = 'PETSC_COMM_SELF'

    return {
        'da': DM(name='da', liveness='eager'),
        'size': PetscMPIInt(name='size'),
        'comm': communicator,
        'err': PetscErrorCode(name='err')
    }


def create_dmda(target, objs):
    # MPI communicator
    args = [objs['comm']]

    # Type of ghost nodes
    args.extend(['DM_BOUNDARY_GHOSTED' for _ in range(len(target.space_dimensions))])

    # Stencil type
    if len(target.space_dimensions) > 1:
        args.append('DMDA_STENCIL_BOX')

    # Global dimensions
    args.extend(list(target.shape_global)[::-1])
    # No.of processors in each dimension
    if len(target.space_dimensions) > 1:
        args.extend(list(target.grid.distributor.topology)[::-1])

    # Number of degrees of freedom per node
    args.append(1)
    # "Stencil width" -> size of overlap
    args.append(target.space_order)
    args.extend([Null for _ in range(len(target.space_dimensions))])

    # The resulting distributed array object
    args.append(Byref(objs['da']))

    # The PETSc call used to create the DMDA
    dmda = petsc_call('DMDACreate%sd' % len(target.space_dimensions), args)

    return dmda


def build_solver_objs(target):
    name = target.name

    return {
        'Jac': Mat(name='J_%s' % name),
        'x_global': Vec(name='x_global_%s' % name),
        'x_local': Vec(name='x_local_%s' % name, liveness='eager'),
        'b_global': Vec(name='b_global_%s' % name),
        'b_local': Vec(name='b_local_%s' % name, liveness='eager'),
        'ksp': KSP(name='ksp_%s' % name),
        'pc': PC(name='pc_%s' % name),
        'snes': SNES(name='snes_%s' % name),
        'X_global': Vec(name='X_global_%s' % name),
        'Y_global': Vec(name='Y_global_%s' % name),
        'X_local': Vec(name='X_local_%s' % name, liveness='eager'),
        'Y_local': Vec(name='Y_local_%s' % name, liveness='eager')
    }


def generate_solver_calls(solver_objs, objs, matvec, target):

    solver_params = matvec.expr.rhs.solver_parameters

    snes_create = petsc_call('SNESCreate', [objs['comm'], Byref(solver_objs['snes'])])

    snes_set_dm = petsc_call('SNESSetDM', [solver_objs['snes'], objs['da']])

    create_matrix = petsc_call('DMCreateMatrix', [objs['da'], Byref(solver_objs['Jac'])])

    # NOTE: Assumming all solves are linear for now.
    snes_set_type = petsc_call('SNESSetType', [solver_objs['snes'], 'SNESKSPONLY'])

    global_x = petsc_call('DMCreateGlobalVector',
                          [objs['da'], Byref(solver_objs['x_global'])])

    local_x = petsc_call('DMCreateLocalVector',
                         [objs['da'], Byref(solver_objs['x_local'])])

    global_b = petsc_call('DMCreateGlobalVector',
                          [objs['da'], Byref(solver_objs['b_global'])])

    local_b = petsc_call('DMCreateLocalVector',
                         [objs['da'], Byref(solver_objs['b_local'])])

    snes_get_ksp = petsc_call('SNESGetKSP',
                              [solver_objs['snes'], Byref(solver_objs['ksp'])])

    vec_replace_array = petsc_call(
        'VecReplaceArray', [solver_objs['x_local'],
                            FieldFromPointer(target._C_field_data, target._C_symbol)])

    ksp_set_tols = petsc_call(
        'KSPSetTolerances', [solver_objs['ksp'], solver_params['ksp_rtol'],
                             solver_params['ksp_atol'], solver_params['ksp_divtol'],
                             solver_params['ksp_max_it']])

    ksp_set_type = petsc_call(
        'KSPSetType', [solver_objs['ksp'], solver_mapper[solver_params['ksp_type']]])

    ksp_get_pc = petsc_call('KSPGetPC',
                            [solver_objs['ksp'], Byref(solver_objs['pc'])])

    pc_set_type = petsc_call('PCSetType',
                             [solver_objs['pc'], solver_mapper[solver_params['pc_type']]])

    ksp_set_from_ops = petsc_call('KSPSetFromOptions', [solver_objs['ksp']])

    return (
        snes_create,
        snes_set_dm,
        create_matrix,
        snes_set_type,
        global_x,
        local_x,
        global_b,
        local_b,
        snes_get_ksp,
        vec_replace_array,
        ksp_set_tols,
        ksp_set_type,
        ksp_get_pc,
        pc_set_type,
        ksp_set_from_ops
    )


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

    mat_get_dm = petsc_call('MatGetDM', [solver_objs['Jac'], Byref(objs['da'])])

    dm_get_app_context = petsc_call(
        'DMGetApplicationContext', [objs['da'], Byref(struct._C_symbol)])

    dm_get_local_xvec = petsc_call(
        'DMGetLocalVector', [objs['da'], Byref(solver_objs['X_local'])])

    global_to_local_begin = petsc_call(
        'DMGlobalToLocalBegin', [objs['da'], solver_objs['X_global'],
                                 'INSERT_VALUES', solver_objs['X_local']])

    global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
        objs['da'], solver_objs['X_global'], 'INSERT_VALUES', solver_objs['X_local']])

    dm_get_local_yvec = petsc_call(
        'DMGetLocalVector', [objs['da'], Byref(solver_objs['Y_local'])])

    vec_get_array_y = petsc_call(
        'VecGetArray', [solver_objs['Y_local'], Byref(petsc_arr_write.function)])

    vec_get_array_x = petsc_call(
        'VecGetArray', [solver_objs['X_local'], Byref(petsc_arrays[0])])

    dm_get_local_info = petsc_call('DMDAGetLocalInfo', [
        objs['da'], Byref(petsc_arrays[0].function.dmda_info)])

    casts = [PointerCast(i.function) for i in petsc_arrays]

    vec_restore_array_y = petsc_call(
        'VecRestoreArray', [solver_objs['Y_local'], Byref(petsc_arr_write)])

    vec_restore_array_x = petsc_call(
        'VecRestoreArray', [solver_objs['X_local'], Byref(petsc_arr_seed)])

    dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
        objs['da'], solver_objs['Y_local'], 'INSERT_VALUES', solver_objs['Y_global']])

    dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
        objs['da'], solver_objs['Y_local'], 'INSERT_VALUES', solver_objs['Y_global']])

    func_return = Call('PetscFunctionReturn', arguments=[0])

    matvec_body = List(body=[
        petsc_func_begin_user,
        defn_struct,
        mat_get_dm,
        dm_get_app_context,
        dm_get_local_xvec,
        global_to_local_begin,
        global_to_local_end,
        dm_get_local_yvec,
        vec_get_array_y,
        vec_get_array_x,
        dm_get_local_info,
        casts,
        BlankLine,
        body,
        vec_restore_array_y,
        vec_restore_array_x,
        dm_local_to_global_begin,
        dm_local_to_global_end,
        func_return])

    matvec_callback = Callable(
        'MyMatShellMult_%s' % target.name, matvec_body, retval=objs['err'],
        parameters=(solver_objs['Jac'], solver_objs['X_global'], solver_objs['Y_global']))

    matvec_operation = petsc_call(
        'MatShellSetOperation', [solver_objs['Jac'], 'MATOP_MULT',
                                 Callback(matvec_callback.name, void, void)])

    return matvec_callback, matvec_operation


def rebuild_expr_mapper(callable):

    return {expr: expr._rebuild(
        expr=expr.expr._rebuild(rhs=expr.expr.rhs.expr)) for
        expr in FindNodes(LinearSolverExpression).visit(callable)}


def uxreplace_efuncs(efuncs, struct):
    efuncs_new = []
    for efunc in efuncs:
        new_body = efunc.body
        for i in struct.usr_ctx:
            new_body = Uxreplace({i: FieldFromPointer(i, struct)}).visit(new_body)
        efuncs_new.append(efunc._rebuild(body=new_body))
    return efuncs_new


Null = Macro('NULL')
void = 'void'

# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
