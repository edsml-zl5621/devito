from collections import OrderedDict
import cgen as c

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (FindNodes, Call,
                           Transformer, FindSymbols,
                           MapNodes, Iteration, Callable, Callback, List, Uxreplace,
                           Definition, BlankLine, PointerCast, filter_iterations,
                           retrieve_iteration_tree)
from devito.symbolics import Byref, Macro, FieldFromPointer
from devito.petsc.types import (PetscMPIInt, PETScStruct, DM, Mat,
                                Vec, KSP, PC, SNES, PetscErrorCode, PETScArray)
from devito.petsc.iet.nodes import MatVecAction, LinearSolverExpression
from devito.petsc.utils import (solver_mapper, petsc_call, petsc_call_mpi,
                                core_metadata)


@iet_pass
def lower_petsc(iet, **kwargs):

    # Check if PETScSolve was used
    petsc_nodes = FindNodes(MatVecAction).visit(iet)

    if not petsc_nodes:
        return iet, {}

    # Collect all petsc solution fields
    unique_targets = list({i.expr.rhs.target for i in petsc_nodes})
    init = init_petsc(**kwargs)

    # Assumption is that all targets have the same grid so can use any target here
    objs = build_core_objects(unique_targets[-1], **kwargs)
    objs['struct'] = build_petsc_struct(iet)

    # Create core PETSc calls (not specific to each PETScSolve)
    core = make_core_petsc_calls(objs, **kwargs)

    # Create matvec mapper from the spatial iteration loops (exclude time loop if present)
    spatial_body = []
    for tree in retrieve_iteration_tree(iet):
        root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]
        spatial_body.append(root)
    matvec_mapper = MapNodes(Iteration, MatVecAction,
                             'groupby').visit(List(body=spatial_body))

    setup = []

    # Create a different DMDA for each target with a unique space order
    unique_dmdas = create_dmda_objs(unique_targets)
    objs.update(unique_dmdas)
    for dmda in unique_dmdas.values():
        setup.extend(create_dmda_calls(dmda, objs))

    subs = {}
    efuncs = OrderedDict()

    # Create the PETSc calls which are specific to each target
    for target in unique_targets:
        solver_objs = build_solver_objs(target)

        # Generate solver setup for target
        for iter, (matvec,) in matvec_mapper.items():
            # Skip the MatVecAction if it is not associated with the target
            # There will most likely be more than one MatVecAction
            # associated with the target e.g interior matvec + BC matvecs
            if matvec.expr.rhs.target != target:
                continue
            solver = generate_solver_calls(solver_objs, objs, matvec, target)
            setup.extend(solver)
            break
        
        # Create the body of the matrix-vector callback for target
        matvec_body_list = []
        for iter, (matvec,) in matvec_mapper.items():
            if matvec.expr.rhs.target != target:
                continue
            matvec_body_list.append(iter[0])
            # Remove the iteration loop from the main kernel encapsulating
            # the matvec equations since they are moved into the callback
            subs.update({iter[0]: None})

        # Create the matvec callback and operation for each target
        matvec_callback, matvec_op = create_matvec_callback(
            target, List(body=matvec_body_list), solver_objs, objs
        )

        setup.extend([matvec_op, BlankLine])
        efuncs[matvec_callback.name] = matvec_callback

    # Remove the LinSolveExpr's from iet and efuncs
    subs.update(rebuild_expr_mapper(iet))
    iet = Transformer(subs).visit(iet)
    efuncs = transform_efuncs(efuncs, objs['struct'])

    body = iet.body._rebuild(init=init, body=core+tuple(setup)+iet.body.body)
    iet = iet._rebuild(body=body)

    metadata = core_metadata()
    metadata.update({'efuncs': tuple(efuncs.values())})

    return iet, metadata


def init_petsc(**kwargs):
    # Initialize PETSc -> for now, assuming all solver options have to be
    # specifed via the parameters dict in PETScSolve
    # TODO: Are users going to be able to use PETSc command line arguments?
    # In firedrake, they have an options_prefix for each solver, enabling the use
    # of command line options
    initialize = petsc_call('PetscInitialize', [Null, Null, Null, Null])

    return petsc_func_begin_user, initialize


def build_petsc_struct(iet):
    # Place all context data required by the shell routines
    # into a PETScStruct
    usr_ctx = []
    basics = FindSymbols('basics').visit(iet)
    avoid = FindSymbols('dimensions|indexedbases').visit(iet)
    usr_ctx.extend(data for data in basics if data not in avoid)

    return PETScStruct('ctx', usr_ctx)


def make_core_petsc_calls(objs, **kwargs):
    call_mpi = petsc_call_mpi('MPI_Comm_size', [objs['comm'], Byref(objs['size'])])

    return call_mpi, BlankLine


def build_core_objects(target, **kwargs):
    if kwargs['options']['mpi']:
        communicator = target.grid.distributor._obj_comm
    else:
        communicator = 'PETSC_COMM_SELF'

    return {
        'size': PetscMPIInt(name='size'),
        'comm': communicator,
        'err': PetscErrorCode(name='err'),
        'grid': target.grid
    }


def create_dmda_objs(unique_targets):
    unique_dmdas = {}
    for target in unique_targets:
        name = 'da_so_%s' % target.space_order
        unique_dmdas[name] = DM(name=name, liveness='eager',
                                stencil_width=target.space_order)
    return unique_dmdas


def create_dmda_calls(dmda, objs):
    dmda_create = create_dmda(dmda, objs)
    dm_setup = petsc_call('DMSetUp', [dmda])
    dm_app_ctx = petsc_call('DMSetApplicationContext', [dmda, objs['struct']])
    dm_mat_type = petsc_call('DMSetMatType', [dmda, 'MATSHELL'])

    return dmda_create, dm_setup, dm_app_ctx, dm_mat_type, BlankLine


def create_dmda(dmda, objs):
    no_of_space_dims = len(objs['grid'].dimensions)

    # MPI communicator
    args = [objs['comm']]

    # Type of ghost nodes
    args.extend(['DM_BOUNDARY_GHOSTED' for _ in range(no_of_space_dims)])

    # Stencil type
    if no_of_space_dims > 1:
        args.append('DMDA_STENCIL_BOX')

    # Global dimensions
    args.extend(list(objs['grid'].shape)[::-1])
    # No.of processors in each dimension
    if no_of_space_dims > 1:
        args.extend(list(objs['grid'].distributor.topology)[::-1])

    # Number of degrees of freedom per node
    args.append(1)
    # "Stencil width" -> size of overlap
    args.append(dmda.stencil_width)
    args.extend([Null for _ in range(no_of_space_dims)])

    # The distributed array object
    args.append(Byref(dmda))

    # The PETSc call used to create the DMDA
    dmda = petsc_call('DMDACreate%sd' % no_of_space_dims, args)

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
    dmda = objs['da_so_%s' % target.space_order]

    solver_params = matvec.expr.rhs.solver_parameters

    snes_create = petsc_call('SNESCreate', [objs['comm'], Byref(solver_objs['snes'])])

    snes_set_dm = petsc_call('SNESSetDM', [solver_objs['snes'], dmda])

    create_matrix = petsc_call('DMCreateMatrix', [dmda, Byref(solver_objs['Jac'])])

    # NOTE: Assumming all solves are linear for now.
    snes_set_type = petsc_call('SNESSetType', [solver_objs['snes'], 'SNESKSPONLY'])

    global_x = petsc_call('DMCreateGlobalVector',
                          [dmda, Byref(solver_objs['x_global'])])

    local_x = petsc_call('DMCreateLocalVector',
                         [dmda, Byref(solver_objs['x_local'])])

    global_b = petsc_call('DMCreateGlobalVector',
                          [dmda, Byref(solver_objs['b_global'])])

    local_b = petsc_call('DMCreateLocalVector',
                         [dmda, Byref(solver_objs['b_local'])])

    snes_get_ksp = petsc_call('SNESGetKSP',
                              [solver_objs['snes'], Byref(solver_objs['ksp'])])

    vec_replace_array = petsc_call(
        'VecReplaceArray', [solver_objs['x_local'],
                            FieldFromPointer(target._C_field_data, target._C_symbol)]
    )

    ksp_set_tols = petsc_call(
        'KSPSetTolerances', [solver_objs['ksp'], solver_params['ksp_rtol'],
                             solver_params['ksp_atol'], solver_params['ksp_divtol'],
                             solver_params['ksp_max_it']]
    )

    ksp_set_type = petsc_call(
        'KSPSetType', [solver_objs['ksp'], solver_mapper[solver_params['ksp_type']]]
    )

    ksp_get_pc = petsc_call('KSPGetPC', [solver_objs['ksp'], Byref(solver_objs['pc'])])

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


def create_matvec_callback(target, body, solver_objs, objs):
    dmda = objs['da_so_%s' % target.space_order]

    # There will be 2 PETScArrays within the body
    petsc_arrays = [i for i in FindSymbols('indexedbases').visit(body)
                    if isinstance(i.function, PETScArray)]

    # There will only be one PETScArray that is written to within this body and
    # one PETScArray which corresponds to the 'seed' vector
    petsc_arr_write, = FindSymbols('writes').visit(body)
    petsc_arr_seed, = [i.function for i in petsc_arrays
                       if i.function != petsc_arr_write.function]

    # Struct needs to be defined explicitly here since CompositeObjects
    # do not have 'liveness'
    defn_struct = Definition(objs['struct'])

    mat_get_dm = petsc_call('MatGetDM', [solver_objs['Jac'], Byref(dmda)])

    dm_get_app_context = petsc_call(
        'DMGetApplicationContext', [dmda, Byref(objs['struct']._C_symbol)])

    dm_get_local_xvec = petsc_call(
        'DMGetLocalVector', [dmda, Byref(solver_objs['X_local'])])

    global_to_local_begin = petsc_call(
        'DMGlobalToLocalBegin', [dmda, solver_objs['X_global'],
                                 'INSERT_VALUES', solver_objs['X_local']])

    global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
        dmda, solver_objs['X_global'], 'INSERT_VALUES', solver_objs['X_local']])

    dm_get_local_yvec = petsc_call(
        'DMGetLocalVector', [dmda, Byref(solver_objs['Y_local'])])

    vec_get_array_y = petsc_call(
        'VecGetArray', [solver_objs['Y_local'], Byref(petsc_arr_write._C_symbol)])

    vec_get_array_x = petsc_call(
        'VecGetArray', [solver_objs['X_local'], Byref(petsc_arr_seed._C_symbol)])

    dm_get_local_info = petsc_call('DMDAGetLocalInfo', [
        dmda, Byref(petsc_arr_seed.function.dmda_info)])

    casts = [PointerCast(i.function) for i in petsc_arrays]

    vec_restore_array_y = petsc_call(
        'VecRestoreArray', [solver_objs['Y_local'], Byref(petsc_arr_write._C_symbol)])

    vec_restore_array_x = petsc_call(
        'VecRestoreArray', [solver_objs['X_local'], Byref(petsc_arr_seed._C_symbol)])

    dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
        dmda, solver_objs['Y_local'], 'INSERT_VALUES', solver_objs['Y_global']])

    dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
        dmda, solver_objs['Y_local'], 'INSERT_VALUES', solver_objs['Y_global']])

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
    # This mapper removes LinSolveExpr instances from the callable
    # These expressions were previously used in lower_petc to carry metadata,
    # such as solver_parameters
    nodes = FindNodes(LinearSolverExpression).visit(callable)
    return {expr: expr._rebuild(
        expr=expr.expr._rebuild(rhs=expr.expr.rhs.expr)) for expr in nodes}


def transform_efuncs(efuncs, struct):
    subs = {i: FieldFromPointer(i, struct) for i in struct.usr_ctx}
    for efunc in efuncs.values():
        transformed_efunc = Transformer(rebuild_expr_mapper(efunc)).visit(efunc)
        transformed_efunc = Uxreplace(subs).visit(transformed_efunc)
        efuncs[efunc.name] = transformed_efunc
    return efuncs


Null = Macro('NULL')
void = 'void'

# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
