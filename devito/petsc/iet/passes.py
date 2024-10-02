import cgen as c

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (Transformer, MapNodes, Iteration, List, BlankLine,
                           Callable, CallableBody, DummyExpr, Call)
from devito.symbolics import Byref, Macro, FieldFromPointer
from devito.tools import filter_ordered
from devito.petsc.types import (PetscMPIInt, DM, Mat, LocalVec, GlobalVec,
                                KSP, PC, SNES, PetscErrorCode, DummyArg, PetscInt)
from devito.petsc.iet.nodes import InjectSolveDummy
from devito.petsc.utils import solver_mapper, core_metadata
from devito.petsc.iet.routines import PETScCallbackBuilder
from devito.petsc.iet.utils import (petsc_call, petsc_call_mpi, petsc_struct,
                                    spatial_injectsolve_iter, assign_time_iters,
                                    retrieve_mod_dims)


@iet_pass
def lower_petsc(iet, **kwargs):
    # from IPython import embed; embed()
    # Check if PETScSolve was used
    injectsolve_mapper = MapNodes(Iteration, InjectSolveDummy,
                                  'groupby').visit(iet)

    if not injectsolve_mapper:
        return iet, {}

    targets = [i.expr.rhs.target for (i,) in injectsolve_mapper.values()]
    init = init_petsc(**kwargs)

    # Assumption is that all targets have the same grid so can use any target here
    objs = build_core_objects(targets[-1], **kwargs)

    # Create core PETSc calls (not specific to each PETScSolve)
    core = make_core_petsc_calls(objs, **kwargs)

    setup = []
    subs = {}

    # Create a different DMDA for each target with a unique space order
    unique_dmdas = create_dmda_objs(targets)
    objs.update(unique_dmdas)
    for dmda in unique_dmdas.values():
        setup.extend(create_dmda_calls(dmda, objs))

    builder = PETScCallbackBuilder(**kwargs)

    for iters, (injectsolve,) in injectsolve_mapper.items():
        target = injectsolve.expr.rhs.target
        solver_objs = build_solver_objs(target, **kwargs)

        # Generate the solver setup for each InjectSolveDummy
        solver_setup = generate_solver_setup(solver_objs, objs, injectsolve)
        setup.extend(solver_setup)

        # Retrieve `ModuloDimensions` for use in callback functions
        solver_objs['mod_dims'] = retrieve_mod_dims(iters)
        solver_objs['time_dim'] = injectsolve.expr.rhs.time_dim
        # from IPython import embed; embed()
        # new_callbacks = []
        # for callback in injectsolve.expr.rhs.formrhs:

        # Generate all PETSc callback functions for the target via recursive compilation
        matvec_op, formfunc_op, runsolve = builder.make(injectsolve,
                                                        objs, solver_objs)
        setup.extend([matvec_op, formfunc_op, BlankLine])
        # Only Transform the spatial iteration loop
        space_iter, = spatial_injectsolve_iter(iters, injectsolve)
        subs.update({space_iter: List(body=runsolve)})

    # Generate callback to populate main struct object
    struct_main = petsc_struct('ctx', filter_ordered(builder.struct_params))
    struct_callback = generate_struct_callback(struct_main)
    call_struct_callback = petsc_call(struct_callback.name, [Byref(struct_main)])
    calls_set_app_ctx = [petsc_call('DMSetApplicationContext', [i, Byref(struct_main)])
                         for i in unique_dmdas]
    setup.extend([call_struct_callback] + calls_set_app_ctx)

    iet = Transformer(subs).visit(iet)

    iet = assign_time_iters(iet, struct_main)

    body = core + tuple(setup) + (BlankLine,) + iet.body.body
    body = iet.body._rebuild(
        init=init, body=body,
        frees=(c.Line("PetscCall(PetscFinalize());"),)
    )
    iet = iet._rebuild(body=body)
    metadata = core_metadata()
    efuncs = tuple(builder.efuncs.values())+(struct_callback,)
    metadata.update({'efuncs': efuncs})

    return iet, metadata


def init_petsc(**kwargs):
    # Initialize PETSc -> for now, assuming all solver options have to be
    # specifed via the parameters dict in PETScSolve
    # TODO: Are users going to be able to use PETSc command line arguments?
    # In firedrake, they have an options_prefix for each solver, enabling the use
    # of command line options
    initialize = petsc_call('PetscInitialize', [Null, Null, Null, Null])

    return petsc_func_begin_user, initialize


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
    dm_mat_type = petsc_call('DMSetMatType', [dmda, 'MATSHELL'])
    dm_get_local_info = petsc_call('DMDAGetLocalInfo', [dmda, Byref(dmda.info)])
    return dmda_create, dm_setup, dm_mat_type, dm_get_local_info, BlankLine


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


def build_solver_objs(target, **kwargs):
    sreg = kwargs['sregistry']
    return {
        'Jac': Mat(sreg.make_name(prefix='J_')),
        'x_global': GlobalVec(sreg.make_name(prefix='x_global_')),
        'x_local': LocalVec(sreg.make_name(prefix='x_local_'), liveness='eager'),
        'b_global': GlobalVec(sreg.make_name(prefix='b_global_')),
        'b_local': LocalVec(sreg.make_name(prefix='b_local_')),
        'ksp': KSP(sreg.make_name(prefix='ksp_')),
        'pc': PC(sreg.make_name(prefix='pc_')),
        'snes': SNES(sreg.make_name(prefix='snes_')),
        'X_global': GlobalVec(sreg.make_name(prefix='X_global_')),
        'Y_global': GlobalVec(sreg.make_name(prefix='Y_global_')),
        'X_local': LocalVec(sreg.make_name(prefix='X_local_'), liveness='eager'),
        'Y_local': LocalVec(sreg.make_name(prefix='Y_local_'), liveness='eager'),
        'dummy': DummyArg(sreg.make_name(prefix='dummy_')),
        'localsize': PetscInt(sreg.make_name(prefix='localsize_'))
    }


def generate_solver_setup(solver_objs, objs, injectsolve):
    target = injectsolve.expr.rhs.target

    dmda = objs['da_so_%s' % target.space_order]

    solver_params = injectsolve.expr.rhs.solver_parameters

    snes_create = petsc_call('SNESCreate', [objs['comm'], Byref(solver_objs['snes'])])

    snes_set_dm = petsc_call('SNESSetDM', [solver_objs['snes'], dmda])

    create_matrix = petsc_call('DMCreateMatrix', [dmda, Byref(solver_objs['Jac'])])

    # NOTE: Assumming all solves are linear for now.
    snes_set_type = petsc_call('SNESSetType', [solver_objs['snes'], 'SNESKSPONLY'])

    snes_set_jac = petsc_call(
        'SNESSetJacobian', [solver_objs['snes'], solver_objs['Jac'],
                            solver_objs['Jac'], 'MatMFFDComputeJacobian', Null]
    )

    global_x = petsc_call('DMCreateGlobalVector',
                          [dmda, Byref(solver_objs['x_global'])])

    global_b = petsc_call('DMCreateGlobalVector',
                          [dmda, Byref(solver_objs['b_global'])])

    local_b = petsc_call('DMCreateLocalVector',
                         [dmda, Byref(solver_objs['b_local'])])

    snes_get_ksp = petsc_call('SNESGetKSP',
                              [solver_objs['snes'], Byref(solver_objs['ksp'])])

    ksp_set_tols = petsc_call(
        'KSPSetTolerances', [solver_objs['ksp'], solver_params['ksp_rtol'],
                             solver_params['ksp_atol'], solver_params['ksp_divtol'],
                             solver_params['ksp_max_it']]
    )

    ksp_set_type = petsc_call(
        'KSPSetType', [solver_objs['ksp'], solver_mapper[solver_params['ksp_type']]]
    )

    ksp_get_pc = petsc_call('KSPGetPC', [solver_objs['ksp'], Byref(solver_objs['pc'])])

    # Even though the default will be jacobi, set to PCNONE for now
    pc_set_type = petsc_call('PCSetType', [solver_objs['pc'], 'PCNONE'])

    ksp_set_from_ops = petsc_call('KSPSetFromOptions', [solver_objs['ksp']])

    return (
        snes_create,
        snes_set_dm,
        create_matrix,
        snes_set_jac,
        snes_set_type,
        global_x,
        global_b,
        local_b,
        snes_get_ksp,
        ksp_set_tols,
        ksp_set_type,
        ksp_get_pc,
        pc_set_type,
        ksp_set_from_ops
    )


def generate_struct_callback(struct):
    body = [
        DummyExpr(FieldFromPointer(i._C_symbol, struct), i._C_symbol)
        for i in struct.fields if i not in struct.time_dim_fields
    ]
    struct_callback_body = CallableBody(
        List(body=body), init=tuple([petsc_func_begin_user]),
        retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])])
    )
    struct_callback = Callable(
        'PopulateMatContext', struct_callback_body, PetscErrorCode(name='err'),
        parameters=[struct]
    )
    return struct_callback


Null = Macro('NULL')
void = 'void'

# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
