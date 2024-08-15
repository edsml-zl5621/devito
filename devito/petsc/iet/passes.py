import cgen as c

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (FindNodes, Transformer,
                           MapNodes, Iteration, List, BlankLine,
                           Callable, CallableBody, DummyExpr, Call)
from devito.symbolics import Byref, Macro, FieldFromPointer
from devito.tools import filter_ordered
from devito.petsc.types import (PetscMPIInt, DM, Mat, LocalVec, GlobalVec,
                                KSP, PC, SNES, PetscErrorCode, DummyArg, PetscInt)
from devito.petsc.iet.nodes import InjectSolveDummy
from devito.petsc.utils import solver_mapper, core_metadata
from devito.petsc.iet.routines import PETScCallbackBuilder
from devito.petsc.iet.utils import (petsc_call, petsc_call_mpi, petsc_struct,
                                    spatial_iteration_loops)


@iet_pass
def lower_petsc(iet, **kwargs):

    # Check if PETScSolve was used
    petsc_nodes = FindNodes(InjectSolveDummy).visit(iet)

    if not petsc_nodes:
        return iet, {}

    unique_targets = list({i.expr.rhs.target for i in petsc_nodes})
    init = init_petsc(**kwargs)

    # Assumption is that all targets have the same grid so can use any target here
    objs = build_core_objects(unique_targets[-1], **kwargs)

    # Create core PETSc calls (not specific to each PETScSolve)
    core = make_core_petsc_calls(objs, **kwargs)

    # Create injectsolve mapper from the spatial iteration loops
    # (exclude time loop if present)
    spatial_body = spatial_iteration_loops(iet)
    injectsolve_mapper = MapNodes(Iteration, InjectSolveDummy,
                                  'groupby').visit(List(body=spatial_body))

    setup = []
    subs = {}

    # Create a different DMDA for each target with a unique space order
    unique_dmdas = create_dmda_objs(unique_targets)
    objs.update(unique_dmdas)
    for dmda in unique_dmdas.values():
        setup.extend(create_dmda_calls(dmda, objs))

    builder = PETScCallbackBuilder(**kwargs)

    # Create the PETSc calls which are specific to each target
    for target in unique_targets:
        solver_objs = build_solver_objs(target)

        # Generate the solver setup for target. This is required only
        # once per target
        for (injectsolve,) in injectsolve_mapper.values():
            # Skip if not associated with the target
            if injectsolve.expr.rhs.target != target:
                continue
            solver_setup = generate_solver_setup(solver_objs, objs, injectsolve, target)
            setup.extend(solver_setup)
            break

        # Generate all PETSc callback functions for the target via recusive compilation
        for iter, (injectsolve,) in injectsolve_mapper.items():
            if injectsolve.expr.rhs.target != target:
                continue
            matvec_op, formfunc_op, runsolve = builder.make(injectsolve,
                                                            objs, solver_objs)
            setup.extend([matvec_op, formfunc_op])
            subs.update({iter[0]: List(body=runsolve)})
            break

    # Generate callback to populate main struct object
    struct_main = petsc_struct('ctx', filter_ordered(builder.struct_params))
    struct_callback = generate_struct_callback(struct_main)
    call_struct_callback = petsc_call(struct_callback.name, [Byref(struct_main)])
    calls_set_app_ctx = [petsc_call('DMSetApplicationContext', [i, Byref(struct_main)])
                         for i in unique_dmdas]
    setup.extend([BlankLine, call_struct_callback] + calls_set_app_ctx)

    iet = Transformer(subs).visit(iet)

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
        'grid': target.grid,
        'localsize': PetscInt(name='localsize')
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


def build_solver_objs(target):
    name = target.name
    return {
        'Jac': Mat(name='J_%s' % name),
        'x_global': GlobalVec(name='x_global_%s' % name),
        'x_local': LocalVec(name='x_local_%s' % name, liveness='eager'),
        'b_global': GlobalVec(name='b_global_%s' % name),
        'b_local': LocalVec(name='b_local_%s' % name),
        'ksp': KSP(name='ksp_%s' % name),
        'pc': PC(name='pc_%s' % name),
        'snes': SNES(name='snes_%s' % name),
        'X_global': GlobalVec(name='X_global_%s' % name),
        'Y_global': GlobalVec(name='Y_global_%s' % name),
        'X_local': LocalVec(name='X_local_%s' % name, liveness='eager'),
        'Y_local': LocalVec(name='Y_local_%s' % name, liveness='eager'),
        'dummy': DummyArg(name='dummy_%s' % name)
    }


def generate_solver_setup(solver_objs, objs, injectsolve, target):
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
    body = [DummyExpr(FieldFromPointer(i._C_symbol, struct),
                      i._C_symbol) for i in struct.fields]
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
