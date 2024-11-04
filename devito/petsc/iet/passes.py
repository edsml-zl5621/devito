import cgen as c

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (Transformer, MapNodes, Iteration, List, BlankLine,
                           DummyExpr, FindNodes, retrieve_iteration_tree,
                           filter_iterations, CallableBody, Call, Callable)
from devito.symbolics import Byref, Macro, FieldFromComposite, FieldFromPointer
from devito.petsc.types import (PetscMPIInt, Mat, CallbackDM, LocalVec, GlobalVec,
                                KSP, PC, SNES, PetscErrorCode, DummyArg, PetscInt,
                                StartPtr, FieldDataNest, DMComposite, IS, SubMat)
from devito.petsc.iet.nodes import InjectSolveDummy, PETScCall
from devito.petsc.utils import solver_mapper, core_metadata
from devito.petsc.iet.routines import PETScCallbackBuilder
from devito.petsc.iet.utils import petsc_call, petsc_call_mpi, petsc_struct
from devito.tools import filter_ordered


@iet_pass
def lower_petsc(iet, **kwargs):
    # Check if PETScSolve was used
    injectsolve_mapper = MapNodes(Iteration, InjectSolveDummy,
                                  'groupby').visit(iet)

    if not injectsolve_mapper:
        return iet, {}

    targets = [i.expr.lhs.function for (i,) in injectsolve_mapper.values()]
    init = init_petsc(**kwargs)

    # Assumption is that all targets have the same grid so can use any target here
    objs = build_core_objects(targets[-1], **kwargs)

    # Create core PETSc calls (not specific to each PETScSolve)
    core = make_core_petsc_calls(objs, **kwargs)

    setup = []
    subs = {}
    
    efuncs = []
    struct_params = []
    # builder = PETScCallbackBuilder(**kwargs)
    # from IPython import embed; embed()
    for iters, (injectsolve,) in injectsolve_mapper.items():
        # There should be a builder per "snes solve"
        # TOOD: if solve is NEST, use a PETSCCallbackNestedBuilder or something...
        builder = PETScCallbackBuilder(**kwargs)
        #TODO: THIS IS WHERE WE CHECK IF FIELD_DATA IS NEST OR NOT
        #TODO: INSTEAD OF GRABBING THE LHS OF INJECTSOLVEDUMMY FOR THE VECCREPLACEARRAY,
        # YOU WILL HAVE TO SEARCH THE RHS .expr FOR THE INDEXIFIED target and use that instead 
        # potentially when you build the solver_objs you can create one called target and search the exprs for the one
        # that matches the target attached to the field data ......
        linsolve = injectsolve.expr.rhs
        # data = injectsolve.expr.rhs.fielddata
        solver_objs = build_solver_objs(linsolve, iters, **kwargs)

        # Setup DMs
        dm_setup = generate_dm_setup(objs, linsolve)
        setup.extend(dm_setup)
        # Setup solver
        solver_setup = generate_solver_setup(solver_objs, objs, linsolve)
        setup.extend(solver_setup)

        # # Generate all PETSc callback functions for the target via recursive compilation
        callback_setup, runsolve = builder.make(
            linsolve, objs, solver_objs
        )
        setup.extend(callback_setup)
        # # Only Transform the spatial iteration loop
        space_iter, = spatial_injectsolve_iter(iters, injectsolve)
        subs.update({space_iter: List(body=runsolve)})
        objs['dmdas'].append(linsolve.parent_dm)
        # from IPython import embed; embed()
        efuncs.extend(builder.efuncs.values())
        struct_params.extend(builder.struct_params)

    # Generate callback to populate main struct object
    # TODO: move all struct stuff into a single function
    struct_main = objs['struct']._rebuild(fields=filter_ordered(struct_params))
    struct_callback = generate_struct_callback(struct_main, objs)
    efuncs.append(struct_callback)
    struct_calls = make_struct_calls(struct_callback, struct_main, objs)
    # from IPython import embed; embed()
    # TODO: clean this up
    setup.extend(list(struct_calls))

    # builder.make_main_struct(objs)
    # setup.extend(struct_calls)
    # from IPython import embed; embed()
    iet = Transformer(subs).visit(iet)
    
    # Assign time iterators 
    # THIS SHOULD BE established within the builder I think
    # Perhaps assign all time iters necessary then drop duplicates
    # iet = assign_time_iters(iet, struct)

    body = core + tuple(setup) + (BlankLine,) + iet.body.body
    body = iet.body._rebuild(
        init=init, body=body,
        frees=(c.Line("PetscCall(PetscFinalize());"),)
    )
    iet = iet._rebuild(body=body)
    metadata = core_metadata()
    # efuncs = tuple(builder.efuncs.values())
    metadata.update({'efuncs': tuple(efuncs)})
    # from IPython import embed; embed()
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
    #TODO: think i can remove dmda here
    return {
        'size': PetscMPIInt(name='size'),
        'comm': communicator,
        'err': PetscErrorCode(name='err'),
        'grid': target.grid,
        'dmdas': [],
        'struct': petsc_struct(name='ctx')
    }


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


# def build_solver_objs(injectsolve, iters, **kwargs):
#     from IPython import embed; embed()
#     # solver_objs = {}
#     field_objs = {}

#     solver_objs = 
#     data = injectsolve.expr.rhs.fielddata
#     if instance(data, FieldDataNest):



#     sreg = kwargs['sregistry']
#     return {
#         'Jac': Mat(sreg.make_name(prefix='J_')),
#         'x_global': GlobalVec(sreg.make_name(prefix='x_global_')),
#         'x_local': LocalVec(sreg.make_name(prefix='x_local_'), liveness='eager'),
#         'b_global': GlobalVec(sreg.make_name(prefix='b_global_')),
#         'b_local': LocalVec(sreg.make_name(prefix='b_local_')),
#         'ksp': KSP(sreg.make_name(prefix='ksp_')),
#         'pc': PC(sreg.make_name(prefix='pc_')),
#         'snes': SNES(sreg.make_name(prefix='snes_')),
#         'X_global': GlobalVec(sreg.make_name(prefix='X_global_')),
#         'Y_global': GlobalVec(sreg.make_name(prefix='Y_global_')),
#         'X_local': LocalVec(sreg.make_name(prefix='X_local_'), liveness='eager'),
#         'Y_local': LocalVec(sreg.make_name(prefix='Y_local_'), liveness='eager'),
#         'dummy': DummyArg(sreg.make_name(prefix='dummy_')),
#         'localsize': PetscInt(sreg.make_name(prefix='localsize_')),
#         'start_ptr': StartPtr(sreg.make_name(prefix='start_ptr_'), rhs.target.dtype),
#         'true_dims': retrieve_time_dims(iters),
#         'target': rhs.target,
#         'time_mapper': rhs.time_mapper,
#         'CallbackDM': CallbackDM(rhs.dmda.name, stencil_width=rhs.dmda.stencil_width),
#     }


def build_solver_objs(linsolve, iters, **kwargs):
    sreg = kwargs['sregistry']
    # TODO: change y to F etc
    solver_objs = {
        'Jac': Mat(sreg.make_name(prefix='J_')),
        'ksp': KSP(sreg.make_name(prefix='ksp_')),
        'pc': PC(sreg.make_name(prefix='pc_')),
        'snes': SNES(sreg.make_name(prefix='snes_')),
        'dummy': DummyArg(sreg.make_name(prefix='dummy_')),
        'localsize': PetscInt(sreg.make_name(prefix='localsize_')),
        'true_dims': retrieve_time_dims(iters),
        'x_global': GlobalVec(sreg.make_name(prefix='x_global_')),
        'b_global': GlobalVec(sreg.make_name(prefix='b_global_')),
        'X_global': GlobalVec(sreg.make_name(prefix='X_global_')),
        'Y_global': GlobalVec(sreg.make_name(prefix='Y_global_')),
        'F_global': GlobalVec(sreg.make_name(prefix='F_global_')),
        'time_mapper': linsolve.time_mapper,
    }
    func = build_objs_nest if isinstance(linsolve.fielddata, FieldDataNest) else build_field_objs
    solver_objs.update(func(linsolve.fielddata, sreg))

    return solver_objs


def build_field_objs(fielddata, sreg):
    target = fielddata.target
    name = target.name
    #TODO: dont think i need double local vecs ..
    # plus can probbaly get rid of y?
    return {
        'x_local_%s' % name: LocalVec(sreg.make_name(prefix='x_local_'), liveness='eager'),
        'b_local_%s' % name: LocalVec(sreg.make_name(prefix='b_local_')),
        'X_local_%s' % name: LocalVec(sreg.make_name(prefix='X_local_'), liveness='eager'),
        'Y_local_%s' % name: LocalVec(sreg.make_name(prefix='Y_local_'), liveness='eager'),
        'F_local_%s' % name: LocalVec(sreg.make_name(prefix='F_local_'), liveness='eager'),
        'start_ptr_%s' % name: StartPtr(sreg.make_name(prefix='start_ptr_'), target.dtype),
    }



def build_objs_nest(fielddata, sreg):
    objs = {}
    targets = fielddata.targets 

    for field_data in fielddata.field_data_list:
        objs.update(build_field_objs(field_data, sreg))

    nest_objs = {
        'DMComposite': DMComposite(sreg.make_name(prefix='pack_'), targets=targets),
        # TODO: fix .. this is wrong
        'indexset': IS(name='is_', nindices=len(targets)),
    }
    sub_mats = {
        # submatrices
        'J%s%s' % (t1.name, t2.name): SubMat(name='J%s%s' % (t1.name, t2.name))
        for t1 in targets
        for t2 in targets
    }

    objs.update(nest_objs)
    objs.update(sub_mats)
    return objs


def generate_dm_setup(objs, linsolve):
    params = linsolve.solver_parameters
    parent_dm = linsolve.parent_dm
    children_dms = linsolve.children_dms
    calls = []

    for dm in children_dms:
        calls.append(create_dmda(dm, objs))
        calls.append(petsc_call('DMSetUp', [dm]))
        calls.append(petsc_call('DMSetMatType', [dm, 'MATSHELL']))
        calls.append(BlankLine)

    if isinstance(parent_dm, DMComposite):
        calls.append(petsc_call('DMCompositeCreate', [objs['comm'], Byref(parent_dm)]))
        calls.extend([petsc_call('DMCompositeAddDM', [parent_dm, child]) for child in children_dms])
        calls.append(petsc_call('DMSetMatType', [parent_dm, 'MATNEST']))
        calls.append(BlankLine)

    return tuple(calls)


def generate_solver_setup(solver_objs, objs, linsolve):
    dm = linsolve.parent_dm
    solver_params = linsolve.solver_parameters

    snes_create = petsc_call('SNESCreate', [objs['comm'], Byref(solver_objs['snes'])])

    snes_set_dm = petsc_call('SNESSetDM', [solver_objs['snes'], dm])

    create_matrix = petsc_call('DMCreateMatrix', [dm, Byref(solver_objs['Jac'])])

    # NOTE: Assumming all solves are linear for now.
    snes_set_type = petsc_call('SNESSetType', [solver_objs['snes'], 'SNESKSPONLY'])
    
    # # TOOD: move this to builder -> makes more sense 
    # snes_set_jac = petsc_call(
    #     'SNESSetJacobian', [solver_objs['snes'], solver_objs['Jac'],
    #                         solver_objs['Jac'], 'MatMFFDComputeJacobian', Null]
    # )

    global_x = petsc_call('DMCreateGlobalVector',
                          [dm, Byref(solver_objs['x_global'])])

    global_b = petsc_call('DMCreateGlobalVector',
                          [dm, Byref(solver_objs['b_global'])])

    # insert calls to get local b vectors (maybe not local x vectors not sure yet)
    if isinstance(dm, DMComposite):
        targets = dm.targets
        local_b_refs = [Byref(solver_objs['b_local_%s'%target.name]) for target in targets]
        local_b = petsc_call('DMCompositeGetLocalVectors', [dm] + local_b_refs)

    else:
        local_b = petsc_call('DMCreateLocalVector',
                             [dm, Byref(solver_objs['b_local_%s'%linsolve.fielddata.target.name])])

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
        # dmda_create,
        # dm_setup,
        # dm_mat_type,
        snes_create,
        snes_set_dm,
        create_matrix,
        # snes_set_jac,
        snes_set_type,
        global_x,
        global_b,
        local_b,
        snes_get_ksp,
        ksp_set_tols,
        ksp_set_type,
        ksp_get_pc,
        pc_set_type,
        ksp_set_from_ops,
        BlankLine
    )


def assign_time_iters(iet, struct):
    """
    Assign time iterators to the struct within loops containing PETScCalls.
    Ensure that assignment occurs only once per time loop, if necessary.
    Assign only the iterators that are common between the struct fields
    and the actual Iteration.
    """
    time_iters = [
        i for i in FindNodes(Iteration).visit(iet)
        if i.dim.is_Time and FindNodes(PETScCall).visit(i)
    ]

    if not time_iters:
        return iet

    mapper = {}
    for iter in time_iters:
        common_dims = [d for d in iter.dimensions if d in struct.fields]
        common_dims = [
            DummyExpr(FieldFromComposite(d, struct), d) for d in common_dims
        ]
        iter_new = iter._rebuild(nodes=List(body=tuple(common_dims)+iter.nodes))
        mapper.update({iter: iter_new})

    return Transformer(mapper).visit(iet)


def retrieve_time_dims(iters):
    time_iter = [i for i in iters if any(d.is_Time for d in i.dimensions)]
    mapper = {}
    if not time_iter:
        return mapper
    for d in time_iter[0].dimensions:
        if d.is_Modulo:
            mapper[d.origin] = d
        elif d.is_Time:
            mapper[d] = d
    return mapper


def spatial_injectsolve_iter(iter, injectsolve):
    spatial_body = []
    for tree in retrieve_iteration_tree(iter[0]):
        root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]
        if injectsolve in FindNodes(InjectSolveDummy).visit(root):
            spatial_body.append(root)
    return spatial_body


def generate_struct_callback(struct, objs):
    body = [
        DummyExpr(FieldFromPointer(i._C_symbol, struct), i._C_symbol)
        for i in struct.fields if i not in struct.time_dim_fields
    ]
    struct_callback_body = CallableBody(
        List(body=body), init=tuple([petsc_func_begin_user]),
        retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])])
    )
    struct_callback = Callable(
        'PopulateMatContext', struct_callback_body, objs['err'],
        parameters=[struct]
    )
    return struct_callback


def make_struct_calls(struct_callback, struct_main, objs):
    # struct_main = objs['struct']
    # struct_main = petsc_struct('ctx', filter_ordered(self.struct_params))
    # struct_main = objs['struct']._rebuild(fields=self.struct_params)
    # from IPython import embed; embed()
    # struct_main = objs['struct']._rebuild(fields=filter_ordered(self.struct_params), liveness='lazy')
    # struct_main = petsc_struct('ctx', filter_ordered(self.struct_params))
    # struct_callback = self.generate_struct_callback(struct_main, objs)
    call_struct_callback = [petsc_call(struct_callback.name, [Byref(struct_main)])]
    calls_set_app_ctx = [
        petsc_call('DMSetApplicationContext', [i, Byref(struct_main)])
        for i in objs['dmdas']
    ]
    calls = call_struct_callback + calls_set_app_ctx
    # self._efuncs[struct_callback.name] = struct_callback
    return tuple(calls)


Null = Macro('NULL')
void = 'void'

# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
