import cgen as c

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (Transformer, MapNodes, Iteration, List, BlankLine,
                           DummyExpr, FindNodes, retrieve_iteration_tree,
                           filter_iterations, CallableBody, Call, Callable,
                           Uxreplace)
from devito.symbolics import Byref, Macro, FieldFromComposite, FieldFromPointer
from devito.petsc.types import (PetscMPIInt, Mat, LocalVec, GlobalVec,
                                KSP, PC, SNES, PetscErrorCode, DummyArg, PetscInt,
                                StartPtr, FieldDataNest, IS, SubMat,
                                PETScStruct, CallbackStruct)
from devito.petsc.iet.nodes import InjectSolveDummy, PETScCall
from devito.petsc.utils import solver_mapper, core_metadata
from devito.petsc.iet.routines import NestedCallbackBuilder, CallbackBuilder
from devito.petsc.iet.utils import petsc_call, petsc_call_mpi
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

    # Shared between each solve
    setup, struct_params = [], []
    efuncs, subs = {}, {}

    # Specific to each solve
    for iters, (injectsolve,) in injectsolve_mapper.items():
        linsolve = injectsolve.expr.rhs
        objs['dmdas'].extend(linsolve.dms)

        sreg = kwargs['sregistry']
        ObjBuilder, CBBuilder, SolverSetup, DMSetup = get_builder_classes(linsolve)

        solver_objs = ObjBuilder(sregistry=sreg).build(linsolve, iters)
        builder = CBBuilder(**kwargs)

        dm_setup = DMSetup().setup(objs, linsolve)
        setup.extend(dm_setup)
        solver_setup = SolverSetup().setup(solver_objs, objs, linsolve)
        setup.extend(solver_setup)

        # # Generate all PETSc callback functions for the target via recursive compilation
        callback_setup, runsolve = builder.make(
            linsolve, objs, solver_objs
        )
        setup.extend(callback_setup)
        # Only Transform the spatial iteration loop
        space_iter, = spatial_injectsolve_iter(iters, injectsolve)
        subs.update({space_iter: List(body=runsolve)})

        efuncs.update(builder.efuncs)
        struct_params.extend(builder.struct_params)

    # Build and run struct callback
    struct_params = filter_ordered(struct_params)
    struct, struct_calls, struct_callback = struct_setup(struct_params, objs)
    efuncs.update({struct_callback.name: struct_callback})
    setup.extend(list(struct_calls))

    efuncs = uxreplace_efuncs(struct_params, efuncs, objs)

    iet = Transformer(subs).visit(iet)

    # Assign time iterators
    iet = assign_time_iters(iet, struct)

    body = core + tuple(setup) + (BlankLine,) + iet.body.body
    body = iet.body._rebuild(
        init=init, body=body,
        frees=(c.Line("PetscCall(PetscFinalize());"),)
    )
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


def make_core_petsc_calls(objs, **kwargs):
    call_mpi = petsc_call_mpi('MPI_Comm_size', [objs['comm'], Byref(objs['size'])])

    return call_mpi, BlankLine


def build_core_objects(target, **kwargs):
    if kwargs['options']['mpi']:
        communicator = target.grid.distributor._obj_comm
    else:
        communicator = 'PETSC_COMM_SELF'
    # TODO: think i can remove dmda here
    return {
        'size': PetscMPIInt(name='size'),
        'comm': communicator,
        'err': PetscErrorCode(name='err'),
        'grid': target.grid,
        'dmdas': [],
        # callback struct, fields are added after all solves are built
        'cbstruct': CallbackStruct('ctx', pname='MatContext', fields=[])
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


class ObjBuilder:
    def __new__(cls, sregistry=None, **kwargs):
        obj = object.__new__(cls)
        obj.sregistry = sregistry
        return obj

    def build(self, linsolve, iters):
        sreg = self.sregistry
        # TODO: change y to F etc
        solverobjs = {
            'J': Mat(sreg.make_name(prefix='J_')),
            'B': Mat(sreg.make_name(prefix='B_')),
            'ksp': KSP(sreg.make_name(prefix='ksp_')),
            'pc': PC(sreg.make_name(prefix='pc_')),
            'snes': SNES(sreg.make_name(prefix='snes_')),
            'dummy': DummyArg(sreg.make_name(prefix='dummy_')),
            'true_dims': retrieve_time_dims(iters),
            'x_global': GlobalVec(sreg.make_name(prefix='x_global_')),
            'b_global': GlobalVec(sreg.make_name(prefix='b_global_')),
            'X_global': GlobalVec(sreg.make_name(prefix='X_global_')),
            'Y_global': GlobalVec(sreg.make_name(prefix='Y_global_')),
            'F_global': GlobalVec(sreg.make_name(prefix='F_global_')),
            'time_mapper': linsolve.time_mapper
        }
        solverobjs.update(self.all_field_objs(linsolve.fielddata))
        return solverobjs

    def all_field_objs(self, fielddata):
        return self.field_objs(fielddata)

    def field_objs(self, fielddata):
        """
        Generate all PETSc objects required for a single field
        """
        t = fielddata.target
        name = t.name
        sreg = self.sregistry
        # TODO: dont think I need double local vecs ..
        # plus can probbaly get rid of y?
        prefixes = ['x', 'y', 'b', 'X', 'Y', 'F']

        field_objs = {
            '%s_local_%s' % (p, name): LocalVec(
                sreg.make_name(prefix='%s_local_' % p), liveness='eager'
            ) for p in prefixes
        }
        field_objs.update({
            'start_ptr_%s' % name: StartPtr(sreg.make_name(prefix='start_ptr_'), t.dtype),
            'localsize_%s' % name: PetscInt(sreg.make_name(prefix='localsize_')),
            fielddata.dmda.name: fielddata.dmda,
        })
        field_objs.update(fielddata.arrays)
        return field_objs


class NestedObjBuilder(ObjBuilder):
    def all_field_objs(self, fielddata):
        objs = {}
        targets = fielddata.targets

        for field_data in fielddata.field_data_list:
            objs.update(self.field_objs(field_data))

        sub_mats = {
            # submatrices
            'B%s%s' % (t1.name, t2.name): SubMat(
                name='B%s%s' % (t1.name, t2.name), row=i, col=j
            ) for i, t1 in enumerate(targets) for j, t2 in enumerate(targets)
        }
        nest_objs = {
            # TODO: remove restrict qualifier
            'indexset': IS(name='is', nindices=2),
            'targets': targets
        }
        objs.update(nest_objs)
        objs.update(sub_mats)
        return objs


def get_builder_classes(linsolve):
    """
    Selects the appropriate classes to build/run this solve.
    This function is designed to support future extensions, enabling
    different combinations of solver types, preconditioning methods,
    and other functionalities as needed.
    """
    if isinstance(linsolve.fielddata, FieldDataNest):
        return NestedObjBuilder, NestedCallbackBuilder, NestedSetupSolver, NestedSetupDM
    return ObjBuilder, CallbackBuilder, SetupSolver, SetupDM


class SetupDM:
    def setup(self, objs, linsolve):
        parent_dm = linsolve.parent_dm
        children_dms = linsolve.children_dms

        # TODO: change name maybe? not children/parent?
        children_dm_calls = self.setup_children(children_dms, objs)
        parent_dm_calls = self.setup_parent(parent_dm, children_dms, objs)
        return children_dm_calls + parent_dm_calls

    def setup_children(self, children_dms, objs):
        calls = []
        for dm in children_dms:
            calls.append(create_dmda(dm, objs))
            calls.append(petsc_call('DMSetUp', [dm]))
            calls.append(petsc_call('DMSetMatType', [dm, 'MATSHELL']))
            calls.append(BlankLine)
        return tuple(calls)

    def setup_parent(self, pdm, cdms, objs):
        return ()


class NestedSetupDM(SetupDM):
    def setup_parent(self, pdm, cdms, objs):
        calls = []
        calls.append(petsc_call('DMCompositeCreate', [objs['comm'], Byref(pdm)]))
        calls.extend([petsc_call('DMCompositeAddDM', [pdm, cdm]) for cdm in cdms])
        calls.append(petsc_call('DMSetMatType', [pdm, 'MATNEST']))
        calls.append(BlankLine)
        return tuple(calls)


class SetupSolver:
    def setup(self, solver_objs, objs, linsolve):
        dm = linsolve.parent_dm
        solver_params = linsolve.solver_parameters

        snescreate = petsc_call('SNESCreate', [objs['comm'], Byref(solver_objs['snes'])])

        set_dm = petsc_call('SNESSetDM', [solver_objs['snes'], dm])

        create_mat = petsc_call('DMCreateMatrix', [dm, Byref(solver_objs['J'])])

        # NOTE: Assumming all solves are linear for now.
        snes_set_type = petsc_call('SNESSetType', [solver_objs['snes'], 'SNESKSPONLY'])

        xglobal = petsc_call('DMCreateGlobalVector', [dm, Byref(solver_objs['x_global'])])

        bglobal = petsc_call('DMCreateGlobalVector', [dm, Byref(solver_objs['b_global'])])

        blocal = self.local_rhs_vecs(dm, solver_objs, linsolve)

        snes_get_ksp = petsc_call(
            'SNESGetKSP', [solver_objs['snes'], Byref(solver_objs['ksp'])]
        )
        ksp_set_tols = petsc_call(
            'KSPSetTolerances', [
                solver_objs['ksp'], solver_params['ksp_rtol'], solver_params['ksp_atol'],
                solver_params['ksp_divtol'], solver_params['ksp_max_it']
            ]
        )
        ksp_set_type = petsc_call(
            'KSPSetType', [solver_objs['ksp'], solver_mapper[solver_params['ksp_type']]]
        )
        ksp_get_pc = petsc_call(
            'KSPGetPC', [solver_objs['ksp'], Byref(solver_objs['pc'])]
        )

        # Even though the default will be jacobi, set to PCNONE for now
        pc_set_type = petsc_call('PCSetType', [solver_objs['pc'], 'PCNONE'])

        ksp_set_from_ops = petsc_call('KSPSetFromOptions', [solver_objs['ksp']])

        return (
            snescreate,
            set_dm,
            create_mat,
            snes_set_type,
            xglobal,
            bglobal,
            blocal,
            snes_get_ksp,
            ksp_set_tols,
            ksp_set_type,
            ksp_get_pc,
            pc_set_type,
            ksp_set_from_ops,
            BlankLine
        )

    def local_rhs_vecs(self, dm, solver_objs, linsolve):
        local_b = petsc_call('DMCreateLocalVector', [
            dm, Byref(solver_objs['b_local_%s' % linsolve.fielddata.target.name])
        ])
        return local_b


class NestedSetupSolver(SetupSolver):
    def local_rhs_vecs(self, dm, solver_objs, linsolve):
        targets = dm.targets
        blocal = [Byref(solver_objs['b_local_%s' % target.name]) for target in targets]
        blocal = petsc_call('DMCompositeGetLocalVectors', [dm] + blocal)
        return blocal


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


def struct_setup(struct_params, objs):
    struct = PETScStruct(name='ctx', pname='MatContext', fields=struct_params)
    struct_callback = generate_struct_callback(struct, objs)

    call_struct_callback = [petsc_call(struct_callback.name, [Byref(struct)])]
    set_appctx = [
        petsc_call('DMSetApplicationContext', [i, Byref(struct)])
        for i in filter_ordered(objs['dmdas'])
    ]
    calls = call_struct_callback + set_appctx
    return struct, tuple(calls), struct_callback


def uxreplace_efuncs(params, efuncs, objs):
    """
    Replace the placeholder struct in efuncs with a newly rebuilt struct
    that includes fields from all specifed parameters.
    """
    old = objs['cbstruct']
    subs = {old: old._rebuild(fields=filter_ordered(params))}
    return {n: Uxreplace(subs).visit(e) for n, e in efuncs.items()}


Null = Macro('NULL')
void = 'void'

# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
