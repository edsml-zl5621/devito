import cgen as c

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (Transformer, MapNodes, Iteration, List, BlankLine,
                           DummyExpr, FindNodes, retrieve_iteration_tree,
                           filter_iterations, Uxreplace)
from devito.symbolics import Byref, Macro, FieldFromComposite, FieldFromPointer, cast_mapper
from devito.symbolics.unevaluation import Mul
from devito.types import Symbol
from devito.petsc.types import (PetscMPIInt, DM, Mat, LocalVec, GlobalVec,
                                KSP, PC, SNES, PetscErrorCode, DummyArg, PetscInt,
                                StartPtr)
from devito.petsc.iet.nodes import (InjectSolveDummy, PETScCall, FormFunctionCallback,
                                    MatVecCallback)
from devito.petsc.utils import solver_mapper, core_metadata
from devito.petsc.iet.routines import CallbackBuilder, ObjectBuilder, SetupSolver, RunSolver
from devito.petsc.iet.utils import petsc_call, petsc_call_mpi


@iet_pass
def lower_petsc(iet, **kwargs):
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
    efuncs = {}

    for iters, (injectsolve,) in injectsolve_mapper.items():
        # Provides flexibility to use various solvers with different combinations
        # of callbacks and configurations
        ObjBuilder, CCBuilder, SolverSetup, SolverRun = get_builder_classes(injectsolve)

        solver_objs = ObjBuilder(**kwargs).build(injectsolve, iters)
        cbbuilder = CCBuilder(**kwargs)

        # Generate all PETSc callback functions for the target via recursive compilation
        matvec_callback, formfunc_callback, formrhs_callback = cbbuilder.make_core(
            injectsolve, objs, solver_objs
        )

        solver_objs['localctx'] = cbbuilder.local_struct(solver_objs)
        solver_objs['mainctx'] = cbbuilder.main_struct(solver_objs)

        struct_callback = cbbuilder.make_struct_callback(solver_objs, objs)

        # Generate the solver setup for each InjectSolveDummy
        solver_setup = SolverSetup().setup(solver_objs, objs, injectsolve, cbbuilder)
        setup.extend(solver_setup)

        # Only Transform the spatial iteration loop
        space_iter, = spatial_loop_nest(iters, injectsolve)
        runsolve = SolverRun().run(
            solver_objs, objs, injectsolve, iters, cbbuilder
        )
        subs.update({space_iter: runsolve})

        # Uxreplace the efuncs to replace the dummy struct with the actual local struct
        # since now all of the struct params have been determined
        new_efuncs = uxreplace_efuncs(cbbuilder.efuncs, solver_objs)
        efuncs.update(new_efuncs)

    iet = Transformer(subs).visit(iet)

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

    return {
        'size': PetscMPIInt(name='size'),
        'comm': communicator,
        'err': PetscErrorCode(name='err'),
        'grid': target.grid
    }


# def create_dmda_objs(unique_targets):
#     unique_dmdas = {}
#     for target in unique_targets:
#         name = 'da_so_%s' % target.space_order
#         unique_dmdas[name] = DM(name=name, liveness='eager',
#                                 stencil_width=target.space_order)
#     return unique_dmdas


def spatial_loop_nest(iter, injectsolve):
    spatial_body = []
    for tree in retrieve_iteration_tree(iter[0]):
        root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]
        if injectsolve in FindNodes(InjectSolveDummy).visit(root):
            spatial_body.append(root)
    return spatial_body


def get_builder_classes(injectsolve):
    """
    Selects the appropriate classes to build/run this solve.
    This function is designed to support future extensions, enabling
    different combinations of solver types, preconditioning methods,
    and other functionalities as needed.
    """
    # NOTE: This function will extend to support different solver types
    # returning subclasses of the classes listed below,
    # based on properties of `injectsolve`
    return ObjectBuilder, CallbackBuilder, SetupSolver, RunSolver


def uxreplace_efuncs(efuncs, solver_objs):
    efuncs_new = {}
    for key, efunc in efuncs.items():
        updated = Uxreplace({solver_objs['dummyctx']: solver_objs['localctx']}).visit(
            efunc
        )
        efuncs_new[key] = updated
    return efuncs_new


Null = Macro('NULL')
void = 'void'

# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
