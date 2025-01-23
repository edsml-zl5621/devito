import cgen as c
from collections import namedtuple

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (Transformer, MapNodes, Iteration, BlankLine,
                           FindNodes, Uxreplace)
from devito.symbolics import Byref, Macro
from devito.petsc.types import (PetscMPIInt, PetscErrorCode)
from devito.petsc.iet.nodes import InjectSolveDummy
from devito.petsc.utils import core_metadata
from devito.petsc.iet.routines import (CallbackBuilder, ObjectBuilder,
                                       SetupSolver, Solver, TimeDependent,
                                       NonTimeDependent)
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

        builder = Builder(injectsolve, objs, iters, **kwargs)

        setup.extend(builder.solversetup.calls)

        # Transform the spatial iteration loop with the calls to execute the solver
        subs.update(builder.solve.mapper)

        # Use Uxreplace on the efuncs to replace the dummy struct with
        # the actual local struct, now that all the struct parameters
        # for this solve have been determined
        new_efuncs = uxreplace_efuncs(builder.cbbuilder.efuncs, builder.solver_objs)
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


class Builder:
    """
    This class is designed to support future extensions, enabling
    different combinations of solver types, preconditioning methods,
    and other functionalities as needed.

    The class will be extended to accommodate different solver types by
    returning subclasses of the objects initialised in __init__, 
    depending on the properties of `injectsolve`.
    """
    def __init__(self, injectsolve, objs, iters, **kwargs):

        # Determine the time dependency class
        time_mapper = injectsolve.expr.rhs.time_mapper
        timedep = TimeDependent if time_mapper else NonTimeDependent
        self.timedep = timedep(injectsolve, **kwargs)

        # Objects
        self.objbuilder = ObjectBuilder(
            injectsolve, iters, timedep=self.timedep, **kwargs
        )
        self.solver_objs = self.objbuilder.solver_objs

        # Callbacks
        self.cbbuilder = CallbackBuilder(
            injectsolve, objs, self.solver_objs, timedep=self.timedep, **kwargs
        )

        # Solver setup
        self.solversetup = SetupSolver(
            self.solver_objs, objs, injectsolve, self.cbbuilder
        )

        # Execute the solver
        self.solve = Solver(
            self.solver_objs, objs, injectsolve, iters, self.cbbuilder, timedep=self.timedep
            )


def uxreplace_efuncs(efuncs, solver_objs):
    def replace(efunc):
        mapper = {solver_objs['dummyctx']: solver_objs['localctx']}
        return Uxreplace(mapper).visit(efunc)

    return {k: replace(v) for k, v in efuncs.items()}


Null = Macro('NULL')
void = 'void'

# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
