import cgen as c

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import Transformer, MapNodes, Iteration, BlankLine, DummyExpr, CallableBody, List, Call, Callable
from devito.symbolics import Byref, Macro, FieldFromPointer
from devito.petsc.types import (PetscMPIInt, PetscErrorCode, FieldData, MultipleFieldData,
                                SubDM, IS, PETScStruct)
from devito.petsc.iet.nodes import InjectSolveDummy
from devito.petsc.utils import core_metadata
from devito.petsc.iet.routines import (CBBuilder, CCBBuilder, BaseObjectBuilder, CoupledObjectBuilder,
                                       BaseSetup, CoupledSetup, Solver, CoupledSolver,
                                       TimeDependent, NonTimeDependent)
from devito.petsc.iet.utils import petsc_call, petsc_call_mpi


@iet_pass
def lower_petsc(iet, **kwargs):
    # Check if PETScSolve was used
    injectsolve_mapper = MapNodes(Iteration, InjectSolveDummy,
                                  'groupby').visit(iet)

    if not injectsolve_mapper:
        return iet, {}

    unique_grids = {i.expr.rhs.grid for (i,) in injectsolve_mapper.values()}
    # Assumption is that all solves are on the same grid
    if len(unique_grids) > 1:
        raise ValueError("All PETScSolves must use the same Grid, but multiple found.")
    grid = unique_grids.pop()
    objs = build_core_objects(grid, **kwargs)

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

        efuncs.update(builder.cbbuilder.efuncs)

    # TODO: Only add this efunc if actually required
    populate_matrix_context(efuncs, objs)

    iet = Transformer(subs).visit(iet)
    
    init = init_petsc(**kwargs)
    body = core + tuple(setup) + (BlankLine,) + iet.body.body
    body = iet.body._rebuild(
        init=init, body=body,
        frees=(petsc_call('PetscFinalize', []),)
    )
    iet = iet._rebuild(body=body)
    metadata = core_metadata()
    metadata.update({'efuncs': tuple(efuncs.values())})
    return iet, metadata


def init_petsc(**kwargs):
    # Initialize PETSc -> for now, assuming all solver options have to be
    # specified via the parameters dict in PETScSolve
    # TODO: Are users going to be able to use PETSc command line arguments?
    # In firedrake, they have an options_prefix for each solver, enabling the use
    # of command line options
    initialize = petsc_call('PetscInitialize', [Null, Null, Null, Null])

    return petsc_func_begin_user, initialize


def make_core_petsc_calls(objs, **kwargs):
    call_mpi = petsc_call_mpi('MPI_Comm_size', [objs['comm'], Byref(objs['size'])])

    return call_mpi, BlankLine


def build_core_objects(grid, **kwargs):
    if kwargs['options']['mpi']:
        communicator = grid.distributor._obj_comm
    else:
        communicator = 'PETSC_COMM_SELF'

    return {
        'size': PetscMPIInt(name='size'),
        'comm': communicator,
        'err': PetscErrorCode(name='err'),
        'grid': grid
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
        self.timedep = timedep(injectsolve, iters, **kwargs)

        # TODO: obvs improve this
        if isinstance(injectsolve.expr.rhs.fielddata, MultipleFieldData):
            coupled = True
        else:
            coupled = False


        # Objects
        if coupled:
            self.objbuilder = CoupledObjectBuilder(injectsolve, **kwargs)
        else:
            self.objbuilder = BaseObjectBuilder(injectsolve, **kwargs)
        self.solver_objs = self.objbuilder.solver_objs

        # Callbacks
        if coupled:
            self.cbbuilder = CCBBuilder(
                injectsolve, objs, self.solver_objs, timedep=self.timedep,
                **kwargs
            )
        else:
            self.cbbuilder = CBBuilder(
                injectsolve, objs, self.solver_objs, timedep=self.timedep,
                **kwargs
            )

        if coupled:
            # Solver setup
            self.solversetup = CoupledSetup(
                self.solver_objs, objs, injectsolve, self.cbbuilder
            )
        else:
            self.solversetup = BaseSetup(
                self.solver_objs, objs, injectsolve, self.cbbuilder
            )

        # NOTE: might not acc need a separate coupled class for this->rethink
        # just addding one for the purposes of debugging and figuring out the coupled abstraction
        if coupled:
            # Execute the solver
            self.solve = CoupledSolver(
                self.solver_objs, objs, injectsolve, iters,
                self.cbbuilder, timedep=self.timedep
            )
        else:
            # Execute the solver
            self.solve = Solver(
                self.solver_objs, objs, injectsolve, iters,
                self.cbbuilder, timedep=self.timedep
            )


def populate_matrix_context(efuncs, objs):
    subdms_expr = DummyExpr(
        FieldFromPointer(Subdms._C_symbol, jctx), Subdms._C_symbol
    )
    fields_expr = DummyExpr(
        FieldFromPointer(Fields._C_symbol, jctx), Fields._C_symbol
    )
    body = CallableBody(
        List(body=[subdms_expr, fields_expr]),
        init=(c.Line('PetscFunctionBeginUser;'),),
        retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])])
    )
    cb = Callable('PopulateMatContext',
        body, objs['err'],
        parameters=[jctx, Subdms, Fields]
    )
    # todo: only want to add this if a coupled solver is used
    efuncs['PopulateMatContext'] = cb



Null = Macro('NULL')
void = 'void'


# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')


# JacMatrixCtx struct members
Subdms = SubDM(name='subdms', nindices=1)
Fields = IS(name='fields', nindices=1)

jctx = PETScStruct(
    name='jctx', pname='JacobianCtx',
    fields=[Subdms, Fields], liveness='lazy'
)

