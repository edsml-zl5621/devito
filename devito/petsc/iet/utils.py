from devito.ir.iet.nodes import Expression
from devito.petsc.iet.nodes import InjectSolveDummy, PETScCall
from devito.ir.equations import OpInjectSolve
from devito.ir.iet import (FindNodes, retrieve_iteration_tree,
                           filter_iterations, Transformer, Iteration,
                           DummyExpr, List)
from devito.symbolics import FieldFromComposite


def petsc_call(specific_call, call_args):
    return PETScCall('PetscCall', [PETScCall(specific_call, arguments=call_args)])


def petsc_call_mpi(specific_call, call_args):
    return PETScCall('PetscCallMPI', [PETScCall(specific_call, arguments=call_args)])


def petsc_struct(name, fields, liveness='lazy'):
    # TODO: Fix this circular import
    from devito.petsc.types.object import PETScStruct
    return PETScStruct(name=name, pname='MatContext',
                       fields=fields, liveness=liveness)


def spatial_iteration_loops(iet):
    spatial_body = []
    for tree in retrieve_iteration_tree(iet):
        root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]
        spatial_body.append(root)
    return spatial_body


# Mapping special Eq operations to their corresponding IET Expression subclass
# types. These operations correspond to subclasses of Eq utilised within PETScSolve
petsc_iet_mapper = {OpInjectSolve: InjectSolveDummy}


def remove_CallbackExpr(body):
    from devito.petsc.types import CallbackExpr
    nodes = FindNodes(Expression).visit(body)
    mapper = {
        expr: expr._rebuild(expr=expr.expr._rebuild(rhs=expr.expr.rhs.args[0]))
        for expr in nodes
        if isinstance(expr.expr.rhs, CallbackExpr)
    }
    return Transformer(mapper).visit(body)


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

    dimension_mapper = {}
    for iter in time_iters:
        common_dimensions = [dim for dim in iter.dimensions if dim in struct.fields]
        common_dimensions = [DummyExpr(FieldFromComposite(dim, struct), dim)
                             for dim in common_dimensions]
        iter_new = iter._rebuild(nodes=List(body=tuple(common_dimensions)+iter.nodes))
        dimension_mapper.update({iter: iter_new})

    return Transformer(dimension_mapper).visit(iet)
