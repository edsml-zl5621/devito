from devito.ir.iet.nodes import Call, Expression
from devito.petsc.iet.nodes import InjectSolveDummy
from devito.ir.equations import OpInjectSolve
from devito.ir.iet import (FindNodes, retrieve_iteration_tree,
                           filter_iterations, Transformer)


def petsc_call(specific_call, call_args):
    return Call('PetscCall', [Call(specific_call, arguments=call_args)])


def petsc_call_mpi(specific_call, call_args):
    return Call('PetscCallMPI', [Call(specific_call, arguments=call_args)])


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


# Mapping special Eq operations to their corresponding IET Expression subclass types.
# These operations correspond to subclasses of Eq utilised within PETScSolve.
petsc_iet_mapper = {OpInjectSolve: InjectSolveDummy}


def transform_efuncs(efuncs):
    from devito.petsc.types import CallbackExpr
    new_efuncs = {}

    for key, efunc in efuncs.items():
        nodes = FindNodes(Expression).visit(efunc)
        mapper = {
            expr: expr._rebuild(expr=expr.expr._rebuild(rhs=expr.expr.rhs.expr.args[0]))
            for expr in nodes
            if isinstance(expr.expr.rhs, CallbackExpr)
        }
        new_efuncs[key] = Transformer(mapper).visit(efunc)

    return new_efuncs
