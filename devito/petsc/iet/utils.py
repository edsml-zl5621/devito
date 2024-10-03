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


def spatial_injectsolve_iter(iter, injectsolve):
    spatial_body = []
    for tree in retrieve_iteration_tree(iter[0]):
        root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]
        if injectsolve in FindNodes(InjectSolveDummy).visit(root):
            spatial_body.append(root)
    return spatial_body


# Mapping special Eq operations to their corresponding IET Expression subclass types.
# These operations correspond to subclasses of Eq utilised within PETScSolve.
petsc_iet_mapper = {OpInjectSolve: InjectSolveDummy}


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
        common_dims = [dim for dim in iter.dimensions if dim in struct.fields]
        common_dims = [
            DummyExpr(FieldFromComposite(dim, struct), dim) for dim in common_dims
        ]
        iter_new = iter._rebuild(nodes=List(body=tuple(common_dims)+iter.nodes))
        mapper.update({iter: iter_new})

    return Transformer(mapper).visit(iet)


def retrieve_time_dims(iters):
    outer_iter_dims = iters[0].dimensions

    mapper = {}
    for dim in outer_iter_dims:
        if dim.is_Modulo:
            mapper[dim.origin] = dim
        elif dim.is_Time:
            mapper[dim] = dim
    return mapper
