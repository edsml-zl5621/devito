from devito.ir.iet.nodes import Call
from devito.petsc.iet.nodes import InjectSolveDummy
from devito.ir.equations import OpInjectSolve
from devito.ir.iet import retrieve_iteration_tree, filter_iterations, Transformer, Iteration, FindNodes, List



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


def remove_time_loop(iet):
    spatial_body = []
    for tree in retrieve_iteration_tree(iet):
        root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]
        spatial_body.append(root)
    
    time_loop = FindNodes(Iteration).visit(iet)
    time_loop = [it for it in time_loop if it.dim.is_Time][0]
    iet = Transformer({time_loop: List(body=spatial_body)}).visit(iet)
    return iet


# Mapping special Eq operations to their corresponding IET Expression subclass types.
# These operations correspond to subclasses of Eq utilised within PETScSolve.
petsc_iet_mapper = {OpInjectSolve: InjectSolveDummy}
