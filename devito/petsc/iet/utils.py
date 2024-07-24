from devito.ir.iet.nodes import Call


def petsc_call(specific_call, call_args):
    general_call = 'PetscCall'
    return Call(general_call, [Call(specific_call, arguments=call_args)])


def petsc_call_mpi(specific_call, call_args):
    general_call = 'PetscCallMPI'
    return Call(general_call, [Call(specific_call, arguments=call_args)])