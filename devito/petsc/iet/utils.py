from devito.ir.iet.nodes import Call
from devito.types import CCompositeObject


def petsc_call(specific_call, call_args):
    general_call = 'PetscCall'
    return Call(general_call, [Call(specific_call, arguments=call_args)])


def petsc_call_mpi(specific_call, call_args):
    general_call = 'PetscCallMPI'
    return Call(general_call, [Call(specific_call, arguments=call_args)])


def petsc_struct(name, fields, liveness='lazy'):
    return CCompositeObject(name=name, pname='MatContext',
                            fields=fields, liveness=liveness)
