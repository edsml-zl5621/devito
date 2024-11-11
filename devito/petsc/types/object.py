from ctypes import POINTER

from devito.tools import CustomDtype, dtype_to_cstr, as_tuple
from devito.types import (LocalObject, CCompositeObject, ModuloDimension,
                          TimeDimension, ArrayObject, CustomDimension, Array, Symbol)
from devito.types.array import ArrayBasic
from devito.symbolics import Byref

from devito.petsc.iet.utils import petsc_call
from devito.petsc.types import PETScArray

class PETScObject:
    pass

class BasicDM(LocalObject, PETScObject):
    """
    PETSc Data Management object (DM).
    """
    dtype = CustomDtype('DM')

    def __init__(self, *args, target=None, destroy=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = target
        self._destroy = destroy

    @property
    def target(self):
        return self._target
    
    @property
    def destroy(self):
        return self._destroy

    @property
    def stencil_width(self):
        return self.target.space_order

    @property
    def info(self):
        return DMDALocalInfo(name='%s_info' % self.name, liveness='eager')

    @property
    def _C_free(self):
        return

    @property
    def _C_free_priority(self):
        return


class DM(BasicDM):
    @property
    def _C_free(self):
        if not self.destroy:
            return None
        return petsc_call('DMDestroy', [Byref(self.function)])

    @property
    def _C_free_priority(self):
        return 2


class DMComposite(DM):
    def __init__(self, *args, targets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._targets = targets
    
    @property
    def targets(self):
        return self._targets


class Mat(LocalObject, PETScObject):
    """
    PETSc Matrix object (Mat).
    """
    dtype = CustomDtype('Mat')

    @property
    def _C_free(self):
        return petsc_call('MatDestroy', [Byref(self.function)])

    @property
    def _C_free_priority(self):
        return 1


class SubMat(LocalObject, PETScObject):
    """
    SubMatrix of a PETSc Matrix of type MATNEST.
    """
    dtype = CustomDtype('Mat')

    def __init__(self, *args, row=None, col=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._row = row
        self._col = col

    @property
    def row(self):
        return self._row
    
    @property
    def col(self):
        return self._col


class LocalVec(LocalObject, PETScObject):
    """
    PETSc Vector object (Vec).
    """
    dtype = CustomDtype('Vec')


class GlobalVec(LocalObject, PETScObject):
    """
    PETSc Vector object (Vec).
    """
    dtype = CustomDtype('Vec')

    @property
    def _C_free(self):
        return petsc_call('VecDestroy', [Byref(self.function)])

    @property
    def _C_free_priority(self):
        return 0


class PetscMPIInt(LocalObject, PETScObject):
    """
    PETSc datatype used to represent `int` parameters
    to MPI functions.
    """
    dtype = CustomDtype('PetscMPIInt')


class PetscInt(LocalObject, PETScObject):
    """
    PETSc datatype used to represent `int` parameters
    to PETSc functions.
    """
    dtype = CustomDtype('PetscInt')


class KSP(LocalObject, PETScObject):
    """
    PETSc KSP : Linear Systems Solvers.
    Manages Krylov Methods.
    """
    dtype = CustomDtype('KSP')


class SNES(LocalObject, PETScObject):
    """
    PETSc SNES : Non-Linear Systems Solvers.
    """
    dtype = CustomDtype('SNES')

    @property
    def _C_free(self):
        return petsc_call('SNESDestroy', [Byref(self.function)])

    @property
    def _C_free_priority(self):
        return 3


class PC(LocalObject, PETScObject):
    """
    PETSc object that manages all preconditioners (PC).
    """
    dtype = CustomDtype('PC')


class KSPConvergedReason(LocalObject, PETScObject):
    """
    PETSc object - reason a Krylov method was determined
    to have converged or diverged.
    """
    dtype = CustomDtype('KSPConvergedReason')


class DMDALocalInfo(LocalObject, PETScObject):
    """
    PETSc object - C struct containing information
    about the local grid.
    """
    dtype = CustomDtype('DMDALocalInfo')


class PetscErrorCode(LocalObject, PETScObject):
    """
    PETSc datatype used to return PETSc error codes.
    https://petsc.org/release/manualpages/Sys/PetscErrorCode/
    """
    dtype = CustomDtype('PetscErrorCode')


class DummyArg(LocalObject, PETScObject):
    dtype = CustomDtype('void', modifier='*')


class IS(ArrayObject, PETScObject):
    """
    Index set object used for efficient indexing into vectors and matrices.
    https://petsc.org/release/manualpages/IS/IS/
    """
    _data_alignment = False

    def __init_finalize__(self, *args, **kwargs):
        self._nindices = kwargs.pop('nindices', ())
        super().__init_finalize__(*args, **kwargs)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        try:
            return as_tuple(kwargs['dimensions']), as_tuple(kwargs['dimensions'])
        except KeyError:
            nindices = kwargs.get('nindices', ())
            dim = CustomDimension(name='d', symbolic_size=nindices)
            return (dim,), (dim,)

    @property
    def dim(self):
        assert len(self.dimensions) == 1
        return self.dimensions[0]

    @property
    def nindices(self):
        return self._nindices

    @property
    def dtype(self):
        return CustomDtype('IS', modifier=' *')

    @property
    def _C_name(self):
        return self.name

    @property
    def _mem_stack(self):
        return False

    @property
    def _C_free(self):
        destroy_calls = [
            petsc_call('ISDestroy', [Byref(self.indexify().subs({self.dim: i}))])
            for i in range(self._nindices)
        ]
        destroy_calls.append(petsc_call('PetscFree', [self.function]))
        return destroy_calls


# class PETScStruct(CCompositeObject, PETScObject):

#     __rargs__ = ('name', 'pname', 'fields')

#     def __init__(self, name, pname, fields, liveness='lazy'):
#         pfields = [(i._C_name, i._C_ctype) for i in fields]
#         super().__init__(name, pname, pfields, liveness)
#         self._fields = fields

#     @property
#     def fields(self):
#         return self._fields

#     @property
#     def time_dim_fields(self):
#         return [f for f in self.fields
#                 if isinstance(f, (ModuloDimension, TimeDimension))]

#     # @property
#     # def _C_ctype(self):
#     #     return POINTER(self.dtype) if self.liveness == \
#     #         'eager' else self.dtype

#     # @property
#     # def _C_ctype(self):
#     #     return self.dtype._type_

#     # _C_modifier = ' *'


class PETScStruct(CCompositeObject, PETScObject):

    __rargs__ = ('name', 'pname', 'fields')

    def __init__(self, name, pname, fields, liveness='lazy'):
        pfields = [(i._C_name, i._C_ctype) for i in fields]
        super().__init__(name, pname, pfields, liveness)
        self._fields = fields

    @property
    def fields(self):
        return self._fields

    @property
    def time_dim_fields(self):
        return [f for f in self.fields
                if isinstance(f, (ModuloDimension, TimeDimension))]

    _C_modifier = ' *'


class CallbackStruct(PETScStruct):
    @property
    def _C_ctype(self):
        return POINTER(self.dtype)

    _C_modifier = ''

    


class StartPtr(LocalObject, PETScObject):
    def __init__(self, name, dtype):
        super().__init__(name=name)
        self.dtype = CustomDtype(dtype_to_cstr(dtype), modifier=' *')
