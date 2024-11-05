from ctypes import POINTER

from devito.tools import CustomDtype, dtype_to_cstr, as_tuple
from devito.types import (LocalObject, CCompositeObject, ModuloDimension,
                          TimeDimension, ArrayObject, CustomDimension)
from devito.symbolics import Byref

from devito.petsc.iet.utils import petsc_call


class BasicDM(LocalObject):
    """
    PETSc Data Management object (DM).
    """
    dtype = CustomDtype('DM')

    def __init__(self, *args, target=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = target

    @property
    def target(self):
        return self._target

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


class CallbackDM(BasicDM):
    pass


class Mat(LocalObject):
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


class SubMat(LocalObject):
    """
    SubMatrix of a PETSc Matrix of type MATNEST.
    """
    dtype = CustomDtype('Mat')


class LocalVec(LocalObject):
    """
    PETSc Vector object (Vec).
    """
    dtype = CustomDtype('Vec')


class GlobalVec(LocalObject):
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


class PetscMPIInt(LocalObject):
    """
    PETSc datatype used to represent `int` parameters
    to MPI functions.
    """
    dtype = CustomDtype('PetscMPIInt')


class PetscInt(LocalObject):
    """
    PETSc datatype used to represent `int` parameters
    to PETSc functions.
    """
    dtype = CustomDtype('PetscInt')


class KSP(LocalObject):
    """
    PETSc KSP : Linear Systems Solvers.
    Manages Krylov Methods.
    """
    dtype = CustomDtype('KSP')


class SNES(LocalObject):
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


class PC(LocalObject):
    """
    PETSc object that manages all preconditioners (PC).
    """
    dtype = CustomDtype('PC')


class KSPConvergedReason(LocalObject):
    """
    PETSc object - reason a Krylov method was determined
    to have converged or diverged.
    """
    dtype = CustomDtype('KSPConvergedReason')


class DMDALocalInfo(LocalObject):
    """
    PETSc object - C struct containing information
    about the local grid.
    """
    dtype = CustomDtype('DMDALocalInfo')


class PetscErrorCode(LocalObject):
    """
    PETSc datatype used to return PETSc error codes.
    https://petsc.org/release/manualpages/Sys/PetscErrorCode/
    """
    dtype = CustomDtype('PetscErrorCode')


class DummyArg(LocalObject):
    dtype = CustomDtype('void', modifier='*')


class IS(ArrayObject):
    """
    Index set object used for efficient indexing into vector and matrices.
    """
    _data_alignment = False

    @classmethod
    def __indices_setup__(cls, **kwargs):
        try:
            return as_tuple(kwargs['dimensions']), as_tuple(kwargs['dimensions'])
        except KeyError:
            nindices = kwargs['nindices']
            dim = CustomDimension(name='d', symbolic_size=nindices)
            return (dim,), (dim,)

    @property
    def dim(self):
        assert len(self.dimensions) == 1
        return self.dimensions[0]

    @property
    def nindices(self):
        return self.dim.symbolic_size

    @property
    def index(self):
        if self.size == 1:
            return 0
        else:
            return self.dim


class PETScStruct(CCompositeObject):

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

    @property
    def _C_ctype(self):
        return POINTER(self.dtype) if self.liveness == \
            'eager' else self.dtype

    _C_modifier = ' *'


class StartPtr(LocalObject):
    def __init__(self, name, dtype):
        super().__init__(name=name)
        self.dtype = CustomDtype(dtype_to_cstr(dtype), modifier=' *')
