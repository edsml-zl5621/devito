from devito.tools import dtype_to_petsctype, CustomDtype
from devito.types import LocalObject
from devito.types.array import ArrayBasic
import numpy as np


class DM(LocalObject):
    """
    PETSc Data Management object (DM).
    """
    dtype = CustomDtype('DM')


class Mat(LocalObject):
    """
    PETSc Matrix object (Mat).
    """
    dtype = CustomDtype('Mat')


class Vec(LocalObject):
    """
    PETSc Vector object (Vec).
    """
    dtype = CustomDtype('Vec')


class PetscMPIInt(LocalObject):
    """
    PETSc datatype used to represent `int` parameters
    to MPI functions.
    """
    dtype = CustomDtype('PetscMPIInt')


class KSP(LocalObject):
    """
    PETSc KSP : Linear Systems Solvers.
    Manages Krylov Methods.
    """
    dtype = CustomDtype('KSP')


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


class AbstractArray(ArrayBasic):

    """
    A customised version of ArrayBasic that allows objects to
    be generated without a cast. This is particularly useful for
    PETSc objects. Also, AbstractArray objects can
    be explicitly specifed as constant.
    """

    _data_alignment = False

    is_AbstractArray = True

    __rkwargs__ = ArrayBasic.__rkwargs__ + ('is_const',)

    def __init_finalize__(self, *args, **kwargs):

        super().__init_finalize__(*args, **kwargs)

        self._is_const = kwargs.get('is_const', False)

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return kwargs.get('dtype', np.float32)

    @property
    def _C_name(self):
        return self.name

    @property
    def is_const(self):
        return self._is_const


class PETScArray(AbstractArray):

    @property
    def _C_ctype(self):
        petsc_type = dtype_to_petsctype(self.dtype)
        modifier = '*' * len(self.dimensions)
        return CustomDtype(petsc_type, modifier=modifier)
