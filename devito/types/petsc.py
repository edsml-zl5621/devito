from devito.types.basic import AbstractFunction
from devito.tools import dtype_to_petsctype, CustomDtype
import numpy as np
from devito.types import LocalObject


class DM(LocalObject):
    dtype = CustomDtype('DM')


class Mat(LocalObject):
    dtype = CustomDtype('Mat')


class Vec(LocalObject):
    dtype = CustomDtype('Vec')


class PetscMPIInt(LocalObject):
    dtype = CustomDtype('PetscMPIInt')


class KSP(LocalObject):
    dtype = CustomDtype('KSP')


class PC(LocalObject):
    dtype = CustomDtype('PC')


class KSPConvergedReason(LocalObject):
    dtype = CustomDtype('KSPConvergedReason')


class PETScFunction(AbstractFunction):
    """
    PETScFunctions.
    """
    _data_alignment = False

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        dtype = kwargs.get('dtype')
        if dtype is not None:
            return dtype
        elif grid is not None:
            return grid.dtype
        else:
            return np.float32

    @classmethod
    def __indices_setup__(cls, *args, **kwargs):
        grid = kwargs.get('grid')
        dimensions = kwargs.get('dimensions')
        if grid is None:
            if dimensions is None:
                raise TypeError("Need either `grid` or `dimensions`")
        elif dimensions is None:
            dimensions = grid.dimensions

        return tuple(dimensions), tuple(dimensions)

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def _C_ctype(self):
        petsc_type = dtype_to_petsctype(self.dtype)
        modifier = '*' * len(self.dimensions)
        customtype = CustomDtype(petsc_type, modifier=modifier)
        return customtype

    @property
    def _C_name(self):
        return self.name
