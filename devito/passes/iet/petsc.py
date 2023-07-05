from ctypes import POINTER
from devito.tools import petsc_type_to_ctype
from devito.types import AbstractObjectWithShape
from sympy import Expr

__all__ = ['PetscObject']


class PetscObject(AbstractObjectWithShape, Expr):

    # need to check this?
    __rkwargs__ = (AbstractObjectWithShape.__rkwargs__ +
                   ('petsc_type',))

    def __init__(self, name, petsc_type, **kwargs):
        self.name = name
        self._petsc_type = petsc_type
        self._is_const = kwargs.get('is_const', False)

    @property
    def _C_ctype(self):
        ctype = petsc_type_to_ctype(self.petsc_type)
        r = type(self.petsc_type, (ctype,), {})
        for n in range(self.ndim):
            r = POINTER(r)
        return r

    @property
    def dtype(self):
        return self._petsc_type

    @property
    def petsc_type(self):
        return self._petsc_type
