from ctypes import POINTER
from devito.tools import petsc_type_to_ctype
from devito.types import AbstractObjectWithShape

__all__ = ['PetscObject']


class PetscObject(AbstractObjectWithShape):

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
    def petsc_type(self):
        return self._petsc_type
