from ctypes import POINTER
from devito.tools import petsc_name_to_ctype
from devito.types import AbstractObjectWithShape

__all__ = ['PetscObject']


class PetscObject(AbstractObjectWithShape):

    def __init__(self, name, petsc_name, **kwargs):
        self.name = name
        self._petsc_name = petsc_name
        self._is_const = kwargs.get('is_const', False)

    @property
    def _C_ctype(self):
        ctype = petsc_name_to_ctype(self.petsc_name)
        r = type(self.petsc_name, (ctype,), {})
        for n in range(self.ndim):
            r = POINTER(r)
        return r
    
    @property
    def petsc_name(self):
        return self._petsc_name