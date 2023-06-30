from ctypes import POINTER
from devito.tools import dtype_to_cstr, dtype_to_ctype
from devito.types import AbstractObjectWithShape

__all__ = ['PetscObject']


class PetscObject(AbstractObjectWithShape):

    def __init__(self, name, dtype, **kwargs):
        self.name = name
        self._dtype = dtype
        self._is_const = kwargs.get('is_const', False)

    @property
    def _C_ctype(self):
        """
        """
        ctypename = 'Petsc%s' % dtype_to_cstr(self.dtype).capitalize()
        ctype = dtype_to_ctype(self.dtype)
        r = type(ctypename, (ctype,), {})
        for n in range(self.ndim):
            r = POINTER(r)
        return r