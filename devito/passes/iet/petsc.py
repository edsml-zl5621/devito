from ctypes import POINTER
from devito.tools import petsc_type_to_ctype
from devito.types import AbstractObjectWithShape
from sympy import Expr
from devito.ir.iet import Call, Callable, Transformer
from devito.passes.iet.engine import iet_pass

__all__ = ['PetscObject', 'lower_petsc']


class PetscObject(AbstractObjectWithShape, Expr):

    __rkwargs__ = AbstractObjectWithShape.__rkwargs__ + ('petsc_type',)

    def __init_finalize__(self, *args, **kwargs):

        super(PetscObject, self).__init_finalize__(*args, **kwargs)

        self._petsc_type = kwargs.get('petsc_type')

    def _hashable_content(self):
        return super()._hashable_content() + (self.petsc_type,)

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


@iet_pass
def lower_petsc(iet):
    call_back = Callable('call_back', iet.body, 'int', parameters=iet.parameters)
    iet = Transformer({iet.body: Call(call_back.name)}).visit(iet)
    return iet, {'efuncs': [call_back]}
