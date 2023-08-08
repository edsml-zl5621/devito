from ctypes import POINTER
from devito.tools import petsc_type_to_ctype
from devito.types import AbstractObjectWithShape
from sympy import Expr
from devito.ir.iet import Call, Callable, Transformer, Definition
from devito.passes.iet.engine import iet_pass
from devito.symbolics import FunctionPointer, ccode

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
def lower_petsc(iet, **kwargs):
    # from IPython import embed; embed()

    symbs_petsc = {'retval': PetscObject(name='retval', petsc_type='PetscErrorCode'),
                   'A_matfree': PetscObject(name='A_matfree', petsc_type='Mat'),
                   'xvec': PetscObject(name='xvec', petsc_type='Vec'),
                   'yvec': PetscObject(name='yvec', petsc_type='Vec'),
                   'x' : PetscObject(name='x', petsc_type='Vec')}
    
    call_back = Callable('MyMatShellMult', iet.body.body[1], retval=symbs_petsc['retval'],
                          parameters=(symbs_petsc['A_matfree'], symbs_petsc['xvec'], symbs_petsc['yvec']))
    
    # from IPython import embed; embed()

    tmp = FunctionPointer(call_back.name, 'void', 'void')
    kernel_body = Call('PetscCall', [Call('MatShellSetOperation', arguments=[symbs_petsc['A_matfree'], tmp])])

    # iet = Transformer({iet.body.body[1]: Call(call_back.name)}).visit(iet)
    iet = Transformer({iet.body.body[1]: kernel_body}).visit(iet)


    # add necessary include directories for petsc
    kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/include')
    kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/arch-linux-c-debug/include')
    kwargs['compiler'].add_libraries('petsc')
    libdir = '/home/zl5621/petsc/arch-linux-c-debug/lib'
    kwargs['compiler'].add_library_dirs(libdir)
    kwargs['compiler'].add_ldflags('-Wl,-rpath,%s' % libdir)


    # return iet, {'efuncs': [kernel_body],
    #              'includes': ['petscksp.h']}

    return iet, {'includes': ['petscksp.h', 'stdio.h']}
