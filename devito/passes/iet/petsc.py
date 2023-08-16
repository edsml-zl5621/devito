from ctypes import POINTER, c_int
from devito.tools import petsc_type_to_ctype
from devito.types import AbstractObjectWithShape, CompositeObject
from sympy import Expr
from devito.ir.iet import Call, Callable, Transformer, Definition, CallBack
from devito.passes.iet.engine import iet_pass
from devito.symbolics import FunctionPointer, ccode
import cgen as c

__all__ = ['PetscObject', 'lower_petsc', 'PetscStruct']


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

    @property
    def name(self):
        return self._name


@iet_pass
def lower_petsc(iet, **kwargs):
    # from IPython import embed; embed()

    symbs_petsc = {'retval': PetscObject(name='retval', petsc_type='PetscErrorCode'),
                   'A_matfree': PetscObject(name='A_matfree', petsc_type='Mat'),
                   'xvec': PetscObject(name='xvec', petsc_type='Vec'),
                   'yvec': PetscObject(name='yvec', petsc_type='Vec'),
                   'x' : PetscObject(name='x', petsc_type='Vec')}
    
    tmp = iet.args_frozen['parameters'][2:-1]
    tmp2 = PetscStruct(tmp)
    
    call_back = Callable('MyMatShellMult', iet.body.body[1], retval=symbs_petsc['retval'],
                          parameters=(symbs_petsc['A_matfree'], symbs_petsc['xvec'], symbs_petsc['yvec'], tmp2))
    
    call_back_arg = CallBack(call_back.name, 'void', 'void')

    kernel_body = Call('PetscCall', [Call('MatShellSetOperation',
                                          arguments=[symbs_petsc['A_matfree'], 'MATOP_MULT', call_back_arg])])
    

    # from IPython import embed; embed()

    iet = Transformer({iet.body.body[1]: kernel_body}).visit(iet)

    # from IPython import embed; embed()


    # add necessary include directories for petsc
    kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/include')
    kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/arch-linux-c-debug/include')
    kwargs['compiler'].add_libraries('petsc')
    libdir = '/home/zl5621/petsc/arch-linux-c-debug/lib'
    kwargs['compiler'].add_library_dirs(libdir)
    kwargs['compiler'].add_ldflags('-Wl,-rpath,%s' % libdir)


    return iet, {'efuncs': [call_back],
                 'includes': ['petscksp.h']}

    # return iet, {'includes': ['petscksp.h']}


# class new_iet(c.Generable):

#     def __init__(self, name):
#         self.name = name

#     def generate(self):
#         yield "%s\n %s" % (self.name, self.name)



class PetscStruct(CompositeObject):

    __rargs__ = ('usr_ctx',)

    def __init__(self, usr_ctx):
        # from IPython import embed; embed()
        self._usr_ctx = usr_ctx

        fields = [(str(i), c_int) for i in usr_ctx]
        super(PetscStruct, self).__init__('ctx', 'MatContext', fields)

    @property
    def entries(self):
        return self._entries

    @property
    def usr_ctx(self):
        return self._usr_ctx

    # @cached_property
    # def _C_typedecl(self):
    #     # Overriding for better code readability
    #     #
    #     # Struct neighborhood                 Struct neighborhood
    #     # {                                   {
    #     #   int ll;                             int ll, lc, lr;
    #     #   int lc;                 VS          ...
    #     #   int lr;                             ...
    #     #   ...                                 ...
    #     # }                                   }
    #     #
    #     # With this override, we generate the one on the right
    #     groups = [list(g) for k, g in groupby(self.pfields, key=lambda x: x[0][0])]
    #     groups = [(j[0], i) for i, j in [zip(*g) for g in groups]]
    #     # from IPython import embed; embed()
    #     return Struct(self.pname, [Value(ctypes_to_cstr(i), ', '.join(j))
    #                                for i, j in groups])

    # def _arg_defaults(self):
    #     values = super(MPINeighborhood, self)._arg_defaults()
    #     for name, i in zip(self.fields, self.entries):
    #         setattr(values[self.name]._obj, name, self.neighborhood[i])
    #     return values

    # def _arg_values(self, *args, **kwargs):
    #     grid = kwargs.get('grid', None)
    #     # Update `nb` based on object attached to `grid`
    #     if grid is not None:
    #         return grid.distributor._obj_neighborhood._arg_defaults()
    #     else:
    #         return self._arg_defaults()