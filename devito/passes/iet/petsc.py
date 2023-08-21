from ctypes import POINTER, c_int
from devito.tools import petsc_type_to_ctype, ctypes_to_cstr, dtype_to_ctype
from devito.types import AbstractObjectWithShape, CompositeObject
from devito.types.basic import Basic, Symbol
from sympy import Expr
from devito.ir.iet import Call, Callable, Transformer, Definition, CallBack, Uxreplace
from devito.passes.iet.engine import iet_pass
from devito.symbolics import FunctionPointer, ccode, Byref, FieldFromPointer
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
                   'x' : PetscObject(name='x', petsc_type='Vec'),
                   iet.functions[0].name : PetscObject(name=iet.functions[0].name,
                                                            petsc_type='PetscScalar', grid=iet.functions[0].grid),
                   iet.functions[1].name : PetscObject(name=iet.functions[1].name,
                                                            petsc_type='PetscScalar', grid=iet.functions[1].grid)}

    usr_ctx = list(iet.parameters)
    # from IPython import embed; embed()
    xsize = iet.functions[0].dimensions[0].symbolic_size
    ysize = iet.functions[0].dimensions[1].symbolic_size
    usr_ctx.extend([xsize, ysize])
    matctx = PetscStruct(usr_ctx)


    # from IPython import embed; embed()
    mymatshellmult_body = [Definition(matctx),
                           Call('PetscCall', [Call('MatShellGetContext', arguments=[symbs_petsc['A_matfree'], Byref(matctx.name)])]),
                           Definition(symbs_petsc[iet.functions[0].name]),
                           Definition(symbs_petsc[iet.functions[1].name]),
                           Call('PetscCall', [Call('VecGetArray2dRead', arguments=[symbs_petsc['xvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[1].name)])]),
                           Call('PetscCall', [Call('VecGetArray2d', arguments=[symbs_petsc['yvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[0].name)])]),
                           iet.body.body[1],
                           Call('PetscCall', [Call('VecRestoreArray2dRead', arguments=[symbs_petsc['xvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[1].name)])]),
                           Call('PetscCall', [Call('VecRestoreArray2d', arguments=[symbs_petsc['yvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[0].name)])]),
                           Call('PetscFunctionReturn', arguments=[0])]

    call_back = Callable('MyMatShellMult', mymatshellmult_body, retval=symbs_petsc['retval'],
                          parameters=(symbs_petsc['A_matfree'], symbs_petsc['xvec'], symbs_petsc['yvec']))
    
    from IPython import embed; embed()
    tmp = [i for i in usr_ctx if isinstance(i, Symbol)]
    # from IPython import embed; embed()
    for i in tmp:
        call_back = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(call_back)

    call_back_arg = CallBack(call_back.name, 'void', 'void')

    kernel_body = Call('PetscCall', [Call('MatShellSetOperation',
                                          arguments=[symbs_petsc['A_matfree'], 'MATOP_MULT', call_back_arg])])
    

    # from IPython import embed; embed()

    iet = Transformer({iet.body.body[1]: kernel_body}).visit(iet)



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

        self._usr_ctx = usr_ctx

        pfields = [(i._C_name, dtype_to_ctype(i.dtype)) for i in self.usr_ctx if isinstance(i, Symbol)]

        super(PetscStruct, self).__init__('ctx', 'MatContext', pfields)
    
    @property
    def usr_ctx(self):
        return self._usr_ctx
    
    # can use this if I want to generate typedefs
    # @property
    # def _C_typedecl(self):
        # struct = [c.Value(ctypes_to_cstr(ctype), param) for (param, ctype) in self.pfields]
    #     struct = c.Struct(self.pname, struct)
    #     # return c.Typedef(struct)
    #     return struct
    
