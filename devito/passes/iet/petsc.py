from ctypes import POINTER, c_int
from devito.tools import petsc_type_to_ctype, ctypes_to_cstr, dtype_to_ctype
from devito.types import AbstractObjectWithShape, CompositeObject
from devito.types.basic import Basic, Symbol
from sympy import Expr
from devito.ir.iet import Call, Callable, Transformer, Definition, CallBack, Uxreplace, Conditional, DummyExpr, Block, EntryFunction, List, FindNodes, Iteration
from devito.passes.iet.engine import iet_pass
from devito.symbolics import FunctionPointer, ccode, Byref, FieldFromPointer
import cgen as c
import sympy

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

    symbs_callback = {'retval': PetscObject(name='retval', petsc_type='PetscErrorCode'),
                   'A_matfree': PetscObject(name='A_matfree', petsc_type='Mat'),
                   'xvec': PetscObject(name='xvec', petsc_type='Vec'),
                   'yvec': PetscObject(name='yvec', petsc_type='Vec'),
                   iet.functions[0].name : PetscObject(name=iet.functions[0].name,
                                                            petsc_type='PetscScalar', grid=iet.functions[0].grid),
                   iet.functions[1].name : PetscObject(name=iet.functions[1].name,
                                                            petsc_type='PetscScalar', grid=iet.functions[1].grid)}

    usr_ctx = list(iet.parameters)
    # from IPython import embed; embed()

    # this will be a list comprehension based on no.of dimensions used
    xsize = iet.parameters[0].grid.dimensions[0].symbolic_size
    ysize = iet.parameters[0].grid.dimensions[1].symbolic_size
    usr_ctx.extend([xsize, ysize])
    matctx = PetscStruct(usr_ctx)

    # from IPython import embed; embed()

    kernel_body = Call('PetscCall', [Call('MatShellSetOperation',
                                          arguments=[symbs_callback['A_matfree']])])
    
    # from IPython import embed; embed()

    mymatshellmult_body = [Definition(matctx),
                           Call('PetscCall', [Call('MatShellGetContext', arguments=[symbs_callback['A_matfree'], Byref(matctx.name)])]),
                           Definition(symbs_callback[iet.functions[0].name]),
                           Definition(symbs_callback[iet.functions[1].name]),
                           Call('PetscCall', [Call('VecGetArray2dRead', arguments=[symbs_callback['xvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[1].name)])]),
                           Call('PetscCall', [Call('VecGetArray2d', arguments=[symbs_callback['yvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[0].name)])]),
                           iet.body.body[1],
                           Call('PetscCall', [Call('VecRestoreArray2dRead', arguments=[symbs_callback['xvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[1].name)])]),
                           Call('PetscCall', [Call('VecRestoreArray2d', arguments=[symbs_callback['yvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[0].name)])]),
                           Call('PetscFunctionReturn', arguments=[0])]

    call_back = Callable('MyMatShellMult', mymatshellmult_body, retval=symbs_callback['retval'],
                          parameters=(symbs_callback['A_matfree'], symbs_callback['xvec'], symbs_callback['yvec']))

    # from IPython import embed; embed()
    # transform the very inner loop of call_back function
    # this may not be needed...
    else_body = call_back.body.body[6].body[1].body[0].body[0].nodes[0].nodes[0]
    then_body = DummyExpr(symbs_callback[iet.functions[0].name].indexify(), symbs_callback[iet.functions[1].name].indexify())
    condition = sympy.Or(sympy.Eq(iet.functions[0].dimensions[0], 0), sympy.Eq(iet.functions[0].dimensions[0], iet.parameters[4]),
                         sympy.Eq(iet.functions[0].dimensions[1], 0), sympy.Eq(iet.functions[1].dimensions[0], iet.parameters[6]))
    call_back = Transformer({else_body: Conditional(condition, then_body, else_body)}).visit(call_back)


    tmp = [i for i in usr_ctx if isinstance(i, Symbol)]
    # from IPython import embed; embed()
    for i in tmp:
        call_back = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(call_back)

    call_back_arg = CallBack(call_back.name, 'void', 'void')


    kernel_body = List(body=[Call('PetscCall', [Call('MatShellSetOperation',
                                          arguments=[symbs_callback['A_matfree'], 'MATOP_MULT', call_back_arg])]),
                             Call('PetscFunctionReturn', arguments=[0]),
                             Definition(matctx)])
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
    
