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

    petsc_objs = {'retval': PetscObject(name='retval', petsc_type='PetscErrorCode'),
                  'A_matfree': PetscObject(name='A_matfree', petsc_type='Mat'),
                  'xvec': PetscObject(name='xvec', petsc_type='Vec'),
                  'yvec': PetscObject(name='yvec', petsc_type='Vec'),
                  iet.functions[0].name : PetscObject(name=iet.functions[0].name,
                                                      petsc_type='PetscScalar', grid=iet.functions[0].grid),
                  iet.functions[1].name : PetscObject(name=iet.functions[1].name,
                                                      petsc_type='PetscScalar', grid=iet.functions[1].grid),
                  'x': PetscObject(name='x', petsc_type='Vec'),
                  'b': PetscObject(name='b', petsc_type='Vec'),
                  'ksp': PetscObject(name='ksp', petsc_type='KSP'),
                  'size': PetscObject(name='size', petsc_type='PetscMPIInt'),
                  'matsize': PetscObject(name='matsize', petsc_type='PetscInt')}


    # this will be a list comprehension based on no.of dimensions used
    usr_ctx = list(iet.parameters)
    xsize = iet.parameters[0].grid.dimensions[0].symbolic_size
    ysize = iet.parameters[0].grid.dimensions[1].symbolic_size
    dims  =  [xsize, ysize]
    usr_ctx.extend(dims)
    matctx = PetscStruct(usr_ctx)


    mymatshellmult_body = [Definition(matctx),
                           Call('PetscCall', [Call('MatShellGetContext', arguments=[petsc_objs['A_matfree'], Byref(matctx.name)])]),
                           Definition(petsc_objs[iet.functions[0].name]),
                           Definition(petsc_objs[iet.functions[1].name]),
                           Call('PetscCall', [Call('VecGetArray2dRead', arguments=[petsc_objs['xvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[1].name)])]),
                           Call('PetscCall', [Call('VecGetArray2d', arguments=[petsc_objs['yvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[0].name)])]),
                           iet.body.body[1],
                           Call('PetscCall', [Call('VecRestoreArray2dRead', arguments=[petsc_objs['xvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[1].name)])]),
                           Call('PetscCall', [Call('VecRestoreArray2d', arguments=[petsc_objs['yvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[0].name)])]),
                           Call('PetscFunctionReturn', arguments=[0])]

    call_back = Callable('MyMatShellMult', mymatshellmult_body, retval=petsc_objs['retval'],
                          parameters=(petsc_objs['A_matfree'], petsc_objs['xvec'], petsc_objs['yvec']))

    # from IPython import embed; embed()
    # transform the very inner loop of call_back function
    # this may not be needed...
    else_body = call_back.body.body[6].body[1].body[0].body[0].nodes[0].nodes[0]
    then_body = DummyExpr(petsc_objs[iet.functions[0].name].indexify(), petsc_objs[iet.functions[1].name].indexify())
    condition = sympy.Or(sympy.Eq(iet.functions[0].dimensions[0], 0), sympy.Eq(iet.functions[0].dimensions[0], iet.parameters[4]),
                         sympy.Eq(iet.functions[0].dimensions[1], 0), sympy.Eq(iet.functions[1].dimensions[0], iet.parameters[6]))
    call_back = Transformer({else_body: Conditional(condition, then_body, else_body)}).visit(call_back)


    tmp = [i for i in usr_ctx if isinstance(i, Symbol)]
    for i in tmp:
        call_back = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(call_back)

    call_back_arg = CallBack(call_back.name, 'void', 'void')

    kernel_body = List(body=[Definition(petsc_objs['x']),
                             Definition(petsc_objs['b']),
                             Definition(petsc_objs['A_matfree']),
                             Definition(petsc_objs['ksp']),
                             Definition(petsc_objs['size']),
                             Definition(petsc_objs['matsize']),
                             DummyExpr(petsc_objs['matsize'], xsize*ysize),
                             Call('PetscCall', [Call('PetscInitialize', arguments=["temp"])]),
                             Call('PetscCallMPI', [Call('MPI_Comm_size', arguments=['PETSC_COMM_WORLD', Byref(petsc_objs['size'].name)])]),
                             Call('PetscCall', [Call('VecCreate', arguments=['PETSC_COMM_SELF', Byref(petsc_objs['x'].name)])]),
                             Call('PetscCall', [Call('VecSetSizes', arguments=[petsc_objs['x'], 'PETSC_DECIDE', petsc_objs['matsize']])]),
                             Call('PetscCall', [Call('VecSetFromOptions', arguments=[petsc_objs['x']])]),
                             Call('PetscCall', [Call('VecDuplicate', arguments=[petsc_objs['x'], Byref(petsc_objs['b'].name)])]),
                             Call('PetscCall', [Call('MatCreateShell', arguments=['PETSC_COMM_WORLD', 'PETSC_DECIDE', 'PETSC_DECIDE', petsc_objs['matsize'], petsc_objs['matsize'], Byref(matctx.name), Byref(petsc_objs['A_matfree'].name)])]),
                             Call('PetscCall', [Call('MatSetFromOptions', arguments=[petsc_objs['A_matfree']])]), 
                             Call('PetscCall', [Call('MatShellSetOperation', arguments=[petsc_objs['A_matfree'], 'MATOP_MULT', call_back_arg])]),
                             Call('PetscCall', [Call('KSPCreate', arguments=['PETSC_COMM_WORLD', Byref(petsc_objs['ksp'].name)])]),
                             Call('PetscCall', [Call('KSPSetInitialGuessNonzero', arguments=[petsc_objs['ksp'], 'PETSC_TRUE'])]),
                             Call('PetscCall', [Call('KSPSetOperators', arguments=[petsc_objs['ksp'], petsc_objs['A_matfree'], petsc_objs['A_matfree']])]),




#   PetscCall(KSPSetTolerances(ksp, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
#   PetscCall(KSPSetFromOptions(ksp));

#   // Solve system.
#   PetscCall(KSPSolve(ksp, b, x));
#   PetscCall(KSPGetIterationNumber(ksp, &its));
#   PetscCall(PetscPrintf(PETSC_COMM_SELF, "Solution achieved in %d iterations\n", its));


#   PetscCall(VecView(x, PETSC_VIEWER_STDOUT_SELF));

#   PetscCall(VecDestroy(&x));
#   PetscCall(VecDestroy(&b));
#   PetscCall(MatDestroy(&A_matfree));
#   PetscCall(KSPDestroy(&ksp));
#   // need to destroy ctx maybe?
#   PetscCall(PetscFinalize());

#   return 0;





                             Call('PetscCall', [Call('MatShellSetOperation',
                                          arguments=[petsc_objs['A_matfree'], 'MATOP_MULT', call_back_arg])]),
                             Call('PetscFunctionReturn', arguments=[0]),
                             Call('hi', arguments=[xsize])])
    # from IPython import embed; embed()
    iet = Transformer({iet.body.body[1]: kernel_body}).visit(iet)


    for i in dims:
        iet = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(iet)


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
    
