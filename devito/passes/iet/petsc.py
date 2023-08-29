from ctypes import POINTER
from devito.tools import petsc_type_to_ctype, dtype_to_ctype
from devito.types import AbstractObjectWithShape, CompositeObject
from devito.types.basic import Symbol
from sympy import Expr
from devito.ir.iet import Call, Callable, Transformer, Definition, CallBack, Uxreplace, Conditional, DummyExpr, List, FindNodes, Iteration, Expression
from devito.passes.iet.engine import iet_pass
from devito.ir import retrieve_iteration_tree
from devito.symbolics import Byref, FieldFromPointer
import cgen as c
import sympy

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
                  'matsize': PetscObject(name='matsize', petsc_type='PetscInt'),
                  'tmp': PetscObject(name='tmp', petsc_type='PetscScalar', grid=iet.functions[0].grid),
                  'position': PetscObject(name='position', petsc_type='PetscInt')}

    # collect the components for the MatContext struct
    usr_ctx = [i for i in iet.parameters if isinstance(i, Symbol)]
    sizes = [iet.functions[0].grid.dimensions[i].symbolic_size for i in range(len(iet.functions[0].grid.dimensions))]
    usr_ctx.extend(sizes)
    matctx = PetscStruct(usr_ctx)

    # alter the iteration bounds since I turned off the shifting in lower_exprs
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        for i in tree:
            mapper[i] = i._rebuild(limits=(i.dim.symbolic_min-1, i.dim.symbolic_max+1, i.dim.symbolic_incr))
    # nested is True since there is more than one mapping inside mapper
    iet = Transformer(mapper, nested=True).visit(iet)

    # retrieve the adjusted iteration loop
    iteration_root = retrieve_iteration_tree(iet)[0][0]

    # build call_back function body 
    mymatshellmult_body = List(body=[c.Line('PetscFunctionBegin;'),
                                     Definition(matctx),
                                     Call('PetscCall', [Call('MatShellGetContext', arguments=[petsc_objs['A_matfree'], Byref(matctx.name)])]),
                                     Definition(petsc_objs[iet.functions[0].name]),
                                     Definition(petsc_objs[iet.functions[1].name]),
                                     Call('PetscCall', [Call('VecGetArray2dRead', arguments=[petsc_objs['xvec'], FieldFromPointer(sizes[0].name, matctx.name), FieldFromPointer(sizes[1].name, matctx.name), 0, 0, Byref(iet.functions[1].name)])]),
                                     Call('PetscCall', [Call('VecGetArray2d', arguments=[petsc_objs['yvec'], FieldFromPointer(sizes[0].name, matctx.name), FieldFromPointer(sizes[1].name, matctx.name), 0, 0, Byref(iet.functions[0].name)])]),
                                     iteration_root,
                                     Call('PetscCall', [Call('VecRestoreArray2dRead', arguments=[petsc_objs['xvec'], FieldFromPointer(sizes[0].name, matctx.name), FieldFromPointer(sizes[1].name, matctx.name), 0, 0, Byref(iet.functions[1].name)])]),
                                     Call('PetscCall', [Call('VecRestoreArray2d', arguments=[petsc_objs['yvec'], FieldFromPointer(sizes[0].name, matctx.name), FieldFromPointer(sizes[1].name, matctx.name), 0, 0, Byref(iet.functions[0].name)])]),
                                     Call('PetscFunctionReturn', arguments=[0])])

    call_back = Callable('MyMatShellMult', mymatshellmult_body, retval=petsc_objs['retval'],
                          parameters=(petsc_objs['A_matfree'], petsc_objs['xvec'], petsc_objs['yvec']))
    
    iteration_root2 = retrieve_iteration_tree(call_back)[0][0]

    # transform the very inner loop of call_back function to have the if, else section
    # this may not ultimately be needed...but for current setup it is needed.
    else_body = FindNodes(Expression).visit(iteration_root)[0]
    then_body = DummyExpr(petsc_objs[iet.functions[0].name].indexify(), petsc_objs[iet.functions[1].name].indexify())
    condition = sympy.Or(sympy.Eq(iet.functions[0].dimensions[0], iet.parameters[5]-1), sympy.Eq(iet.functions[0].dimensions[0], iet.parameters[4]+1),
                         sympy.Eq(iet.functions[0].dimensions[1], iet.parameters[7]-1), sympy.Eq(iet.functions[0].dimensions[1], iet.parameters[6]+1))
    call_back = Transformer({else_body: Conditional(condition, then_body, else_body)}).visit(call_back)

    # from IPython import embed; embed()
    # replace all of the struct MatContext symbols that appear in the callback function with a fieldfrompointer i.e ctx->symbol
    for i in usr_ctx:
        call_back = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(call_back)
    call_back_arg = CallBack(call_back.name, 'void', 'void')


    # to map petsc vector back to devito function
    petsc_2_dev = DummyExpr(petsc_objs[iet.functions[0].name].indexify(indices=(petsc_objs['tmp'].dimensions[0]+iet.functions[0].space_order, petsc_objs['tmp'].dimensions[1]+iet.functions[0].space_order)),
                            petsc_objs['tmp'].indexify(indices=(petsc_objs['tmp'].dimensions[0], petsc_objs['tmp'].dimensions[1])))
    petsc_2_dev = Transformer({else_body: petsc_2_dev}).visit(iteration_root2)
    for i in usr_ctx:
        petsc_2_dev = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(petsc_2_dev)

    
    # to map devito array to petsc vector
    dev_2_petsc = List(body=[DummyExpr(petsc_objs['position'], (iet.functions[0].dimensions[0]-iet.functions[0].space_order)*sizes[1] + (iet.functions[0].dimensions[1]-iet.functions[0].space_order), init=True),
                             Call('PetscCall', [Call('VecSetValue', arguments=[petsc_objs['b'], petsc_objs['position'], petsc_objs[iet.functions[0].name].indexify(), 'INSERT_VALUES'])])])
    
    dev_2_petsc_iters = [lambda ex: Iteration(ex, iet.functions[0].dimensions[0], (iet.functions[0].space_order, iet.functions[0].space_order+sizes[0]-1, 1)),
                         lambda ex: Iteration(ex, iet.functions[0].dimensions[1], (iet.functions[0].space_order, iet.functions[0].space_order+sizes[1]-1, 1))]
    
    dev_2_petsc = dev_2_petsc_iters[0](dev_2_petsc_iters[1](dev_2_petsc))
    

    # from IPython import embed; embed()
    kernel_body = List(body=[Definition(petsc_objs['x']),
                             Definition(petsc_objs['b']),
                             Definition(petsc_objs['A_matfree']),
                             Definition(petsc_objs['ksp']),
                             Definition(petsc_objs['size']),
                             Definition(petsc_objs['matsize']),
                             DummyExpr(petsc_objs['matsize'], sizes[0]*sizes[1]),
                             c.Line('PetscFunctionBeginUser;'),
                             Call('PetscCall', [Call('PetscInitialize', arguments=['NULL', 'NULL', 'NULL', 'NULL'])]),
                             Call('PetscCallMPI', [Call('MPI_Comm_size', arguments=['PETSC_COMM_WORLD', Byref(petsc_objs['size'].name)])]),
                             Call('PetscCall', [Call('VecCreate', arguments=['PETSC_COMM_SELF', Byref(petsc_objs['x'].name)])]),
                             Call('PetscCall', [Call('VecSetSizes', arguments=[petsc_objs['x'], 'PETSC_DECIDE', petsc_objs['matsize']])]),
                             Call('PetscCall', [Call('VecSetFromOptions', arguments=[petsc_objs['x']])]),
                             Call('PetscCall', [Call('VecDuplicate', arguments=[petsc_objs['x'], Byref(petsc_objs['b'].name)])]),
                             dev_2_petsc,
                             Call('PetscCall', [Call('MatCreateShell', arguments=['PETSC_COMM_WORLD', 'PETSC_DECIDE', 'PETSC_DECIDE', petsc_objs['matsize'], petsc_objs['matsize'], matctx.name, Byref(petsc_objs['A_matfree'].name)])]),
                             Call('PetscCall', [Call('MatSetFromOptions', arguments=[petsc_objs['A_matfree']])]), 
                             Call('PetscCall', [Call('MatShellSetOperation', arguments=[petsc_objs['A_matfree'], 'MATOP_MULT', call_back_arg])]),
                             Call('PetscCall', [Call('KSPCreate', arguments=['PETSC_COMM_WORLD', Byref(petsc_objs['ksp'].name)])]),
                             Call('PetscCall', [Call('KSPSetInitialGuessNonzero', arguments=[petsc_objs['ksp'], 'PETSC_TRUE'])]),
                            #  you can then initialise x with initial guess
                             Call('PetscCall', [Call('KSPSetOperators', arguments=[petsc_objs['ksp'], petsc_objs['A_matfree'], petsc_objs['A_matfree']])]),
                             Call('PetscCall', [Call('KSPSetTolerances', arguments=[petsc_objs['ksp'], 1.e-5, 'PETSC_DEFAULT', 'PETSC_DEFAULT', 'PETSC_DEFAULT'])]),
                            #  means that you can do ./exe -ksp_type fgmres etc -> probably delete since we would use KSPSetType(ksp, KSPCG);?
                             Call('PetscCall', [Call('KSPSetFromOptions', arguments=[petsc_objs['ksp']])]),
                             Call('PetscCall', [Call('KSPSolve', arguments=[petsc_objs['ksp'], petsc_objs['b'], petsc_objs['x']])]),
                             Definition(petsc_objs['tmp']),
                             Call('PetscCall', [Call('VecGetArray2d', arguments=[petsc_objs['x'], FieldFromPointer(sizes[0].name, matctx.name), FieldFromPointer(sizes[1].name, matctx.name), 0, 0, Byref(petsc_objs['tmp'].name)])]),
                             petsc_2_dev,
                             Call('PetscCall', [Call('VecRestoreArray2d', arguments=[petsc_objs['x'], FieldFromPointer(sizes[0].name, matctx.name), FieldFromPointer(sizes[1].name, matctx.name), 0, 0, Byref(petsc_objs['tmp'].name)])]),
                             Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['x'].name)])]),
                             Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['b'].name)])]),
                             Call('PetscCall', [Call('MatDestroy', arguments=[Byref(petsc_objs['A_matfree'].name)])]),
                             Call('PetscCall', [Call('KSPDestroy', arguments=[Byref(petsc_objs['ksp'].name)])]),
                             Call('PetscCall', [Call('PetscFinalize')])])


    iet = Transformer({iet.body.body[0]: kernel_body}).visit(iet)

    for i in sizes:
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



class PetscStruct(CompositeObject):

    __rargs__ = ('usr_ctx',)

    def __init__(self, usr_ctx):

        self._usr_ctx = usr_ctx

        pfields = [(i._C_name, dtype_to_ctype(i.dtype)) for i in self.usr_ctx if isinstance(i, Symbol)]

        super(PetscStruct, self).__init__('ctx', 'MatContext', pfields)
    
    @property
    def usr_ctx(self):
        return self._usr_ctx
    
    # maybe also want to def _arg_defaults here?

    def _arg_values(self, **kwargs):
        values = super()._arg_values(**kwargs)
        for i in self.fields:
            setattr(values[self.name]._obj, i, kwargs['args'][i])

        return values

    
