from ctypes import POINTER, c_int
from devito.tools import petsc_type_to_ctype, ctypes_to_cstr, dtype_to_ctype, split
from devito.types import AbstractObjectWithShape, CompositeObject
from devito.types.basic import Basic, Symbol, Scalar
from sympy import Expr
from devito.ir.iet import Call, Callable, Transformer, Definition, CallBack, Uxreplace, Conditional, DummyExpr, Block, EntryFunction, List, FindNodes, Iteration, CallableBody
from devito.passes.iet.engine import iet_pass
from devito.ir import retrieve_iteration_tree, Forward, Any
from devito.symbolics import FunctionPointer, ccode, Byref, FieldFromPointer, evalrel
import cgen as c
import sympy

__all__ = ['PetscObject', 'lower_petsc', 'PetscStruct', 'adjust_iter']


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

    # from IPython import embed; embed()

    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        iterations = [i for i in tree]
        for i in iterations:
            mapper[i] = i._rebuild(limits=(i.dim.symbolic_min-1, i.dim.symbolic_max+1, i.dim.symbolic_incr))


    if mapper:
        iet = Transformer(mapper, nested=True).visit(iet)

    mymatshellmult_body = [Definition(matctx),
                           Call('PetscCall', [Call('MatShellGetContext', arguments=[petsc_objs['A_matfree'], Byref(matctx.name)])]),
                           Definition(petsc_objs[iet.functions[0].name]),
                           Definition(petsc_objs[iet.functions[1].name]),
                           Call('PetscCall', [Call('VecGetArray2dRead', arguments=[petsc_objs['xvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[1].name)])]),
                           Call('PetscCall', [Call('VecGetArray2d', arguments=[petsc_objs['yvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[0].name)])]),
                           iet.body.body[0].body[0].body[0].body[0],
                           Call('PetscCall', [Call('VecRestoreArray2dRead', arguments=[petsc_objs['xvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[1].name)])]),
                           Call('PetscCall', [Call('VecRestoreArray2d', arguments=[petsc_objs['yvec'], FieldFromPointer(xsize.name, matctx.name), FieldFromPointer(ysize.name, matctx.name), 0, 0, Byref(iet.functions[0].name)])]),
                           Call('PetscFunctionReturn', arguments=[0])]

    call_back = Callable('MyMatShellMult', mymatshellmult_body, retval=petsc_objs['retval'],
                          parameters=(petsc_objs['A_matfree'], petsc_objs['xvec'], petsc_objs['yvec']))

    # from IPython import embed; embed()
    # transform the very inner loop of call_back function
    # this may not be needed...
    # else_body = call_back.body.body[6].body[1].body[0].body[0].nodes[0].nodes[0]

    # from IPython import embed; embed()
    for tree in retrieve_iteration_tree(call_back):
        else_body = tree[0].nodes[0].nodes[0]

    then_body = DummyExpr(petsc_objs[iet.functions[0].name].indexify(), petsc_objs[iet.functions[1].name].indexify())
    condition = sympy.Or(sympy.Eq(iet.functions[0].dimensions[0], iet.parameters[5]-1), sympy.Eq(iet.functions[0].dimensions[0], iet.parameters[4]+1),
                         sympy.Eq(iet.functions[0].dimensions[1], iet.parameters[7]-1), sympy.Eq(iet.functions[0].dimensions[1], iet.parameters[6]+1))
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
                             Call('PetscCall', [Call('PetscInitialize', arguments=['NULL', 'NULL', 'NULL', 'NULL'])]),
                             Call('PetscCallMPI', [Call('MPI_Comm_size', arguments=['PETSC_COMM_WORLD', Byref(petsc_objs['size'].name)])]),
                             Call('PetscCall', [Call('VecCreate', arguments=['PETSC_COMM_SELF', Byref(petsc_objs['x'].name)])]),
                             Call('PetscCall', [Call('VecSetSizes', arguments=[petsc_objs['x'], 'PETSC_DECIDE', petsc_objs['matsize']])]),
                             Call('PetscCall', [Call('VecSetFromOptions', arguments=[petsc_objs['x']])]),
                             Call('PetscCall', [Call('VecDuplicate', arguments=[petsc_objs['x'], Byref(petsc_objs['b'].name)])]),
                             Call('PetscCall', [Call('MatCreateShell', arguments=['PETSC_COMM_WORLD', 'PETSC_DECIDE', 'PETSC_DECIDE', petsc_objs['matsize'], petsc_objs['matsize'], matctx.name, Byref(petsc_objs['A_matfree'].name)])]),
                             Call('PetscCall', [Call('MatSetFromOptions', arguments=[petsc_objs['A_matfree']])]), 
                             Call('PetscCall', [Call('MatShellSetOperation', arguments=[petsc_objs['A_matfree'], 'MATOP_MULT', call_back_arg])]),
                             Call('PetscCall', [Call('KSPCreate', arguments=['PETSC_COMM_WORLD', Byref(petsc_objs['ksp'].name)])]),
                             Call('PetscCall', [Call('KSPSetInitialGuessNonzero', arguments=[petsc_objs['ksp'], 'PETSC_TRUE'])]),
                             Call('PetscCall', [Call('KSPSetOperators', arguments=[petsc_objs['ksp'], petsc_objs['A_matfree'], petsc_objs['A_matfree']])]),
                             Call('PetscCall', [Call('KSPSetTolerances', arguments=[petsc_objs['ksp'], 1.e-5, 'PETSC_DEFAULT', 'PETSC_DEFAULT', 'PETSC_DEFAULT'])]),
                             Call('PetscCall', [Call('KSPSetFromOptions', arguments=[petsc_objs['ksp']])]),
                             Call('PetscCall', [Call('KSPSolve', arguments=[petsc_objs['ksp'], petsc_objs['b'], petsc_objs['x']])]),
                             Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['x'].name)])]),
                             Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['b'].name)])]),
                             Call('PetscCall', [Call('MatDestroy', arguments=[Byref(petsc_objs['A_matfree'].name)])]),
                             Call('PetscCall', [Call('KSPDestroy', arguments=[Byref(petsc_objs['ksp'].name)])]),
                             Call('PetscCall', [Call('PetscFinalize')])])


    # from IPython import embed; embed()
    iet = Transformer({iet.body.body[0]: kernel_body}).visit(iet)


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




@iet_pass
def adjust_iter(iet, **kwargs):

    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        iterations = [i for i in tree]
        from IPython import embed; embed()
        if not iterations:
            continue

        root = iterations[0]
        if root in mapper:
            continue

        # assert all(i.direction in (Forward, Any) for i in iterations)
        outer, inner = split(iterations, lambda i: i.dim)

        # Get root's `symbolic_max` out of each outer Dimension
        roots_max = {i.dim.root: i.symbolic_max for i in outer}

        # Process inner iterations and adjust their bounds
        for n, i in enumerate(inner):
            # If definitely in-bounds, as ensured by a prior compiler pass, then
            # we can skip this step
            if i.is_Inbound:
                continue

            # The Iteration's maximum is the Min of (a) the `symbolic_max` of current
            # Iteration e.g. `x0_blk0 + x0_blk0_size - 1` and (b) the `symbolic_max`
            # of the current Iteration's root Dimension e.g. `x_M`. The generated
            # maximum will be `Min(x0_blk0 + x0_blk0_size - 1, x_M)

            # In some corner cases an offset may be added (e.g. after CIRE passes)
            # E.g. assume `i.symbolic_max = x0_blk0 + x0_blk0_size + 1` and
            # `i.dim.symbolic_max = x0_blk0 + x0_blk0_size - 1` then the generated
            # maximum will be `Min(x0_blk0 + x0_blk0_size + 1, x_M + 2)`

            root_max = roots_max[i.dim.root] + i.symbolic_max - i.dim.symbolic_max
            iter_max = evalrel(min, [i.symbolic_max, root_max])
            mapper[i] = i._rebuild(limits=(i.symbolic_min, iter_max, i.step))

    if mapper:
        iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {}





class PetscStruct(CompositeObject):

    __rargs__ = ('usr_ctx',)

    def __init__(self, usr_ctx):

        # from IPython import embed; embed()

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

    def _arg_values(self, **kwargs):
        values = super()._arg_values(**kwargs)

        # from IPython import embed; embed()
        # 0.0 needs to be changed to value of h_x etc found in kwargs
        for i in self.fields:
            # if (str(i) == 'h_x' or str(i) == 'h_y'):
            #     setattr(values[self.name]._obj, i, 0.0)
            setattr(values[self.name]._obj, i, kwargs['args'][i])

        return values
    
    # maybe also want to override arg defaults?

    
