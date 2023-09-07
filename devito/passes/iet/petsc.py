from ctypes import POINTER
from devito.tools import petsc_type_to_ctype, dtype_to_ctype
from devito.types import AbstractObjectWithShape, CompositeObject, Dimension
from devito.types.basic import Symbol
from sympy import Expr
from devito.ir.iet import Call, Callable, Transformer, Definition, CallBack, Uxreplace, Conditional, DummyExpr, List, FindNodes, Iteration, Expression
from devito.passes.iet.engine import iet_pass
from devito.ir import retrieve_iteration_tree
from devito.symbolics import Byref, FieldFromPointer, IndexedPointer
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


# @iet_pass
# def lower_petsc(iet, **kwargs):
#     # from IPython import embed; embed()

#     petsc_objs = {'retval': PetscObject(name='retval', petsc_type='PetscErrorCode'),
#                   'A_matfree': PetscObject(name='A_matfree', petsc_type='Mat'),
#                   'xvec': PetscObject(name='xvec', petsc_type='Vec'),
#                   'yvec': PetscObject(name='yvec', petsc_type='Vec'),
#                   iet.functions[0].name : PetscObject(name=iet.functions[0].name,
#                                                       petsc_type='PetscScalar', grid=iet.functions[0].grid),
#                   iet.functions[1].name : PetscObject(name=iet.functions[1].name,
#                                                       petsc_type='PetscScalar', grid=iet.functions[1].grid),
#                   'x': PetscObject(name='x', petsc_type='Vec'),
#                   'b': PetscObject(name='b', petsc_type='Vec'),
#                   'ksp': PetscObject(name='ksp', petsc_type='KSP'),
#                   'size': PetscObject(name='size', petsc_type='PetscMPIInt'),
#                   'matsize': PetscObject(name='matsize', petsc_type='PetscInt'),
#                   'tmp': PetscObject(name='tmp', petsc_type='PetscScalar', grid=iet.functions[0].grid),
#                   'position': PetscObject(name='position', petsc_type='PetscInt')}

#     # collect the components for the MatContext struct
#     usr_ctx = [i for i in iet.parameters if isinstance(i, Symbol)]
#     sizes = [iet.functions[0].grid.dimensions[i].symbolic_size for i in range(len(iet.functions[0].grid.dimensions))]
#     usr_ctx.extend(sizes)
#     matctx = PetscStruct(usr_ctx)

#     # alter the iteration bounds since I turned off the shifting in lower_exprs
#     mapper = {}
#     for tree in retrieve_iteration_tree(iet):
#         for i in tree:
#             mapper[i] = i._rebuild(limits=(i.dim.symbolic_min-1, i.dim.symbolic_max+1, i.dim.symbolic_incr))
#     # nested is True since there is more than one mapping inside mapper
#     iet = Transformer(mapper, nested=True).visit(iet)

#     # Retrieve the adjusted iteration loop.
#     # Purposely not using the TIMER parts since I cannot supply 'struct profiler * timers' as an arg to the 
#     # call back function. Will need to review this
#     iteration_root = retrieve_iteration_tree(iet)[0][0]

#     # build call_back function body 
#     mymatshellmult_body = List(body=[c.Line('PetscFunctionBegin;'),
#                                      Definition(matctx),
#                                      Call('PetscCall', [Call('MatShellGetContext', arguments=[petsc_objs['A_matfree'], Byref(matctx.name)])]),
#                                      Definition(petsc_objs[iet.functions[0].name]),
#                                      Definition(petsc_objs[iet.functions[1].name]),
#                                      Call('PetscCall', [Call('VecGetArray2dRead', arguments=[petsc_objs['xvec'], FieldFromPointer(sizes[0].name, matctx.name), FieldFromPointer(sizes[1].name, matctx.name), 0, 0, Byref(iet.functions[1].name)])]),
#                                      Call('PetscCall', [Call('VecGetArray2d', arguments=[petsc_objs['yvec'], FieldFromPointer(sizes[0].name, matctx.name), FieldFromPointer(sizes[1].name, matctx.name), 0, 0, Byref(iet.functions[0].name)])]),
#                                      iteration_root,
#                                      Call('PetscCall', [Call('VecRestoreArray2dRead', arguments=[petsc_objs['xvec'], FieldFromPointer(sizes[0].name, matctx.name), FieldFromPointer(sizes[1].name, matctx.name), 0, 0, Byref(iet.functions[1].name)])]),
#                                      Call('PetscCall', [Call('VecRestoreArray2d', arguments=[petsc_objs['yvec'], FieldFromPointer(sizes[0].name, matctx.name), FieldFromPointer(sizes[1].name, matctx.name), 0, 0, Byref(iet.functions[0].name)])]),
#                                      Call('PetscFunctionReturn', arguments=[0])])

#     call_back = Callable('MyMatShellMult', mymatshellmult_body, retval=petsc_objs['retval'],
#                           parameters=(petsc_objs['A_matfree'], petsc_objs['xvec'], petsc_objs['yvec']))
    
#     iteration_root2 = retrieve_iteration_tree(call_back)[0][0]

#     # transform the very inner loop of call_back function to have the if, else section
#     # this may not ultimately be needed...but for current setup it is needed.
#     else_body = FindNodes(Expression).visit(iteration_root)[0]
#     then_body = DummyExpr(petsc_objs[iet.functions[0].name].indexify(), petsc_objs[iet.functions[1].name].indexify())
#     condition = sympy.Or(sympy.Eq(iet.functions[0].dimensions[0], iet.parameters[5]-1), sympy.Eq(iet.functions[0].dimensions[0], iet.parameters[4]+1),
#                          sympy.Eq(iet.functions[0].dimensions[1], iet.parameters[7]-1), sympy.Eq(iet.functions[0].dimensions[1], iet.parameters[6]+1))
#     call_back = Transformer({else_body: Conditional(condition, then_body, else_body)}).visit(call_back)

#     # from IPython import embed; embed()
#     # replace all of the struct MatContext symbols that appear in the callback function with a fieldfrompointer i.e ctx->symbol
#     for i in usr_ctx:
#         call_back = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(call_back)
#     call_back_arg = CallBack(call_back.name, 'void', 'void')


#     # to map petsc vector back to devito function
#     petsc_2_dev = DummyExpr(petsc_objs[iet.functions[0].name].indexify(indices=(petsc_objs['tmp'].dimensions[0]+iet.functions[0].space_order, petsc_objs['tmp'].dimensions[1]+iet.functions[0].space_order)),
#                             petsc_objs['tmp'].indexify(indices=(petsc_objs['tmp'].dimensions[0], petsc_objs['tmp'].dimensions[1])))
#     petsc_2_dev = Transformer({else_body: petsc_2_dev}).visit(iteration_root2)
#     for i in usr_ctx:
#         petsc_2_dev = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(petsc_2_dev)

    
#     # to map devito array to petsc vector
#     dev_2_petsc = List(body=[DummyExpr(petsc_objs['position'], (iet.functions[0].dimensions[0]-iet.functions[0].space_order)*sizes[1] + (iet.functions[0].dimensions[1]-iet.functions[0].space_order), init=True),
#                              Call('PetscCall', [Call('VecSetValue', arguments=[petsc_objs['b'], petsc_objs['position'], petsc_objs[iet.functions[0].name].indexify(), 'INSERT_VALUES'])])])
    
#     dev_2_petsc_iters = [lambda ex: Iteration(ex, iet.functions[0].dimensions[0], (iet.functions[0].space_order, iet.functions[0].space_order+sizes[0]-1, 1)),
#                          lambda ex: Iteration(ex, iet.functions[0].dimensions[1], (iet.functions[0].space_order, iet.functions[0].space_order+sizes[1]-1, 1))]
    
#     dev_2_petsc = dev_2_petsc_iters[0](dev_2_petsc_iters[1](dev_2_petsc))
    

#     # from IPython import embed; embed()
#     kernel_body = List(body=[Definition(petsc_objs['x']),
#                              Definition(petsc_objs['b']),
#                              Definition(petsc_objs['A_matfree']),
#                              Definition(petsc_objs['ksp']),
#                              Definition(petsc_objs['size']),
#                              Definition(petsc_objs['matsize']),
#                              DummyExpr(petsc_objs['matsize'], sizes[0]*sizes[1]),
#                              c.Line('PetscFunctionBeginUser;'),
#                              Call('PetscCall', [Call('PetscInitialize', arguments=['NULL', 'NULL', 'NULL', 'NULL'])]),
#                              Call('PetscCallMPI', [Call('MPI_Comm_size', arguments=['PETSC_COMM_WORLD', Byref(petsc_objs['size'].name)])]),
#                              Call('PetscCall', [Call('VecCreate', arguments=['PETSC_COMM_SELF', Byref(petsc_objs['x'].name)])]),
#                              Call('PetscCall', [Call('VecSetSizes', arguments=[petsc_objs['x'], 'PETSC_DECIDE', petsc_objs['matsize']])]),
#                              Call('PetscCall', [Call('VecSetFromOptions', arguments=[petsc_objs['x']])]),
#                              Call('PetscCall', [Call('VecDuplicate', arguments=[petsc_objs['x'], Byref(petsc_objs['b'].name)])]),
#                              dev_2_petsc,
#                              Call('PetscCall', [Call('MatCreateShell', arguments=['PETSC_COMM_WORLD', 'PETSC_DECIDE', 'PETSC_DECIDE', petsc_objs['matsize'], petsc_objs['matsize'], matctx.name, Byref(petsc_objs['A_matfree'].name)])]),
#                              Call('PetscCall', [Call('MatSetFromOptions', arguments=[petsc_objs['A_matfree']])]), 
#                              Call('PetscCall', [Call('MatShellSetOperation', arguments=[petsc_objs['A_matfree'], 'MATOP_MULT', call_back_arg])]),
#                              Call('PetscCall', [Call('KSPCreate', arguments=['PETSC_COMM_WORLD', Byref(petsc_objs['ksp'].name)])]),
#                              Call('PetscCall', [Call('KSPSetInitialGuessNonzero', arguments=[petsc_objs['ksp'], 'PETSC_TRUE'])]),
#                             #  you can then initialise x with initial guess
#                              Call('PetscCall', [Call('KSPSetOperators', arguments=[petsc_objs['ksp'], petsc_objs['A_matfree'], petsc_objs['A_matfree']])]),
#                              Call('PetscCall', [Call('KSPSetTolerances', arguments=[petsc_objs['ksp'], 1.e-5, 'PETSC_DEFAULT', 'PETSC_DEFAULT', 'PETSC_DEFAULT'])]),
#                             #  means that you can do ./exe -ksp_type fgmres etc -> probably delete since we would use KSPSetType(ksp, KSPCG);?
#                              Call('PetscCall', [Call('KSPSetFromOptions', arguments=[petsc_objs['ksp']])]),
#                              Call('PetscCall', [Call('KSPSolve', arguments=[petsc_objs['ksp'], petsc_objs['b'], petsc_objs['x']])]),
#                              Definition(petsc_objs['tmp']),
#                              Call('PetscCall', [Call('VecGetArray2d', arguments=[petsc_objs['x'], FieldFromPointer(sizes[0].name, matctx.name), FieldFromPointer(sizes[1].name, matctx.name), 0, 0, Byref(petsc_objs['tmp'].name)])]),
#                              petsc_2_dev,
#                              Call('PetscCall', [Call('VecRestoreArray2d', arguments=[petsc_objs['x'], FieldFromPointer(sizes[0].name, matctx.name), FieldFromPointer(sizes[1].name, matctx.name), 0, 0, Byref(petsc_objs['tmp'].name)])]),
#                              Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['x'].name)])]),
#                              Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['b'].name)])]),
#                              Call('PetscCall', [Call('MatDestroy', arguments=[Byref(petsc_objs['A_matfree'].name)])]),
#                              Call('PetscCall', [Call('KSPDestroy', arguments=[Byref(petsc_objs['ksp'].name)])]),
#                              Call('PetscCall', [Call('PetscFinalize')])])


#     iet = Transformer({iet.body.body[0]: kernel_body}).visit(iet)

#     for i in sizes:
#         iet = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(iet)

#     # add necessary include directories for petsc
#     kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/include')
#     kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/arch-linux-c-debug/include')
#     kwargs['compiler'].add_libraries('petsc')
#     libdir = '/home/zl5621/petsc/arch-linux-c-debug/lib'
#     kwargs['compiler'].add_library_dirs(libdir)
#     kwargs['compiler'].add_ldflags('-Wl,-rpath,%s' % libdir)


#     return iet, {'efuncs': [call_back],
#                  'includes': ['petscksp.h']}





# updating version using subdomains
@iet_pass
def lower_petsc(iet, **kwargs):
    # from IPython import embed; embed()

    petsc_objs = {'retval': PetscObject(name='retval', petsc_type='PetscErrorCode'),
                  'A_matfree': PetscObject(name='A_matfree', petsc_type='Mat'),
                  'xvec': PetscObject(name='xvec', petsc_type='Vec'),
                  'yvec': PetscObject(name='yvec', petsc_type='Vec'),
                  iet.functions[0].name : PetscObject(name=iet.functions[0].name,
                                                      petsc_type='PetscScalar', dimensions=iet.functions[0].dimensions, shape=iet.functions[0].shape),
                  iet.functions[1].name : PetscObject(name=iet.functions[1].name,
                                                      petsc_type='PetscScalar', grid=iet.functions[1].grid),
                  'x': PetscObject(name='x', petsc_type='Vec'),
                  'b': PetscObject(name='b', petsc_type='Vec'),
                  'ksp': PetscObject(name='ksp', petsc_type='KSP'),
                  'size': PetscObject(name='size', petsc_type='PetscMPIInt'),
                  'vec_size': PetscObject(name='vec_size', petsc_type='PetscInt'),
                  'tmp': PetscObject(name='tmp', petsc_type='PetscScalar', grid=iet.functions[1].grid),
                  'position': PetscObject(name='position', petsc_type='PetscInt'),
                  'pn_tmp': PetscObject(name='pn_tmp', petsc_type='PetscScalar', grid=iet.functions[1].grid),
                  'b_left': PetscObject(name='b_left', petsc_type='PetscScalar'),
                  'b_right': PetscObject(name='b_right', petsc_type='PetscScalar'),
                  'b_down': PetscObject(name='b_down', petsc_type='PetscScalar'),
                  'b_up': PetscObject(name='b_up', petsc_type='PetscScalar')}

    # collect the components for the MatContext struct
    usr_ctx = [i for i in iet.parameters if isinstance(i, Symbol)]
    sizes = [iet.functions[1].grid.dimensions[i].symbolic_size for i in range(len(iet.functions[1].grid.dimensions))]
    usr_ctx.extend(sizes)
    matctx = PetscStruct(usr_ctx)

    # from IPython import embed; embed()


    # building RHS vector b from the equation provided by the user:
    exprs = FindNodes(Expression).visit(iet)
    ordered_terms = exprs[0].expr.rhs.as_ordered_terms()
    ordered_terms_all = [i.as_ordered_terms() for i in ordered_terms]
    flattened_list = [term for sublist in ordered_terms_all for term in sublist]

    b_left_coeff = '[%s + %s, %s + %s]' % (iet.functions[0].dimensions[0], iet.functions[1].space_order-1,
                                           iet.functions[0].dimensions[1], iet.functions[1].space_order)

    b_right_coeff = '[%s + %s, %s + %s]' % (iet.functions[0].dimensions[0], iet.functions[1].space_order+1,
                                            iet.functions[0].dimensions[1], iet.functions[1].space_order)
    
    b_down_coeff = '[%s + %s, %s + %s]' % (iet.functions[0].dimensions[0], iet.functions[1].space_order,
                                            iet.functions[0].dimensions[1], iet.functions[1].space_order-1)
    
    b_up_coeff = '[%s + %s, %s + %s]' % (iet.functions[0].dimensions[0], iet.functions[1].space_order,
                                         iet.functions[0].dimensions[1], iet.functions[1].space_order+1)
    
    expr_left = [expr for expr in flattened_list if b_left_coeff in str(expr)][0]
    expr_right = [expr for expr in flattened_list if b_right_coeff in str(expr)][0]
    expr_down = [expr for expr in flattened_list if b_down_coeff in str(expr)][0]
    expr_up = [expr for expr in flattened_list if b_up_coeff in str(expr)][0]

    dims_op = {'i': Dimension(name='i'),
               'j': Dimension(name='j')}
    
    # from IPython import embed; embed()
    
    pos_line = DummyExpr(petsc_objs['position'], (iet.functions[0].dimensions[0]-1)*(FieldFromPointer(sizes[1].name, matctx.name)-2) + (iet.functions[0].dimensions[1]-1), init=True)
    
    b_iters = [lambda ex: Iteration(ex, iet.functions[0].dimensions[0], (1, FieldFromPointer(sizes[0].name, matctx.name)-2, 1)),
               lambda ex: Iteration(ex, iet.functions[0].dimensions[1], (1, FieldFromPointer(sizes[1].name, matctx.name)-2, 1))]
    

    condition1 = sympy.Eq(iet.functions[0].dimensions[0], 1)
    then_body1 = List(body=[DummyExpr(petsc_objs['b_left'], -expr_left, init=True),
                            Call('PetscCall', [Call('VecSetValue', arguments=[petsc_objs['b'], petsc_objs['position'], petsc_objs['b_left'], 'ADD_VALUES'])])
    ])
    conditional1 = Conditional(condition1, then_body1)

    # from IPython import embed; embed()

    condition2 = sympy.Eq(iet.functions[0].dimensions[0], FieldFromPointer(sizes[0].name, matctx.name)-2)
    then_body2 = List(body=[DummyExpr(petsc_objs['b_right'], -expr_right, init=True),
                            Call('PetscCall', [Call('VecSetValue', arguments=[petsc_objs['b'], petsc_objs['position'], petsc_objs['b_right'], 'ADD_VALUES'])])
    ])
    conditional2 = Conditional(condition2, then_body2)


    condition3 = sympy.Eq(iet.functions[0].dimensions[1], 1)
    then_body3 = List(body=[DummyExpr(petsc_objs['b_down'], -expr_down, init=True),
                            Call('PetscCall', [Call('VecSetValue', arguments=[petsc_objs['b'], petsc_objs['position'], petsc_objs['b_down'], 'ADD_VALUES'])])
    ])
    conditional3 = Conditional(condition3, then_body3)


    condition4 = sympy.Eq(iet.functions[0].dimensions[1], FieldFromPointer(sizes[1].name, matctx.name)-2)
    then_body4 = List(body=[DummyExpr(petsc_objs['b_up'], -expr_up, init=True),
                            Call('PetscCall', [Call('VecSetValue', arguments=[petsc_objs['b'], petsc_objs['position'], petsc_objs['b_up'], 'ADD_VALUES'])])
    ])
    conditional4 = Conditional(condition4, then_body4)

    build_b = b_iters[0](b_iters[1]([pos_line, conditional1, conditional2, conditional3, conditional4]))
    # from IPython import embed; embed()
    for i in usr_ctx:
        build_b = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(build_b)
    # from IPython import embed; embed()




    # Retrieve the adjusted iteration loop.
    # Purposely not using the TIMER parts since I cannot supply 'struct profiler * timers' as an arg to the 
    # call back function. Will need to review this
    iteration_root = retrieve_iteration_tree(iet)[0][0]

    call_back_iters = [lambda ex: Iteration(ex, dims_op['i'], (0, FieldFromPointer(sizes[0].name, matctx.name)+2*iet.functions[1].space_order-1, 1)),
                       lambda ex: Iteration(ex, dims_op['j'], (0, FieldFromPointer(sizes[1].name, matctx.name)+2*iet.functions[1].space_order-1, 1)),
                       lambda ex: Iteration(ex, dims_op['i'], (0, FieldFromPointer(sizes[0].name, matctx.name)-2-1, 1)),
                       lambda ex: Iteration(ex, dims_op['j'], (0, FieldFromPointer(sizes[1].name, matctx.name)-2-1, 1))]
    
    initalise = call_back_iters[0](call_back_iters[1](DummyExpr(petsc_objs[iet.functions[1].name].indexify(indices=(dims_op['i'],dims_op['j'])), 0.)))

    mapping = call_back_iters[2](call_back_iters[3](DummyExpr(petsc_objs[iet.functions[1].name].indexify(indices=(dims_op['i']+iet.functions[1].space_order+1,dims_op['j']+iet.functions[1].space_order+1)), petsc_objs['pn_tmp'].indexify(indices=(dims_op['i'],dims_op['j'])))))


    new_ptr = IndexedPointer(petsc_objs[iet.functions[1].name], (FieldFromPointer(sizes[0].name, matctx.name)+2*iet.functions[1].space_order,FieldFromPointer(sizes[1].name, matctx.name)+2*iet.functions[1].space_order))
    # build call_back function body 
    mymatshellmult_body = List(body=[c.Line('PetscFunctionBegin;'),
                                     Definition(matctx),
                                     Call('PetscCall', [Call('MatShellGetContext', arguments=[petsc_objs['A_matfree'], Byref(matctx.name)])]),
                                     Definition(petsc_objs['pn_tmp']),
                                     Definition(petsc_objs[iet.functions[0].name]),
                                     c.Line('PetscScalar ' + str(new_ptr) +';'),
                                    #  Definition(petsc_objs[iet.functions[1].name].indexify(indices=(FieldFromPointer(sizes[0].name, matctx.name)+2*iet.functions[1].space_order,))),
                                     Call('PetscCall', [Call('VecGetArray2dRead', arguments=[petsc_objs['xvec'], FieldFromPointer(sizes[0].name, matctx.name)-2, FieldFromPointer(sizes[1].name, matctx.name)-2, 0, 0, Byref(petsc_objs['pn_tmp'])])]),
                                     Call('PetscCall', [Call('VecGetArray2d', arguments=[petsc_objs['yvec'], FieldFromPointer(sizes[0].name, matctx.name)-2, FieldFromPointer(sizes[1].name, matctx.name)-2, 1, 1, Byref(iet.functions[0].name)])]),
                                     initalise,
                                     mapping,
                                     iteration_root,
                                     Call('PetscCall', [Call('VecRestoreArray2dRead', arguments=[petsc_objs['xvec'], FieldFromPointer(sizes[0].name, matctx.name)-2, FieldFromPointer(sizes[1].name, matctx.name)-2, 0, 0, Byref(petsc_objs['pn_tmp'])])]),
                                     Call('PetscCall', [Call('VecRestoreArray2d', arguments=[petsc_objs['yvec'], FieldFromPointer(sizes[0].name, matctx.name)-2, FieldFromPointer(sizes[1].name, matctx.name)-2, 1, 1, Byref(iet.functions[0].name)])]),
                                     Call('PetscFunctionReturn', arguments=[0])])

    call_back = Callable('MyMatShellMult', mymatshellmult_body, retval=petsc_objs['retval'],
                          parameters=(petsc_objs['A_matfree'], petsc_objs['xvec'], petsc_objs['yvec']))
    
    # from IPython import embed; embed()


    # from IPython import embed; embed()
    # replace all of the struct MatContext symbols that appear in the callback function with a fieldfrompointer i.e ctx->symbol
    for i in usr_ctx[:-2]:
        call_back = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(call_back)
    call_back_arg = CallBack(call_back.name, 'void', 'void')

    petsc_2_dev_mapping = DummyExpr(petsc_objs[iet.functions[1].name].indexify(indices=(iet.functions[0].dimensions[0]+iet.functions[1].space_order, iet.functions[0].dimensions[1]+iet.functions[1].space_order)), petsc_objs['pn_tmp'].indexify(indices=(iet.functions[0].dimensions[0], iet.functions[0].dimensions[1])))
    petsc_2_dev = b_iters[0](b_iters[1](petsc_2_dev_mapping))

    # from IPython import embed; embed()
    kernel_body = List(body=[Definition(petsc_objs['x']),
                             Definition(petsc_objs['b']),
                             Definition(petsc_objs['A_matfree']),
                             Definition(petsc_objs['ksp']),
                             Definition(petsc_objs['size']),
                             Definition(petsc_objs['vec_size']),
                             DummyExpr(petsc_objs['vec_size'], (FieldFromPointer(sizes[0].name, matctx.name)-2)*(FieldFromPointer(sizes[1].name, matctx.name)-2)),
                             c.Line('PetscFunctionBeginUser;'),
                             Call('PetscCall', [Call('PetscInitialize', arguments=['NULL', 'NULL', 'NULL', 'NULL'])]),
                             Call('PetscCallMPI', [Call('MPI_Comm_size', arguments=['PETSC_COMM_WORLD', Byref(petsc_objs['size'].name)])]),
                             Call('PetscCall', [Call('VecCreate', arguments=['PETSC_COMM_SELF', Byref(petsc_objs['x'].name)])]),
                             Call('PetscCall', [Call('VecSetSizes', arguments=[petsc_objs['x'], 'PETSC_DECIDE', petsc_objs['vec_size']])]),
                             Call('PetscCall', [Call('VecSetFromOptions', arguments=[petsc_objs['x']])]),
                             Call('PetscCall', [Call('VecDuplicate', arguments=[petsc_objs['x'], Byref(petsc_objs['b'].name)])]),
                             build_b,
                             Call('PetscCall', [Call('MatCreateShell', arguments=['PETSC_COMM_WORLD', 'PETSC_DECIDE', 'PETSC_DECIDE', petsc_objs['vec_size'], petsc_objs['vec_size'], matctx.name, Byref(petsc_objs['A_matfree'].name)])]),
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
                             Definition(petsc_objs['pn_tmp']),
                             Call('PetscCall', [Call('VecGetArray2d', arguments=[petsc_objs['x'], FieldFromPointer(sizes[0].name, matctx.name)-2, FieldFromPointer(sizes[1].name, matctx.name)-2, 1, 1, Byref(petsc_objs['pn_tmp'].name)])]),
                             petsc_2_dev,
                             Call('PetscCall', [Call('VecRestoreArray2d', arguments=[petsc_objs['x'], FieldFromPointer(sizes[0].name, matctx.name)-2, FieldFromPointer(sizes[1].name, matctx.name)-2, 1, 1, Byref(petsc_objs['pn_tmp'].name)])]),
                             Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['x'].name)])]),
                             Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['b'].name)])]),
                             Call('PetscCall', [Call('MatDestroy', arguments=[Byref(petsc_objs['A_matfree'].name)])]),
                             Call('PetscCall', [Call('KSPDestroy', arguments=[Byref(petsc_objs['ksp'].name)])]),
                             Call('PetscCall', [Call('PetscFinalize')])])


    iet = Transformer({iet.body.body[0]: kernel_body}).visit(iet)

    # from IPython import embed; embed()

    # for i in sizes:
    #     iet = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(iet)

    # add necessary include directories for petsc
    kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/include')
    kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/arch-linux-c-debug/include')
    kwargs['compiler'].add_libraries('petsc')
    libdir = '/home/zl5621/petsc/arch-linux-c-debug/lib'
    kwargs['compiler'].add_library_dirs(libdir)
    kwargs['compiler'].add_ldflags('-Wl,-rpath,%s' % libdir)


    return iet, {'efuncs': [call_back],
                 'includes': ['petscksp.h']}







# @iet_pass
# def lower_petsc(iet, **kwargs):
#     # from IPython import embed; embed()

#     # add necessary include directories for petsc
#     kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/include')
#     kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/arch-linux-c-debug/include')
#     kwargs['compiler'].add_libraries('petsc')
#     libdir = '/home/zl5621/petsc/arch-linux-c-debug/lib'
#     kwargs['compiler'].add_library_dirs(libdir)
#     kwargs['compiler'].add_ldflags('-Wl,-rpath,%s' % libdir)


    return iet, {
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

    
