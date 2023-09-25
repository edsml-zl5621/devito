from ctypes import POINTER, c_int
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


# # no subdomain and shifting turned off - not working now
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
#     matctx = PetscStruct(usr_ctx, iet.functions[0].space_order)

#     # alter the iteration bounds since I turned off the shifting in lower_exprs
#     mapper = {}
#     for tree in retrieve_iteration_tree(iet):
#         for i in tree:
#             mapper[i] = i._rebuild(limits=(i.dim.symbolic_min-2, i.dim.symbolic_max+2, i.dim.symbolic_incr))
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
    
#     dev_2_petsc_iters = [lambda ex: Iteration(ex, iet.functions[0].dimensions[0], (iet.functions[0].space_order, iet.functions[0].space_order+sizes[0]-2, 1)),
#                          lambda ex: Iteration(ex, iet.functions[0].dimensions[1], (iet.functions[0].space_order, iet.functions[0].space_order+sizes[1]-2, 1))]
    
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
#                              c.Line('PetscCall(VecView(b, PETSC_VIEWER_STDOUT_SELF));'),
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
#                             #  c.Line('PetscCall(VecView(x, PETSC_VIEWER_STDOUT_SELF));'),
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




# # # using subdomains - WORKING BUT CONVERGENCE WRONG FOR SPACE ORDER 4
# @iet_pass
# def lower_petsc(iet, **kwargs):

#     # from IPython import embed; embed()
#     # collect all exprs that appear in first loop - doing this for now since when I add BC equations, they
#     # appear in iet.functions 
#     # code only works if a user supplies 1 eqn (excluding bcs)
#     exprs = FindNodes(Expression).visit(iet)[0]
#     rhs_func = exprs.write
#     lhs_func = exprs.reads[0]


#     petsc_objs = {'retval': PetscObject(name='retval', petsc_type='PetscErrorCode'),
#                   'A_matfree': PetscObject(name='A_matfree', petsc_type='Mat'),
#                   'xvec': PetscObject(name='xvec', petsc_type='Vec'),
#                   'yvec': PetscObject(name='yvec', petsc_type='Vec'),
#                   rhs_func.name : PetscObject(name=rhs_func.name, petsc_type='PetscScalar', dimensions=rhs_func.dimensions, shape=rhs_func.shape),
#                   lhs_func.name : PetscObject(name=lhs_func.name, petsc_type='PetscScalar', dimensions=lhs_func.dimensions, shape=lhs_func.shape),
#                   'x': PetscObject(name='x', petsc_type='Vec'),
#                   'b': PetscObject(name='b', petsc_type='Vec'),
#                   'ksp': PetscObject(name='ksp', petsc_type='KSP'),
#                   'size': PetscObject(name='size', petsc_type='PetscMPIInt'),
#                   'vec_size': PetscObject(name='vec_size', petsc_type='PetscInt'),
#                   'position': PetscObject(name='position', petsc_type='PetscInt'),
#                   '%s_tmp' % lhs_func.name: PetscObject(name='%s_tmp' % lhs_func.name, petsc_type='PetscScalar', dimensions=rhs_func.dimensions, shape=rhs_func.shape),
#                   'b_val': PetscObject(name='b_val', petsc_type='PetscScalar'),
#                   'xvec_halo': PetscObject(name='xvec_halo', petsc_type='Vec'),
#                   'position_extended': PetscObject(name='position_extended', petsc_type='PetscInt'),
#                   'val_extended': PetscObject(name='val_extended', petsc_type='PetscScalar'),
#                   'xsize': PetscObject(name='xsize', petsc_type='PetscInt'),
#                   'ysize': PetscObject(name='ysize', petsc_type='PetscInt'),
#                   'xsize_halo': PetscObject(name='xsize_halo', petsc_type='PetscInt'),
#                   'ysize_halo': PetscObject(name='ysize_halo', petsc_type='PetscInt'),
#                   's_o': PetscObject(name='s_o', petsc_type='PetscInt')}
    
#     # from IPython import embed; embed()

#     # collect the components for the MatContext struct
#     usr_ctx = [i for i in iet.parameters if isinstance(i, Symbol)]
#     # sizes_rhs = [rhs_func.dimensions[i].symbolic_size for i in range(len(rhs_func.dimensions))]
#     sizes_lhs = [lhs_func.dimensions[i].symbolic_size for i in range(len(lhs_func.dimensions))]
#     # usr_ctx.extend(sizes_rhs)
#     usr_ctx.extend(sizes_lhs)
#     matctx = PetscStruct(usr_ctx, lhs_func.space_order)

#     # from IPython import embed; embed()

#     # building RHS vector b from the equation provided by the user:
#     # # this only works for 2D for now......

#     eqn_dims = [exprs.expr.ispace[i].dim for i in range(len(exprs.expr.dimensions))]
    
#     itertree = retrieve_iteration_tree(iet)[0]

#     vec_size0 = rhs_func.symbolic_shape[0]
#     vec_size1 = rhs_func.symbolic_shape[1]

#     iters = FindNodes(Iteration).visit(iet)

#     xmin_terms = iters[0].symbolic_min.as_ordered_terms()
#     ymin_terms = iters[1].symbolic_min.as_ordered_terms()


#     pos_line = DummyExpr(petsc_objs['position'], (eqn_dims[0]-xmin_terms[0]-xmin_terms[1])*vec_size1 + (eqn_dims[1]-ymin_terms[0]-ymin_terms[1]), init=True)


#     # from IPython import embed; embed()
#     val = DummyExpr(petsc_objs['b_val'], -rhs_func.indexify(), init=True)
#     vecsetval = Call('PetscCall', [Call('VecSetValue', arguments=[petsc_objs['b'], petsc_objs['position'], petsc_objs['b_val'], 'ADD_VALUES'])])


#     build_b = Transformer({exprs: List(body=[pos_line, exprs, val, vecsetval])}).visit(itertree.root)

#     for i in usr_ctx:
#         build_b = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(build_b)


#     xsize = rhs_func.dimensions[0].symbolic_size
#     ysize = rhs_func.dimensions[1].symbolic_size

#     # from IPython import embed; embed()

#     # build call_back function body 
#     x_exended_xsize = FieldFromPointer(lhs_func.dimensions[0].symbolic_size.name, matctx.name) + (2*FieldFromPointer(petsc_objs['s_o'].name, matctx.name))
#     x_exended_ysize = FieldFromPointer(lhs_func.dimensions[1].symbolic_size.name, matctx.name) + (2*FieldFromPointer(petsc_objs['s_o'].name, matctx.name))


#     pos_line_xextended = DummyExpr(petsc_objs['position_extended'], (eqn_dims[0]+FieldFromPointer(petsc_objs['s_o'].name, matctx.name))*(FieldFromPointer(lhs_func.dimensions[1].symbolic_size.name, matctx.name)+2*FieldFromPointer(petsc_objs['s_o'].name, matctx.name)) + (eqn_dims[1]+FieldFromPointer(petsc_objs['s_o'].name, matctx.name)), init=True)
#     val2 = DummyExpr(petsc_objs['val_extended'], petsc_objs['%s_tmp' % lhs_func.name].indexify(), init=True)
#     vec_extended_setval = Call('PetscCall', [Call('VecSetValue', arguments=[petsc_objs['xvec_halo'], petsc_objs['position_extended'], petsc_objs['val_extended'], 'ADD_VALUES'])])
#     initalise_extended = Transformer({exprs: List(body=[pos_line_xextended, val2, vec_extended_setval])}).visit(itertree.root)
#     # from IPython import embed; embed()

#     mymatshellmult_body = List(body=[c.Line('PetscFunctionBegin;'),  # temp solution to forming this line for now
#                                      Definition(matctx),
#                                      Call('PetscCall', [Call('MatShellGetContext', arguments=[petsc_objs['A_matfree'], Byref(matctx.name)])]),
#                                      Definition(petsc_objs['%s_tmp' % lhs_func.name]),
#                                      Definition(petsc_objs[rhs_func.name]),
#                                      Definition(petsc_objs[lhs_func.name]),
#                                      DummyExpr(petsc_objs['xsize'], xsize, init=True),
#                                      DummyExpr(petsc_objs['ysize'], ysize, init=True),
#                                      DummyExpr(petsc_objs['xsize_halo'], x_exended_xsize, init=True),
#                                      DummyExpr(petsc_objs['ysize_halo'], x_exended_ysize, init=True),
#                                      Definition(petsc_objs['xvec_halo']),
#                                      Call('PetscCall', [Call('VecCreate', arguments=['PETSC_COMM_SELF', Byref(petsc_objs['xvec_halo'].name)])]),
#                                      Call('PetscCall', [Call('VecSetSizes', arguments=[petsc_objs['xvec_halo'], 'PETSC_DECIDE', petsc_objs['xsize_halo']*petsc_objs['ysize_halo']])]),
#                                      Call('PetscCall', [Call('VecSetFromOptions', arguments=[petsc_objs['xvec_halo']])]),
#                                      Call('PetscCall', [Call('VecGetArray2dRead', arguments=[petsc_objs['xvec'], petsc_objs['xsize'], petsc_objs['ysize'], FieldFromPointer(xmin_terms[0]._C_name, matctx.name), FieldFromPointer(ymin_terms[0]._C_name, matctx.name), Byref(petsc_objs['%s_tmp' % lhs_func.name])])]),
#                                      Call('PetscCall', [Call('VecGetArray2d', arguments=[petsc_objs['yvec'], petsc_objs['xsize'], petsc_objs['ysize'], FieldFromPointer(xmin_terms[0]._C_name, matctx.name), FieldFromPointer(ymin_terms[0]._C_name, matctx.name), Byref(rhs_func.name)])]),
#                                      initalise_extended,
#                                      Call('PetscCall', [Call('VecGetArray2dRead', arguments=[petsc_objs['xvec_halo'], petsc_objs['xsize_halo'], petsc_objs['ysize_halo'], 0, 0, Byref(petsc_objs[lhs_func.name])])]),
#                                      itertree.root,
#                                      Call('PetscCall', [Call('VecRestoreArray2dRead', arguments=[petsc_objs['xvec'], petsc_objs['xsize'], petsc_objs['ysize'], FieldFromPointer(xmin_terms[0]._C_name, matctx.name), FieldFromPointer(ymin_terms[0]._C_name, matctx.name), Byref(petsc_objs['%s_tmp' % lhs_func.name])])]),
#                                      Call('PetscCall', [Call('VecRestoreArray2dRead', arguments=[petsc_objs['xvec_halo'], petsc_objs['xsize_halo'], petsc_objs['ysize_halo'], 0, 0, Byref(petsc_objs[lhs_func.name])])]),
#                                      Call('PetscCall', [Call('VecRestoreArray2d', arguments=[petsc_objs['yvec'], petsc_objs['xsize'], petsc_objs['ysize'], FieldFromPointer(xmin_terms[0]._C_name, matctx.name), FieldFromPointer(ymin_terms[0]._C_name, matctx.name), Byref(rhs_func.name)])]),
#                                      Call('PetscFunctionReturn', arguments=[0])])

#     call_back = Callable('MyMatShellMult', mymatshellmult_body, retval=petsc_objs['retval'],
#                           parameters=(petsc_objs['A_matfree'], petsc_objs['xvec'], petsc_objs['yvec']))
    
#     # from IPython import embed; embed()

#     # replace all of the struct MatContext symbols that appear in the callback function with a fieldfrompointer i.e ctx->symbol
#     for i in usr_ctx[:-2]:
#         call_back = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(call_back)
#     call_back_arg = CallBack(call_back.name, 'void', 'void')

#     petsc_2_dev_mapping = DummyExpr(petsc_objs[lhs_func.name].indexify(indices=(eqn_dims[0]+FieldFromPointer(petsc_objs['s_o'].name, matctx.name), eqn_dims[1]+FieldFromPointer(petsc_objs['s_o'].name, matctx.name))), petsc_objs['%s_tmp' % lhs_func.name].indexify(indices=(eqn_dims[0], eqn_dims[1])))
#     petsc_2_dev = Transformer({exprs: petsc_2_dev_mapping}).visit(itertree.root)

#     for i in usr_ctx:
#         petsc_2_dev = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(petsc_2_dev)

#     # from IPython import embed; embed()


#     # from IPython import embed; embed()
#     kernel_body = List(body=[Definition(petsc_objs['x']),
#                              Definition(petsc_objs['b']),
#                              Definition(petsc_objs['A_matfree']),
#                              Definition(petsc_objs['ksp']),
#                             #  c.Line('PetscBool   flg;'),
#                              Definition(petsc_objs['size']),
#                              DummyExpr(petsc_objs['vec_size'], vec_size0*vec_size1, init=True),
#                              c.Line('PetscFunctionBeginUser;'),
#                              Call('PetscCall', [Call('PetscInitialize', arguments=['NULL', 'NULL', 'NULL', 'NULL'])]),
#                              Call('PetscCallMPI', [Call('MPI_Comm_size', arguments=['PETSC_COMM_WORLD', Byref(petsc_objs['size'].name)])]),
#                              Call('PetscCall', [Call('VecCreate', arguments=['PETSC_COMM_SELF', Byref(petsc_objs['x'].name)])]),
#                              Call('PetscCall', [Call('VecSetSizes', arguments=[petsc_objs['x'], 'PETSC_DECIDE', petsc_objs['vec_size']])]),
#                              Call('PetscCall', [Call('VecSetFromOptions', arguments=[petsc_objs['x']])]),
#                              Call('PetscCall', [Call('VecDuplicate', arguments=[petsc_objs['x'], Byref(petsc_objs['b'].name)])]),
#                              build_b,
#                              c.Line('PetscCall(VecView(b, PETSC_VIEWER_STDOUT_SELF));'),
#                              Call('PetscCall', [Call('MatCreateShell', arguments=['PETSC_COMM_WORLD', 'PETSC_DECIDE', 'PETSC_DECIDE', petsc_objs['vec_size'], petsc_objs['vec_size'], matctx.name, Byref(petsc_objs['A_matfree'].name)])]),
#                              Call('PetscCall', [Call('MatSetFromOptions', arguments=[petsc_objs['A_matfree']])]), 
#                              Call('PetscCall', [Call('MatShellSetOperation', arguments=[petsc_objs['A_matfree'], 'MATOP_MULT', call_back_arg])]),
#                             #  c.Line('PetscCall(MatIsLinear(A_matfree, 10, &flg));'),
#                             #  c.Line('PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "shell matrix is non-linear.");'),
#                              Call('PetscCall', [Call('KSPCreate', arguments=['PETSC_COMM_WORLD', Byref(petsc_objs['ksp'].name)])]),
#                              c.Line('KSPSetType(ksp, KSPCG);'),
#                              #  you can then initialise x with initial guess
#                              #  Call('PetscCall', [Call('KSPSetInitialGuessNonzero', arguments=[petsc_objs['ksp'], 'PETSC_TRUE'])]),
#                              Call('PetscCall', [Call('KSPSetOperators', arguments=[petsc_objs['ksp'], petsc_objs['A_matfree'], petsc_objs['A_matfree']])]),
#                              Call('PetscCall', [Call('KSPSetTolerances', arguments=[petsc_objs['ksp'], 1.e-7, 'PETSC_DEFAULT', 'PETSC_DEFAULT', 'PETSC_DEFAULT'])]),
#                              #  means that you can do ./exe -ksp_type fgmres etc -> probably delete since we would use KSPSetType(ksp, KSPCG);?
#                              Call('PetscCall', [Call('KSPSetFromOptions', arguments=[petsc_objs['ksp']])]),
#                              Call('PetscCall', [Call('KSPSolve', arguments=[petsc_objs['ksp'], petsc_objs['b'], petsc_objs['x']])]),
#                              c.Line('PetscCall(VecView(x, PETSC_VIEWER_STDOUT_SELF));'),
#                              Definition(petsc_objs['%s_tmp' % lhs_func.name]),
#                              Call('PetscCall', [Call('VecGetArray2d', arguments=[petsc_objs['x'], vec_size0, vec_size1, FieldFromPointer(xmin_terms[0]._C_name, matctx.name), FieldFromPointer(ymin_terms[0]._C_name, matctx.name), Byref(petsc_objs['%s_tmp' % lhs_func.name].name)])]),
#                              petsc_2_dev,
#                              Call('PetscCall', [Call('VecRestoreArray2d', arguments=[petsc_objs['x'], vec_size0, vec_size1, FieldFromPointer(xmin_terms[0]._C_name, matctx.name), FieldFromPointer(ymin_terms[0]._C_name, matctx.name), Byref(petsc_objs['%s_tmp' % lhs_func.name].name)])]),
#                              Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['x'].name)])]),
#                              Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['b'].name)])]),
#                              Call('PetscCall', [Call('MatDestroy', arguments=[Byref(petsc_objs['A_matfree'].name)])]),
#                              Call('PetscCall', [Call('KSPDestroy', arguments=[Byref(petsc_objs['ksp'].name)])]),
#                              Call('PetscCall', [Call('PetscFinalize')])])


#     iet = Transformer({iet.body.body[0]: kernel_body}).visit(iet)

#     # from IPython import embed; embed()

#     # add necessary include directories for petsc
#     kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/include')
#     kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/arch-linux-c-debug/include')
#     kwargs['compiler'].add_libraries('petsc')
#     libdir = '/home/zl5621/petsc/arch-linux-c-debug/lib'
#     kwargs['compiler'].add_library_dirs(libdir)
#     kwargs['compiler'].add_ldflags('-Wl,-rpath,%s' % libdir)


#     return iet, {'efuncs': [call_back],
#                  'includes': ['petscksp.h']}




# NOT moving to RHS, vectors of size computational domain and using subdomains
# MORE LOOPS FOR BC EQUATIONS
@iet_pass
def lower_petsc(iet, **kwargs):

    # from IPython import embed; embed()
    # collect all exprs that appear in first loop - doing this for now since when I add BC equations, they
    # appear in iet.functions 
    # code only works if a user supplies 1 eqn (excluding bcs)
    exprs = FindNodes(Expression).visit(iet)[0]
    rhs_func = exprs.write
    lhs_func = exprs.reads[0]


    petsc_objs = {'retval': PetscObject(name='retval', petsc_type='PetscErrorCode'),
                  'A_matfree': PetscObject(name='A_matfree', petsc_type='Mat'),
                  'P': PetscObject(name='P', petsc_type='Mat'),
                  'pc': PetscObject(name='pc', petsc_type='PC'),
                  'xvec': PetscObject(name='xvec', petsc_type='Vec'),
                  'yvec': PetscObject(name='yvec', petsc_type='Vec'),
                  rhs_func.name : PetscObject(name=rhs_func.name, petsc_type='PetscScalar', dimensions=rhs_func.dimensions, shape=rhs_func.shape),
                  lhs_func.name : PetscObject(name=lhs_func.name, petsc_type='PetscScalar', dimensions=lhs_func.dimensions, shape=lhs_func.shape),
                  'x': PetscObject(name='x', petsc_type='Vec'),
                  'b': PetscObject(name='b', petsc_type='Vec'),
                  'ksp': PetscObject(name='ksp', petsc_type='KSP'),
                  'size': PetscObject(name='size', petsc_type='PetscMPIInt'),
                  'vec_size': PetscObject(name='vec_size', petsc_type='PetscInt'),
                  'position': PetscObject(name='position', petsc_type='PetscInt'),
                  '%s_tmp' % lhs_func.name: PetscObject(name='%s_tmp' % lhs_func.name, petsc_type='PetscScalar', dimensions=rhs_func.dimensions, shape=rhs_func.shape),
                  'b_val': PetscObject(name='b_val', petsc_type='PetscScalar'),
                  'xvec_halo': PetscObject(name='xvec_halo', petsc_type='Vec'),
                  'position_extended': PetscObject(name='position_extended', petsc_type='PetscInt'),
                  'val_extended': PetscObject(name='val_extended', petsc_type='PetscScalar'),
                  'xsize': PetscObject(name='xsize', petsc_type='PetscInt'),
                  'ysize': PetscObject(name='ysize', petsc_type='PetscInt'),
                  'xsize_halo': PetscObject(name='xsize_halo', petsc_type='PetscInt'),
                  'ysize_halo': PetscObject(name='ysize_halo', petsc_type='PetscInt'),
                  's_o': PetscObject(name='s_o', petsc_type='PetscInt')}
    
    # from IPython import embed; embed()

    # collect the components for the MatContext struct
    usr_ctx = [i for i in iet.parameters if isinstance(i, Symbol)]
    # sizes_rhs = [rhs_func.dimensions[i].symbolic_size for i in range(len(rhs_func.dimensions))]
    sizes_lhs = [lhs_func.dimensions[i].symbolic_size for i in range(len(lhs_func.dimensions))]
    # usr_ctx.extend(sizes_rhs)
    usr_ctx.extend(sizes_lhs)
    matctx = PetscStruct(usr_ctx, lhs_func.space_order)

    # from IPython import embed; embed()

    # building RHS vector b
    # rhs vector b is initialised from the function that is defined on the entire
    # domain i.e pn

    main_iter = retrieve_iteration_tree(iet)[0]

    
    # from IPython import embed; embed()
    eqn_dims = [exprs.expr.ispace[i].dim for i in range(len(exprs.expr.dimensions))]
    mapping_iters = [lambda ex: Iteration(ex, lhs_func.dimensions[0], (lhs_func.dimensions[0].symbolic_min, lhs_func.dimensions[0].symbolic_max, 1)),
                         lambda ex: Iteration(ex, lhs_func.dimensions[1], (lhs_func.dimensions[1].symbolic_min, lhs_func.dimensions[1].symbolic_max, 1))]
    # from IPython import embed; embed()
    pos_line = DummyExpr(petsc_objs['position'], lhs_func.dimensions[0]*FieldFromPointer(sizes_lhs[0].name, matctx.name) + lhs_func.dimensions[1], init=True)
    val = DummyExpr(petsc_objs['b_val'], lhs_func.indexify(indices=(lhs_func.dimensions[0]+FieldFromPointer(petsc_objs['s_o'].name, matctx.name), lhs_func.dimensions[1]+FieldFromPointer(petsc_objs['s_o'].name, matctx.name))), init=True)
    vecsetval = Call('PetscCall', [Call('VecSetValue', arguments=[petsc_objs['b'], petsc_objs['position'], petsc_objs['b_val'], 'ADD_VALUES'])])

    build_b = mapping_iters[0](mapping_iters[1](List(body=[pos_line, val, vecsetval])))

    # from IPython import embed; embed()

    for i in usr_ctx:
        build_b = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(build_b)


    # from IPython import embed; embed()

    dirichlet_xloop = retrieve_iteration_tree(iet)[1]
    dirichlet_yloop = retrieve_iteration_tree(iet)[2]

    # from IPython import embed; embed()


    # from IPython import embed; embed()

    # build call_back function body 
    x_exended_xsize = FieldFromPointer(lhs_func.dimensions[0].symbolic_size.name, matctx.name) + (2*FieldFromPointer(petsc_objs['s_o'].name, matctx.name))
    x_exended_ysize = FieldFromPointer(lhs_func.dimensions[1].symbolic_size.name, matctx.name) + (2*FieldFromPointer(petsc_objs['s_o'].name, matctx.name))


    pos_line_xextended = DummyExpr(petsc_objs['position_extended'], (lhs_func.dimensions[0] +FieldFromPointer(petsc_objs['s_o'].name, matctx.name))*petsc_objs['xsize_halo'] + FieldFromPointer(petsc_objs['s_o'].name, matctx.name) + lhs_func.dimensions[1], init=True)
    val2 = DummyExpr(petsc_objs['val_extended'], petsc_objs['%s_tmp' % lhs_func.name].indexify(indices=(lhs_func.dimensions[0], lhs_func.dimensions[1])), init=True)
    vec_extended_setval = Call('PetscCall', [Call('VecSetValue', arguments=[petsc_objs['xvec_halo'], petsc_objs['position_extended'], petsc_objs['val_extended'], 'ADD_VALUES'])])

    initalise_extended = mapping_iters[0](mapping_iters[1](List(body=[pos_line_xextended, val2, vec_extended_setval])))

    mymatshellmult_body = List(body=[c.Line('PetscFunctionBegin;'),  # temp solution to forming this line for now
                                     Definition(matctx),
                                     Call('PetscCall', [Call('MatShellGetContext', arguments=[petsc_objs['A_matfree'], Byref(matctx.name)])]),
                                     Definition(petsc_objs['%s_tmp' % lhs_func.name]),
                                     Definition(petsc_objs[rhs_func.name]),
                                     Definition(petsc_objs[lhs_func.name]),
                                     DummyExpr(petsc_objs['xsize_halo'], x_exended_xsize, init=True),
                                     DummyExpr(petsc_objs['ysize_halo'], x_exended_ysize, init=True),
                                     Definition(petsc_objs['xvec_halo']),
                                     Call('PetscCall', [Call('VecCreate', arguments=['PETSC_COMM_SELF', Byref(petsc_objs['xvec_halo'].name)])]),
                                     Call('PetscCall', [Call('VecSetSizes', arguments=[petsc_objs['xvec_halo'], 'PETSC_DECIDE', petsc_objs['xsize_halo']*petsc_objs['ysize_halo']])]),
                                     Call('PetscCall', [Call('VecSetFromOptions', arguments=[petsc_objs['xvec_halo']])]),
                                     Call('PetscCall', [Call('VecGetArray2dRead', arguments=[petsc_objs['xvec'], FieldFromPointer(sizes_lhs[0].name, matctx.name), FieldFromPointer(sizes_lhs[1].name, matctx.name), 0, 0, Byref(petsc_objs['%s_tmp' % lhs_func.name])])]),
                                     Call('PetscCall', [Call('VecGetArray2d', arguments=[petsc_objs['yvec'], FieldFromPointer(sizes_lhs[0].name, matctx.name), FieldFromPointer(sizes_lhs[1].name, matctx.name), 0, 0, Byref(rhs_func.name)])]),
                                     initalise_extended,
                                     Call('PetscCall', [Call('VecGetArray2dRead', arguments=[petsc_objs['xvec_halo'], petsc_objs['xsize_halo'], petsc_objs['ysize_halo'], 0, 0, Byref(petsc_objs[lhs_func.name])])]),
                                     main_iter.root,
                                     dirichlet_xloop,
                                     dirichlet_yloop,
                                     Call('PetscCall', [Call('VecRestoreArray2dRead', arguments=[petsc_objs['xvec'], FieldFromPointer(sizes_lhs[0].name, matctx.name), FieldFromPointer(sizes_lhs[1].name, matctx.name), 0, 0, Byref(petsc_objs['%s_tmp' % lhs_func.name])])]),
                                     Call('PetscCall', [Call('VecRestoreArray2dRead', arguments=[petsc_objs['xvec_halo'], petsc_objs['xsize_halo'], petsc_objs['ysize_halo'], 0, 0, Byref(petsc_objs[lhs_func.name])])]),
                                     Call('PetscCall', [Call('VecRestoreArray2d', arguments=[petsc_objs['yvec'], FieldFromPointer(sizes_lhs[0].name, matctx.name), FieldFromPointer(sizes_lhs[1].name, matctx.name), 0, 0, Byref(rhs_func.name)])]),
                                     Call('PetscFunctionReturn', arguments=[0])])

    call_back = Callable('MyMatShellMult', mymatshellmult_body, retval=petsc_objs['retval'],
                          parameters=(petsc_objs['A_matfree'], petsc_objs['xvec'], petsc_objs['yvec']))
    
    # from IPython import embed; embed()

    # replace all of the struct MatContext symbols that appear in the callback function with a fieldfrompointer i.e ctx->symbol
    for i in usr_ctx[:-2]:
        call_back = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(call_back)

    # from IPython import embed; embed()
    call_back_arg = CallBack(call_back.name, 'void', 'void')

    # from IPython import embed; embed()

    petsc_2_dev_mapping = DummyExpr(petsc_objs[lhs_func.name].indexify(indices=(lhs_func.dimensions[0]+FieldFromPointer(petsc_objs['s_o'].name, matctx.name), lhs_func.dimensions[1]+FieldFromPointer(petsc_objs['s_o'].name, matctx.name))), petsc_objs['%s_tmp' % lhs_func.name].indexify(indices=(lhs_func.dimensions[0], lhs_func.dimensions[1])))
    petsc_2_dev = mapping_iters[0](mapping_iters[1](petsc_2_dev_mapping))




    for i in usr_ctx:
        petsc_2_dev = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(petsc_2_dev)

    # from IPython import embed; embed()


    # from IPython import embed; embed()
    kernel_body = List(body=[Definition(petsc_objs['x']),
                             Definition(petsc_objs['b']),
                             Definition(petsc_objs['A_matfree']),
                             Definition(petsc_objs['P']),
                             Definition(petsc_objs['ksp']),
                             Definition(petsc_objs['pc']),
                             c.Line('PetscBool   flg;'),
                             c.Line('PetscInt   its;'),
                             Definition(petsc_objs['size']),
                             DummyExpr(petsc_objs['vec_size'], sizes_lhs[0]*sizes_lhs[1], init=True),
                             c.Line('PetscFunctionBeginUser;'),
                             Call('PetscCall', [Call('PetscInitialize', arguments=['NULL', 'NULL', 'NULL', 'NULL'])]),
                             Call('PetscCallMPI', [Call('MPI_Comm_size', arguments=['PETSC_COMM_WORLD', Byref(petsc_objs['size'].name)])]),
                             Call('PetscCall', [Call('VecCreate', arguments=['PETSC_COMM_SELF', Byref(petsc_objs['x'].name)])]),
                             Call('PetscCall', [Call('VecSetSizes', arguments=[petsc_objs['x'], 'PETSC_DECIDE', petsc_objs['vec_size']])]),
                             Call('PetscCall', [Call('VecSetFromOptions', arguments=[petsc_objs['x']])]),
                             Call('PetscCall', [Call('VecDuplicate', arguments=[petsc_objs['x'], Byref(petsc_objs['b'].name)])]),
                             Call('PetscCall', [Call('MatCreate', arguments=['PETSC_COMM_SELF', Byref(petsc_objs['P'].name)])]),
                             Call('PetscCall', [Call('MatSetSizes', arguments=[petsc_objs['P'], 'PETSC_DECIDE', 'PETSC_DECIDE', petsc_objs['vec_size'], petsc_objs['vec_size']])]),
                             Call('PetscCall', [Call('MatSetFromOptions', arguments=[petsc_objs['P']])]),
                             Call('PetscCall', [Call('MatSetUp', arguments=[petsc_objs['P']])]),
                             Call('PetscCall', [Call('MatAssemblyBegin', arguments=[petsc_objs['P'], 'MAT_FINAL_ASSEMBLY'])]),
                             Call('PetscCall', [Call('MatAssemblyEnd', arguments=[petsc_objs['P'], 'MAT_FINAL_ASSEMBLY'])]),
                             build_b,
                             c.Line('PetscCall(VecView(b, PETSC_VIEWER_STDOUT_SELF));'),
                             Call('PetscCall', [Call('MatCreateShell', arguments=['PETSC_COMM_WORLD', 'PETSC_DECIDE', 'PETSC_DECIDE', petsc_objs['vec_size'], petsc_objs['vec_size'], matctx.name, Byref(petsc_objs['A_matfree'].name)])]),
                             Call('PetscCall', [Call('MatSetFromOptions', arguments=[petsc_objs['A_matfree']])]), 
                             Call('PetscCall', [Call('MatShellSetOperation', arguments=[petsc_objs['A_matfree'], 'MATOP_MULT', call_back_arg])]),
                             c.Line('PetscCall(MatIsLinear(A_matfree, 10, &flg));'),
                             c.Line('PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "shell matrix is non-linear.");'),
                             Call('PetscCall', [Call('KSPCreate', arguments=['PETSC_COMM_WORLD', Byref(petsc_objs['ksp'].name)])]),

                             #  you can then initialise x with initial guess
                            #  Call('PetscCall', [Call('KSPSetInitialGuessNonzero', arguments=[petsc_objs['ksp'], 'PETSC_TRUE'])]),
                             Call('PetscCall', [Call('KSPSetOperators', arguments=[petsc_objs['ksp'], petsc_objs['A_matfree'], petsc_objs['P']])]),
                             Call('PetscCall', [Call('KSPSetTolerances', arguments=[petsc_objs['ksp'], 1.e-8, 'PETSC_DEFAULT', 'PETSC_DEFAULT', 'PETSC_DEFAULT'])]),
                             Call('PetscCall', [Call('KSPSetType', arguments=[petsc_objs['ksp'], 'KSPGMRES'])]),
                             Call('PetscCall', [Call('KSPGetPC', arguments=[petsc_objs['ksp'], Byref(petsc_objs['pc'].name)])]),
                             Call('PetscCall', [Call('PCSetType', arguments=[petsc_objs['pc'], 'PCNONE'])]),
                             #  means that you can do ./exe -ksp_type fgmres etc -> probably delete since we would use KSPSetType(ksp, KSPCG);?
                             Call('PetscCall', [Call('KSPSetFromOptions', arguments=[petsc_objs['ksp']])]),
                             Call('PetscCall', [Call('KSPSetUp', arguments=[petsc_objs['ksp']])]),
                             Call('PetscCall', [Call('KSPSolve', arguments=[petsc_objs['ksp'], petsc_objs['b'], petsc_objs['x']])]),
                             
                             c.Line('PetscCall(KSPGetIterationNumber(ksp, &its));'),
                             c.Line('KSPType ksptype;'),
                             c.Line('PetscCall(KSPGetType(ksp, &ksptype));'),
                             c.Line('PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Using %s", ksptype));'),
                             c.Line('PetscCall(PetscPrintf(PETSC_COMM_WORLD, "no of iters %d", its));'),

                            #  c.Line('PetscCall(VecView(x, PETSC_VIEWER_STDOUT_SELF));'),
                             Definition(petsc_objs['%s_tmp' % lhs_func.name]),
                             Call('PetscCall', [Call('VecGetArray2d', arguments=[petsc_objs['x'], FieldFromPointer(sizes_lhs[0].name,matctx.name), FieldFromPointer(sizes_lhs[1].name,matctx.name), 0, 0, Byref(petsc_objs['%s_tmp' % lhs_func.name].name)])]),
                             petsc_2_dev,
                             Call('PetscCall', [Call('VecRestoreArray2d', arguments=[petsc_objs['x'], FieldFromPointer(sizes_lhs[0].name,matctx.name), FieldFromPointer(sizes_lhs[1].name,matctx.name), 0, 0, Byref(petsc_objs['%s_tmp' % lhs_func.name].name)])]),
                             Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['x'].name)])]),
                             Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['b'].name)])]),
                             Call('PetscCall', [Call('MatDestroy', arguments=[Byref(petsc_objs['P'].name)])]),
                             Call('PetscCall', [Call('MatDestroy', arguments=[Byref(petsc_objs['A_matfree'].name)])]),
                             Call('PetscCall', [Call('KSPDestroy', arguments=[Byref(petsc_objs['ksp'].name)])]),
                             Call('PetscCall', [Call('PetscFinalize')])])

    # from IPython import embed; embed()
    iet = Transformer({iet.body.body[0]: kernel_body}).visit(iet)

    for i in sizes_lhs:
        iet = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(iet)

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





class PetscStruct(CompositeObject):

    __rargs__ = ('usr_ctx', 'space_order',)

    def __init__(self, usr_ctx, space_order):
        # from IPython import embed; embed()

        self._usr_ctx = usr_ctx
        self._space_order = space_order

        pfields = [(i._C_name, dtype_to_ctype(i.dtype)) for i in self.usr_ctx if isinstance(i, Symbol)]

        pfields.extend([('s_o', c_int)])

        super(PetscStruct, self).__init__('ctx', 'MatContext', pfields)
    
    @property
    def usr_ctx(self):
        return self._usr_ctx

    @property
    def space_order(self):
        return self._space_order
    
    # maybe also want to _arg_defaults here?

    def _arg_values(self, **kwargs):
        values = super()._arg_values(**kwargs)
        for i in self.fields:
            if i == 's_o':
                setattr(values[self.name]._obj, i, self.space_order)
            else:
                setattr(values[self.name]._obj, i, kwargs['args'][i])
        return values

    
