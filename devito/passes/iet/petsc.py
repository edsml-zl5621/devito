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


# DMDA version
@iet_pass
def lower_petsc(iet, **kwargs):

    action = FindNodes(Expression).visit(iet)[0]
    rhs_func = action.write
    lhs_func = action.reads[0]

    petsc_objs = {'retval': PetscObject(name='retval', petsc_type='PetscErrorCode'),
                  'A_matfree': PetscObject(name='A_matfree', petsc_type='Mat'),
                  'pc': PetscObject(name='pc', petsc_type='PC'),
                  'xvec': PetscObject(name='xvec', petsc_type='Vec'),
                  'yvec': PetscObject(name='yvec', petsc_type='Vec'),
                  'local_xvec': PetscObject(name='local_xvec', petsc_type='Vec'),
                  'local_yvec': PetscObject(name='local_yvec', petsc_type='Vec'),
                  rhs_func.name : PetscObject(name=rhs_func.name, petsc_type='PetscScalar',
                                              dimensions=rhs_func.dimensions, shape=rhs_func.shape),
                  lhs_func.name : PetscObject(name=lhs_func.name, petsc_type='PetscScalar',
                                              dimensions=lhs_func.dimensions, shape=lhs_func.shape),
                  'x': PetscObject(name='x', petsc_type='Vec'),
                  'b': PetscObject(name='b', petsc_type='Vec'),
                  'ksp': PetscObject(name='ksp', petsc_type='KSP'),
                  'size': PetscObject(name='size', petsc_type='PetscMPIInt'),
                  'position': PetscObject(name='position', petsc_type='PetscInt'),
                  '%s_tmp' % lhs_func.name: PetscObject(name='%s_tmp' % lhs_func.name,
                                                        petsc_type='PetscScalar',
                                                        dimensions=rhs_func.dimensions,
                                                        shape=rhs_func.shape),
                  'b_val': PetscObject(name='b_val', petsc_type='PetscScalar'),
                  's_o': PetscObject(name='s_o', petsc_type='PetscInt'),
                  'da': PetscObject(name='da', petsc_type='DM')}
    

    # collect the components for the MatContext struct
    usr_ctx = [i for i in iet.parameters if isinstance(i, Symbol)]
    sizes_lhs = [lhs_func.dimensions[i].symbolic_size for i in range(len(lhs_func.dimensions))]
    usr_ctx.extend(sizes_lhs)
    matctx = PetscStruct(usr_ctx, lhs_func.space_order)


    # MAP DEVITO FUNCTION VALUES TO GLOBAL PETSC VECTOR b
    mapping_iters = [lambda ex: Iteration(ex, lhs_func.dimensions[0],
                                          (lhs_func.dimensions[0].symbolic_min,
                                           lhs_func.dimensions[0].symbolic_max, 1)),
                         lambda ex: Iteration(ex, lhs_func.dimensions[1],
                                              (lhs_func.dimensions[1].symbolic_min,
                                               lhs_func.dimensions[1].symbolic_max, 1))]
    
    pos_line = DummyExpr(petsc_objs['position'],
                         lhs_func.dimensions[0]*FieldFromPointer(sizes_lhs[0].name, matctx.name) + lhs_func.dimensions[1],
                         init=True)
    val = DummyExpr(petsc_objs['b_val'],
                    lhs_func.indexify(indices=(lhs_func.dimensions[0]+FieldFromPointer(petsc_objs['s_o'].name, matctx.name),
                                               lhs_func.dimensions[1]+FieldFromPointer(petsc_objs['s_o'].name, matctx.name))),
                                               init=True)
    vecsetval = Call('PetscCall', [Call('VecSetValue',
                                        arguments=[petsc_objs['b'], petsc_objs['position'],
                                                   petsc_objs['b_val'], 'ADD_VALUES'])])

    build_b = mapping_iters[0](mapping_iters[1](List(body=[pos_line, val, vecsetval])))

    for i in usr_ctx:
        build_b = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(build_b)

    ###########################################################################

    # get the core iteration loops
    iters = retrieve_iteration_tree(iet)

    mymatshellmult_body = List(body=[c.Line('PetscFunctionBegin;'),  # tmp for now
                                     Definition(matctx),
                                     Call('PetscCall', [Call('MatShellGetContext',
                                                             arguments=[petsc_objs['A_matfree'],
                                                                        Byref(matctx.name)])]),
                                     Definition(petsc_objs[rhs_func.name]),
                                     Definition(petsc_objs[lhs_func.name]),
                                     Definition(petsc_objs['da']),
                                     Definition(petsc_objs['local_xvec']),
                                     Definition(petsc_objs['local_yvec']),
                                     Call('PetscCall', [Call('MatGetDM',
                                                             arguments=[petsc_objs['A_matfree'],
                                                                        Byref(petsc_objs['da'].name)])]),
                                     Call('PetscCall', [Call('DMGetLocalVector',
                                                             arguments=[petsc_objs['da'],
                                                                        Byref(petsc_objs['local_xvec'].name)])]),
                                     Call('PetscCall', [Call('DMGetLocalVector',
                                                             arguments=[petsc_objs['da'],
                                                                        Byref(petsc_objs['local_yvec'].name)])]),
                                     Call('PetscCall', [Call('DMGlobalToLocalBegin',
                                                             arguments=[petsc_objs['da'],
                                                                        petsc_objs['xvec'],
                                                                        'INSERT_VALUES',
                                                                        petsc_objs['local_xvec']])]),
                                     Call('PetscCall', [Call('DMGlobalToLocalEnd',
                                                             arguments=[petsc_objs['da'],
                                                                        petsc_objs['xvec'],
                                                                        'INSERT_VALUES',
                                                                        petsc_objs['local_xvec']])]),
                                     Call('PetscCall', [Call('DMDAVecGetArrayRead',
                                                             arguments=[petsc_objs['da'],
                                                                        petsc_objs['local_xvec'],
                                                                        Byref(petsc_objs[lhs_func.name].name)])]),
                                     Call('PetscCall', [Call('DMDAVecGetArray',
                                                             arguments=[petsc_objs['da'],
                                                                        petsc_objs['local_yvec'],
                                                                        Byref(petsc_objs[rhs_func.name].name)])]),
                                     iters[0].root,
                                     iters[1],
                                     iters[2],
                                     Call('PetscCall', [Call('DMDAVecRestoreArrayRead',
                                                             arguments=[petsc_objs['da'],
                                                                        petsc_objs['local_xvec'],
                                                                        Byref(petsc_objs[lhs_func.name])])]),
                                     Call('PetscCall', [Call('DMDAVecRestoreArray',
                                                             arguments=[petsc_objs['da'],
                                                                        petsc_objs['local_yvec'],
                                                                        Byref(petsc_objs[rhs_func.name])])]),
                                     Call('PetscCall', [Call('DMLocalToGlobalBegin',
                                                             arguments=[petsc_objs['da'],
                                                                        petsc_objs['local_yvec'],
                                                                        'INSERT_VALUES',
                                                                        petsc_objs['yvec']])]),
                                     Call('PetscCall', [Call('DMLocalToGlobalEnd',
                                                             arguments=[petsc_objs['da'],
                                                                        petsc_objs['local_yvec'],
                                                                        'INSERT_VALUES',
                                                                        petsc_objs['yvec']])]),
                                     Call('PetscCall', [Call('DMRestoreLocalVector',
                                                             arguments=[petsc_objs['da'],
                                                                        Byref(petsc_objs['local_xvec'].name)])]),
                                     Call('PetscCall', [Call('DMRestoreLocalVector',
                                                             arguments=[petsc_objs['da'],
                                                                        Byref(petsc_objs['local_yvec'].name)])]),
                                     Call('PetscFunctionReturn', arguments=[0])])
    


    call_back = Callable('MyMatShellMult', mymatshellmult_body, retval=petsc_objs['retval'],
                          parameters=(petsc_objs['A_matfree'], petsc_objs['xvec'], petsc_objs['yvec']))
    

    # replace all of the struct MatContext symbols that appear in the callback function with a fieldfrompointer i.e ctx->symbol
    for i in usr_ctx[:-2]:
        call_back = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(call_back)


    # PASS SOLUTION VECTOR x back to Devito function
    petsc_2_dev_mapping = DummyExpr(petsc_objs[lhs_func.name].indexify(indices=(lhs_func.dimensions[0]+FieldFromPointer(petsc_objs['s_o'].name, matctx.name), lhs_func.dimensions[1]+FieldFromPointer(petsc_objs['s_o'].name, matctx.name))), petsc_objs['%s_tmp' % lhs_func.name].indexify(indices=(lhs_func.dimensions[0], lhs_func.dimensions[1])))
    petsc_2_dev = mapping_iters[0](mapping_iters[1](petsc_2_dev_mapping))

    for i in usr_ctx:
        petsc_2_dev = Uxreplace({i: FieldFromPointer(i, matctx)}).visit(petsc_2_dev)


    # BUILD MAIN KERNEL BODY
    kernel_body = List(body=[Definition(petsc_objs['x']),
                             Definition(petsc_objs['b']),
                             Definition(petsc_objs['A_matfree']),
                             Definition(petsc_objs['ksp']),
                             Definition(petsc_objs['pc']),
                             Definition(petsc_objs['da']),
                             Definition(petsc_objs['size']),
                             c.Line('PetscFunctionBeginUser;'), #tmp
                             Call('PetscCall', [Call('PetscInitialize',
                                                     arguments=['NULL', 'NULL', 'NULL', 'NULL'])]),
                             Call('PetscCallMPI', [Call('MPI_Comm_size',
                                                        arguments=['PETSC_COMM_WORLD',
                                                                   Byref(petsc_objs['size'].name)])]),
                             Call('PetscCall', [Call('DMDACreate2d',
                                                     arguments=['PETSC_COMM_SELF', 'DM_BOUNDARY_GHOSTED',
                                                                'DM_BOUNDARY_GHOSTED', 'DMDA_STENCIL_BOX',
                                                                FieldFromPointer(sizes_lhs[0].name,matctx.name),
                                                                FieldFromPointer(sizes_lhs[1].name,matctx.name),
                                                                # 3rd last arg below would come from stencil width
                                                                'PETSC_DECIDE', 'PETSC_DECIDE', 1, 1,
                                                                'NULL', 'NULL', Byref(petsc_objs['da'].name)])]),
                             Call('PetscCall', [Call('DMSetFromOptions', arguments=[petsc_objs['da']])]),
                             Call('PetscCall', [Call('DMSetUp', arguments=[petsc_objs['da']])]),
                             Call('PetscCall', [Call('DMCreateGlobalVector',
                                                     arguments=[petsc_objs['da'], Byref(petsc_objs['x'].name)])]),
                             Call('PetscCall', [Call('DMCreateGlobalVector',
                                                     arguments=[petsc_objs['da'], Byref(petsc_objs['b'].name)])]),
                             Call('PetscCall', [Call('DMSetMatType',
                                                     arguments=[petsc_objs['da'], 'MATSHELL'])]),
                             Call('PetscCall', [Call('DMCreateMatrix',
                                                     arguments=[petsc_objs['da'],
                                                                Byref(petsc_objs['A_matfree'].name)])]),
                             Call('PetscCall', [Call('MatShellSetOperation',
                                                     arguments=[petsc_objs['A_matfree'],
                                                                'MATOP_MULT',
                                                                CallBack(call_back.name, 'void', 'void')])]),
                             Call('PetscCall', [Call('MatShellSetContext',
                                                     arguments=[petsc_objs['A_matfree'], matctx.name])]),
                             build_b,
                             Call('PetscCall', [Call('KSPCreate',
                                                     arguments=['PETSC_COMM_SELF',
                                                                Byref(petsc_objs['ksp'].name)])]),
                             Call('PetscCall', [Call('KSPSetOperators',
                                                     arguments=[petsc_objs['ksp'], petsc_objs['A_matfree'],
                                                                petsc_objs['A_matfree']])]),
                             Call('PetscCall', [Call('KSPSetTolerances',
                                                     arguments=[petsc_objs['ksp'], 1.e-8,
                                                                'PETSC_DEFAULT', 'PETSC_DEFAULT',
                                                                'PETSC_DEFAULT'])]),
                             Call('PetscCall', [Call('KSPSetType',
                                                     arguments=[petsc_objs['ksp'], 'KSPGMRES'])]),
                             Call('PetscCall', [Call('KSPGetPC',
                                                     arguments=[petsc_objs['ksp'], Byref(petsc_objs['pc'].name)])]),
                             Call('PetscCall', [Call('PCSetType',
                                                     arguments=[petsc_objs['pc'], 'PCNONE'])]),
                             Call('PetscCall', [Call('KSPSetFromOptions',
                                                     arguments=[petsc_objs['ksp']])]),
                             Call('PetscCall', [Call('KSPSolve',
                                                     arguments=[petsc_objs['ksp'], petsc_objs['b'],
                                                                petsc_objs['x']])]),
                             Definition(petsc_objs['%s_tmp' % lhs_func.name]),
                             Call('PetscCall', [Call('VecGetArray2d',
                                                     arguments=[petsc_objs['x'],
                                                                FieldFromPointer(sizes_lhs[0].name,matctx.name),
                                                                FieldFromPointer(sizes_lhs[1].name,matctx.name),
                                                                0, 0, Byref(petsc_objs['%s_tmp' % lhs_func.name].name)])]),
                             petsc_2_dev,
                             Call('PetscCall', [Call('VecRestoreArray2d',
                                                     arguments=[petsc_objs['x'],
                                                                FieldFromPointer(sizes_lhs[0].name,matctx.name),
                                                                FieldFromPointer(sizes_lhs[1].name,matctx.name),
                                                                0, 0, Byref(petsc_objs['%s_tmp' % lhs_func.name].name)])]),
                             Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['x'].name)])]),
                             Call('PetscCall', [Call('VecDestroy', arguments=[Byref(petsc_objs['b'].name)])]),
                             Call('PetscCall', [Call('MatDestroy', arguments=[Byref(petsc_objs['A_matfree'].name)])]),
                             Call('PetscCall', [Call('KSPDestroy', arguments=[Byref(petsc_objs['ksp'].name)])]),
                             Call('PetscCall', [Call('PetscFinalize')])])

    iet = Transformer({iet.body.body[0]: kernel_body}).visit(iet)

    # add necessary include directories for petsc
    kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/include')
    kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/arch-linux-c-debug/include')
    kwargs['compiler'].add_libraries('petsc')
    libdir = '/home/zl5621/petsc/arch-linux-c-debug/lib'
    kwargs['compiler'].add_library_dirs(libdir)
    kwargs['compiler'].add_ldflags('-Wl,-rpath,%s' % libdir)


    return iet, {'efuncs': [call_back],
                 'includes': ['petscksp.h', 'petscdmda.h']}



class PetscStruct(CompositeObject):

    __rargs__ = ('usr_ctx', 'space_order',)

    def __init__(self, usr_ctx, space_order):

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

    
