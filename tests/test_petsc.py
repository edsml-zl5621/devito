import numpy as np
from devito import Grid, Function, Eq, Operator
from devito.ir.iet import (Call, ElementalFunction, Definition, DummyExpr,
                           MatVecAction, FindNodes, RHSLinearSystem,
                           PointerCast)
from devito.passes.iet.languages.C import CDataManager
from devito.types import (DM, Mat, Vec, PetscMPIInt, KSP,
                          PC, KSPConvergedReason, PETScArray, PETScSolve)


def test_petsc_local_object():
    """
    Test C++ support for PETSc LocalObjects.
    """
    lo0 = DM('da')
    lo1 = Mat('A')
    lo2 = Vec('x')
    lo3 = PetscMPIInt('size')
    lo4 = KSP('ksp')
    lo5 = PC('pc')
    lo6 = KSPConvergedReason('reason')

    iet = Call('foo', [lo0, lo1, lo2, lo3, lo4, lo5, lo6])
    iet = ElementalFunction('foo', iet, parameters=())

    dm = CDataManager(sregistry=None)
    iet = CDataManager.place_definitions.__wrapped__(dm, iet)[0]

    assert 'DM da;' in str(iet)
    assert 'Mat A;' in str(iet)
    assert 'Vec x;' in str(iet)
    assert 'PetscMPIInt size;' in str(iet)
    assert 'KSP ksp;' in str(iet)
    assert 'PC pc;' in str(iet)
    assert 'KSPConvergedReason reason;' in str(iet)


def test_petsc_functions():
    """
    Test C++ support for PETScArrays.
    """
    grid = Grid((2, 2))
    x, y = grid.dimensions

    ptr0 = PETScArray(name='ptr0', dimensions=grid.dimensions, dtype=np.float32)
    ptr1 = PETScArray(name='ptr1', dimensions=grid.dimensions, dtype=np.float32,
                      is_const=True)
    ptr2 = PETScArray(name='ptr2', dimensions=grid.dimensions, dtype=np.float64,
                      is_const=True)
    ptr3 = PETScArray(name='ptr3', dimensions=grid.dimensions, dtype=np.int32)
    ptr4 = PETScArray(name='ptr4', dimensions=grid.dimensions, dtype=np.int64,
                      is_const=True)

    defn0 = Definition(ptr0)
    defn1 = Definition(ptr1)
    defn2 = Definition(ptr2)
    defn3 = Definition(ptr3)
    defn4 = Definition(ptr4)

    expr = DummyExpr(ptr0.indexed[x, y], ptr1.indexed[x, y] + 1)

    assert str(defn0) == 'PetscScalar*restrict ptr0_vec;'
    assert str(defn1) == 'const PetscScalar*restrict ptr1_vec;'
    assert str(defn2) == 'const PetscScalar*restrict ptr2_vec;'
    assert str(defn3) == 'PetscInt*restrict ptr3_vec;'
    assert str(defn4) == 'const PetscInt*restrict ptr4_vec;'
    assert str(expr) == 'ptr0[x][y] = ptr1[x][y] + 1;'


def test_petsc_subs():
    """
    Test support for PETScArrays in substitutions.
    """
    grid = Grid((2, 2))

    f1 = Function(name='f1', grid=grid, space_order=2)
    f2 = Function(name='f2', grid=grid, space_order=2)

    arr = PETScArray(name='arr', dimensions=f2.dimensions, dtype=f2.dtype)

    eqn = Eq(f1, f2.laplace)
    eqn_subs = eqn.subs(f2, arr)

    assert str(eqn) == 'Eq(f1(x, y), Derivative(f2(x, y), (x, 2))' +  \
        ' + Derivative(f2(x, y), (y, 2)))'

    assert str(eqn_subs) == 'Eq(f1(x, y), Derivative(arr(x, y), (x, 2))' +  \
        ' + Derivative(arr(x, y), (y, 2)))'

    assert str(eqn_subs.rhs.evaluate) == '-2.0*arr(x, y)/h_x**2' + \
        ' + arr(x - h_x, y)/h_x**2 + arr(x + h_x, y)/h_x**2 - 2.0*arr(x, y)/h_y**2' + \
        ' + arr(x, y - h_y)/h_y**2 + arr(x, y + h_y)/h_y**2'


def test_petsc_solve():
    """
    Test PETScSolve.
    """
    grid = Grid((2, 2))

    f = Function(name='f', grid=grid, space_order=2)
    g = Function(name='g', grid=grid, space_order=2)

    eqn = Eq(f.laplace, g)

    petsc = PETScSolve(eqn, f)

    op = Operator(petsc, opt='noop')

    action_expr = FindNodes(MatVecAction).visit(op)

    rhs_expr = FindNodes(RHSLinearSystem).visit(op)

    assert str(action_expr[-1].expr.rhs.args[0]) == \
        'x_matvec_f[x + 1, y + 2]/h_x**2 - 2.0*x_matvec_f[x + 2, y + 2]/h_x**2' + \
        ' + x_matvec_f[x + 3, y + 2]/h_x**2 + x_matvec_f[x + 2, y + 1]/h_y**2' + \
        ' - 2.0*x_matvec_f[x + 2, y + 2]/h_y**2 + x_matvec_f[x + 2, y + 3]/h_y**2'

    assert str(rhs_expr[-1].expr.rhs.args[0]) == 'g[x + 2, y + 2]'

    # Check the iteration bounds are correct.
    assert op.arguments().get('x_m') == 0
    assert op.arguments().get('y_m') == 0
    assert op.arguments().get('y_M') == 1
    assert op.arguments().get('x_M') == 1

    # Check the target
    assert rhs_expr[-1].expr.rhs.target == f
    assert action_expr[-1].expr.rhs.target == f

    # Check the solver parameters
    assert rhs_expr[-1].expr.rhs.solver_parameters == \
        {'ksp_type': 'gmres', 'pc_type': 'jacobi'}
    assert action_expr[-1].expr.rhs.solver_parameters == \
        {'ksp_type': 'gmres', 'pc_type': 'jacobi'}


def test_petsc_cast():
    """
    Test casting of PETScArray.
    """
    g0 = Grid((2))
    g1 = Grid((2, 2))
    g2 = Grid((2, 2, 2))

    arr0 = PETScArray(name='arr0', dimensions=g0.dimensions, shape=g0.shape)
    arr1 = PETScArray(name='arr1', dimensions=g1.dimensions, shape=g1.shape)
    arr2 = PETScArray(name='arr2', dimensions=g2.dimensions, shape=g2.shape)

    # Casts will be explictly generated and placed at specific locations in the C code,
    # specifically after various other PETSc calls have been executed.
    cast0 = PointerCast(arr0)
    cast1 = PointerCast(arr1)
    cast2 = PointerCast(arr2)

    assert str(cast0) == \
        'PetscScalar (*restrict arr0) = (PetscScalar (*)) arr0_vec;'
    assert str(cast1) == \
        'PetscScalar (*restrict arr1)[info.gym] = (PetscScalar (*)[info.gym]) arr1_vec;'
    assert str(cast2) == \
        'PetscScalar (*restrict arr2)[info.gym][info.gzm] = ' + \
        '(PetscScalar (*)[info.gym][info.gzm]) arr2_vec;'
