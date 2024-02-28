from devito import Grid, Function, Operator
from devito.ir.iet import (Call, ElementalFunction, Definition,
                           DummyExpr, FindNodes, Expression)
from devito.passes.iet.languages.C import CDataManager
from devito.types import (DM, Mat, Vec, PetscMPIInt, KSP,
                          PC, KSPConvergedReason, PETScArray, Action)
import numpy as np


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

    assert str(defn0) == 'PetscScalar**restrict ptr0;'
    assert str(defn1) == 'const PetscScalar**restrict ptr1;'
    assert str(defn2) == 'const PetscScalar**restrict ptr2;'
    assert str(defn3) == 'PetscInt**restrict ptr3;'
    assert str(defn4) == 'const PetscInt**restrict ptr4;'
    assert str(expr) == 'ptr0[x][y] = ptr1[x][y] + 1;'


def test_no_shifting():

    grid = Grid((2, 2))

    arr = PETScArray(name='arr', dimensions=grid.dimensions)

    f1 = Function(name='f1', grid=grid, space_order=2)

    action = Action(arr, f1.laplace)

    op = Operator([action], opt='noop')

    expr = FindNodes(Expression).visit(op)

    assert str(expr[-1]) == 'arr[x][y] = -2.0F*f1[x][y]/pow(h_x, 2) + ' \
                            'f1[x - 1][y]/pow(h_x, 2) + f1[x + 1][y]/pow(h_x, 2) - ' \
                            '2.0F*f1[x][y]/pow(h_y, 2) + f1[x][y - 1]/pow(h_y, 2) + ' \
                            'f1[x][y + 1]/pow(h_y, 2);'

    assert op.arguments().get('x_m') == 0
