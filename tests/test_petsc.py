from devito import Grid, TimeFunction, Function, Eq, solve, Operator
from devito.ir.iet import Call, ElementalFunction, Definition, DummyExpr
from devito.passes.iet.languages.C import CDataManager
from devito.types.petsc import (DM, Mat, Vec, PetscMPIInt, KSP,
                                PC, KSPConvergedReason, PETScArray)
import numpy as np
from devito.types import PETScSolve


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


# NOTE: This test is purely for illustration of what is currently
# produced by the PETScSolve class. It is not a test of correctness.
def test_petsc_solve():
    """
    """

    grid = Grid(shape=(11, 11), extent=(1., 1.))

    u = TimeFunction(name='u', grid=grid, space_order=2)
    v = TimeFunction(name='v', grid=grid, space_order=2)
    pn = Function(name='pn', grid=grid, space_order=2)

    eq_pn = Eq(pn.laplace, u.dxc+v.dyc)

    petsc = PETScSolve(eq_pn, pn)

    eq_u = Eq(u.dt + v*u.dyc - pn.dxc)
    eq_v = Eq(v.dt + u*v.dxc - pn.dyc)

    update_u = Eq(u.forward, solve(eq_u, u.forward))
    update_v = Eq(v.forward, solve(eq_v, v.forward))

    # # Create the operator
    exprs = petsc + [update_u, update_v]

    op = Operator(exprs, opt='noop')

    assert str(op.ccode) == """\
#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"

struct dataobj
{
  void *restrict data;
  unsigned long * size;
  unsigned long * npsize;
  unsigned long * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
  void * dmap;
} ;

struct profiler
{
  double section0;
  double section1;
} ;

PetscErrorCode MyMatShellMult(Mat A_matfree, Vec xvec, Vec yvec);

int Kernel(struct dataobj *restrict pn_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, const float dt, const float h_x, const float h_y, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct profiler * timers)
{
  PetscScalar**restrict yvec_tmp;

  float (*restrict pn)[pn_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[pn_vec->size[1]]) pn_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  float (*restrict v)[v_vec->size[1]][v_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[v_vec->size[1]][v_vec->size[2]]) v_vec->data;

  PetscCall(MatShellSetOperation(A_matfree,MATOP_MULT,(void (*)(void))MyMatShellMult));
  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {
    START(section1)
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        u[t1][x + 2][y + 2] = dt*(-5.0e-1F*pn[x + 1][y + 2]/h_x + 5.0e-1F*pn[x + 3][y + 2]/h_x - (-5.0e-1F*u[t0][x + 2][y + 1]/h_y + 5.0e-1F*u[t0][x + 2][y + 3]/h_y)*v[t0][x + 2][y + 2] + u[t0][x + 2][y + 2]/dt);

        v[t1][x + 2][y + 2] = dt*(-(-5.0e-1F*v[t0][x + 1][y + 2]/h_x + 5.0e-1F*v[t0][x + 3][y + 2]/h_x)*u[t0][x + 2][y + 2] - 5.0e-1F*pn[x + 2][y + 1]/h_y + 5.0e-1F*pn[x + 2][y + 3]/h_y + v[t0][x + 2][y + 2]/dt);
      }
    }
    STOP(section1,timers)
  }

  return 0;
}

PetscErrorCode MyMatShellMult(Mat A_matfree, Vec xvec, Vec yvec)
{
  START(section0)
  for (int x = x_m; x <= x_M; x += 1)
  {
    for (int y = y_m; y <= y_M; y += 1)
    {
      yvec_tmp[x][y] = pn[x + 1][y + 2]/pow(h_x, 2) - 2.0F*pn[x + 2][y + 2]/pow(h_x, 2) + pn[x + 3][y + 2]/pow(h_x, 2) + pn[x + 2][y + 1]/pow(h_y, 2) - 2.0F*pn[x + 2][y + 2]/pow(h_y, 2) + pn[x + 2][y + 3]/pow(h_y, 2);
    }
  }
  STOP(section0,timers)
}
"""
