#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"

struct MatContext
{
  float h_x;
  float h_y;
  int x_M;
  int x_m;
  int y_M;
  int y_m;
} ;

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
  double section2;
} ;

PetscErrorCode MyMatShellMult(Mat A_matfree, Vec xvec, Vec yvec);

int Kernel(struct dataobj *restrict dummy1_vec, struct dataobj *restrict dummy2_vec, struct dataobj *restrict pn_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, const float dt, const float h_x, const float h_y, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct MatContext * ctx, struct profiler * timers)
{
  Mat A_matfree;
  Vec b;
  DM da;
  KSP ksp;
  PC pc;
  PetscMPIInt size;
  Vec x;

  PetscScalar**restrict b_tmp;
  PetscScalar**restrict xvec_tmp;

  float (*restrict pn)[pn_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[pn_vec->size[1]]) pn_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  float (*restrict v)[v_vec->size[1]][v_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[v_vec->size[1]][v_vec->size[2]]) v_vec->data;

  START(section0)
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(NULL,NULL,NULL,NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&(size)));
  PetscCall(DMCreate2d(PETSC_COMM_SELF,DMDA_BOUNDARY_MIRROR,DMDA_BOUNDARY_MIRROR,DMDA_STENCIL_STAR,11,11,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMSetMatType(da,MATSHELL));
  PetscCall(DMCreateMatrix(da,&(A_matfree)));
  PetscCall(MatShellSetOperation(A_matfree,MATOP_MULT,(void (*)(void))MyMatShellMult));
  PetscCall(MatShellSetContext(A_matfree,ctx));
  PetscCall(KSPCreate(PETSC_COMM_SELF,&(ksp)));
  PetscCall(KSPSetOperators(ksp,A_matfree,A_matfree));
  PetscCall(KSPSetTolerances(ksp,1.00000000000000e-7F,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(KSPSetType(ksp,KSPGMRES));
  PetscCall(KSPGetPC(ksp,&(pc)));
  PetscCall(PCSetType(pc,PCJACOBI));
  PetscCall(PCSetType(pc,PC_JACOBI_DIAGONAL));
  PetscCall(KSPSetFromOptions(ksp));
  STOP(section0,timers)

  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {
    PetscCall(DMCreateGlobalVector(da,&(x)));
    PetscCall(DMCreateGlobalVector(da,&(b)));
    PetscCall(DMDAVecGetArray(da,b,&b_tmp));

    START(section1)
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        b_tmp[x][y] = -5.0e-1F*u[t0][x + 1][y + 2]/h_x + 5.0e-1F*u[t0][x + 3][y + 2]/h_x - 5.0e-1F*v[t0][x + 2][y + 1]/h_y + 5.0e-1F*v[t0][x + 2][y + 3]/h_y;
      }
    }
    STOP(section1,timers)

    PetscCall(DMDAVecRestoreArray(da,b,&b_tmp));
    PetscCall(KSPSolve(ksp,b,x));
    PetscCall(DMDAVecGetArray(da,x,&xvec_tmp));
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        pn[x + 2][y + 2] = xvec_tmp[x][y];
      }
    }
    PetscCall(DMDAVecRestoreArray(da,x,&xvec_tmp));

    START(section2)
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        u[t1][x + 2][y + 2] = dt*(-5.0e-1F*pn[x + 1][y + 2]/h_x + 5.0e-1F*pn[x + 3][y + 2]/h_x - (-5.0e-1F*u[t0][x + 2][y + 1]/h_y + 5.0e-1F*u[t0][x + 2][y + 3]/h_y)*v[t0][x + 2][y + 2] + u[t0][x + 2][y + 2]/dt);

        v[t1][x + 2][y + 2] = dt*(-(-5.0e-1F*v[t0][x + 1][y + 2]/h_x + 5.0e-1F*v[t0][x + 3][y + 2]/h_x)*u[t0][x + 2][y + 2] - 5.0e-1F*pn[x + 2][y + 1]/h_y + 5.0e-1F*pn[x + 2][y + 3]/h_y + v[t0][x + 2][y + 2]/dt);
      }
    }
    STOP(section2,timers)
  }

  return 0;
}

PetscErrorCode MyMatShellMult(Mat A_matfree, Vec xvec, Vec yvec)
{
  DM da;
  Vec local_xvec;

  PetscScalar**restrict xvec_tmp;
  PetscScalar**restrict yvec_tmp;

  PetscFunctionBeginUser;
  struct MatContext * ctx;
  PetscCall(MatShellGetContext(A_matfree,&(ctx)));
  PetscCall(MatGetDM(A_matfree,&(da)));
  PetscCall(MatGetLocalVector(da,&(local_xvec)));
  PetscCall(DMGlobalToLocalBegin(da,xvec,INSERT_VALUES,local_xvec));
  PetscCall(DMGlobalToLocalEnd(da,xvec,INSERT_VALUES,local_xvec));
  PetscCall(DMDAVecGetArrayRead(da,local_xvec,&xvec_tmp));
  PetscCall(DMDAVecGetArray(da,yvec,&yvec_tmp));
  for (int x = ctx->x_m; x <= ctx->x_M; x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      yvec_tmp[x][y] = -2.0F*pow(ctx->h_x, -2)*xvec_tmp[x][y] + pow(ctx->h_x, -2)*xvec_tmp[x - 1][y] + pow(ctx->h_x, -2)*xvec_tmp[x + 1][y] - 2.0F*pow(ctx->h_y, -2)*xvec_tmp[x][y] + pow(ctx->h_y, -2)*xvec_tmp[x][y - 1] + pow(ctx->h_y, -2)*xvec_tmp[x][y + 1];
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(da,local_xvec,&xvec_tmp));
  PetscCall(DMDAVecRestoreArray(da,yvec,&yvec_tmp));
  PetscCall(DMRestoreLocalVector(da,&(local_xvec)));
  PetscFunctionReturn(0);
}