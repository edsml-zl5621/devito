#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "petscksp.h"
#include "petscdmda.h"

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
  double section3;
  double section4;
  double section5;
  double section6;
} ;

PetscErrorCode MyMatShellMult(Mat A_matfree, Vec xvec, Vec yvec);
PetscErrorCode preconditioner_callback(Mat A_matfree, Vec yvec);

int Kernel(const float nu, struct dataobj *restrict pn_vec, const float rho, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, const float dt, const float h_x, const float h_y, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct MatContext * ctx, struct profiler * timers)
{
  Mat A_matfree;
  Vec b;
  DM da;
  KSP ksp;
  PC pc;
  PetscErrorCode reason;
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
  PetscCall(DMDACreate2d(PETSC_COMM_SELF,DM_BOUNDARY_MIRROR,DM_BOUNDARY_MIRROR,DMDA_STENCIL_STAR,51,51,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&(da)));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMSetMatType(da,MATSHELL));
  PetscCall(DMCreateMatrix(da,&(A_matfree)));
  PetscCall(MatShellSetOperation(A_matfree,MATOP_MULT,(void (*)(void))MyMatShellMult));
  PetscCall(MatShellSetOperation(A_matfree,MATOP_GET_DIAGONAL,(void (*)(void))preconditioner_callback));
  PetscCall(MatShellSetContext(A_matfree,ctx));
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
        b_tmp[x][y] = rho*(-2.5e-1F*pow(-u[t0][x + 1][y + 2]/h_x + u[t0][x + 3][y + 2]/h_x, 2) - 2.5e-1F*pow(-v[t0][x + 2][y + 1]/h_y + v[t0][x + 2][y + 3]/h_y, 2) + 1.0e+3F*(-5.0e-1F*u[t0][x + 1][y + 2]/h_x + 5.0e-1F*u[t0][x + 3][y + 2]/h_x) - 2.0F*(-5.0e-1F*v[t0][x + 1][y + 2]/h_x + 5.0e-1F*v[t0][x + 3][y + 2]/h_x)*(-5.0e-1F*u[t0][x + 2][y + 1]/h_y + 5.0e-1F*u[t0][x + 2][y + 3]/h_y) + 1.0e+3F*(-5.0e-1F*v[t0][x + 2][y + 1]/h_y + 5.0e-1F*v[t0][x + 2][y + 3]/h_y));
      }
    }
    STOP(section1,timers)

    PetscCall(DMDAVecRestoreArray(da,b,&b_tmp));
    PetscCall(KSPCreate(PETSC_COMM_SELF,&(ksp)));
    PetscCall(KSPSetOperators(ksp,A_matfree,A_matfree));
    PetscCall(KSPSetTolerances(ksp,1.00000000000000e-7F,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
    PetscCall(KSPSetType(ksp,KSPGMRES));
    PetscCall(KSPGetPC(ksp,&(pc)));
    PetscCall(PCSetType(pc,PCJACOBI));
    PetscCall(PCJacobiSetType(pc,PC_JACOBI_DIAGONAL));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp,b,x));
    PetscCall(KSPGetConvergedReason(ksp,&(reason)));
    PetscPrintf(PETSC_COMM_WORLD, "Convergence reason: %s",                       KSPConvergedReasons[reason]);
    PetscCall(DMDAVecGetArray(da,x,&xvec_tmp));
    for (int y = y_m; y <= y_M; y += 1)
    {
      for (int x = x_m; x <= x_M; x += 1)
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
        u[t1][x + 2][y + 2] = dt*(nu*(u[t0][x + 1][y + 2]/pow(h_x, 2) - 2.0F*u[t0][x + 2][y + 2]/pow(h_x, 2) + u[t0][x + 3][y + 2]/pow(h_x, 2) + u[t0][x + 2][y + 1]/pow(h_y, 2) - 2.0F*u[t0][x + 2][y + 2]/pow(h_y, 2) + u[t0][x + 2][y + 3]/pow(h_y, 2)) - (-5.0e-1F*u[t0][x + 1][y + 2]/h_x + 5.0e-1F*u[t0][x + 3][y + 2]/h_x)*u[t0][x + 2][y + 2] - (-5.0e-1F*u[t0][x + 2][y + 1]/h_y + 5.0e-1F*u[t0][x + 2][y + 3]/h_y)*v[t0][x + 2][y + 2] + u[t0][x + 2][y + 2]/dt - (-5.0e-1F*pn[x + 1][y + 2]/h_x + 5.0e-1F*pn[x + 3][y + 2]/h_x)/rho);

        v[t1][x + 2][y + 2] = dt*(nu*(v[t0][x + 1][y + 2]/pow(h_x, 2) - 2.0F*v[t0][x + 2][y + 2]/pow(h_x, 2) + v[t0][x + 3][y + 2]/pow(h_x, 2) + v[t0][x + 2][y + 1]/pow(h_y, 2) - 2.0F*v[t0][x + 2][y + 2]/pow(h_y, 2) + v[t0][x + 2][y + 3]/pow(h_y, 2)) - (-5.0e-1F*v[t0][x + 1][y + 2]/h_x + 5.0e-1F*v[t0][x + 3][y + 2]/h_x)*u[t0][x + 2][y + 2] - (-5.0e-1F*v[t0][x + 2][y + 1]/h_y + 5.0e-1F*v[t0][x + 2][y + 3]/h_y)*v[t0][x + 2][y + 2] + v[t0][x + 2][y + 2]/dt - (-5.0e-1F*pn[x + 2][y + 1]/h_y + 5.0e-1F*pn[x + 2][y + 3]/h_y)/rho);
      }

      u[t1][x + 2][52] = 1.00000000000000F;
    }
    STOP(section2,timers)

    START(section3)
    for (int y = y_m; y <= y_M; y += 1)
    {
      u[t1][2][y + 2] = 0.0F;

      u[t1][52][y + 2] = 0.0F;
    }
    STOP(section3,timers)

    START(section4)
    for (int x = x_m; x <= x_M; x += 1)
    {
      u[t1][x + 2][2] = 0.0F;
    }
    STOP(section4,timers)

    START(section5)
    for (int y = y_m; y <= y_M; y += 1)
    {
      v[t1][2][y + 2] = 0.0F;

      v[t1][52][y + 2] = 0.0F;
    }
    STOP(section5,timers)

    START(section6)
    for (int x = x_m; x <= x_M; x += 1)
    {
      v[t1][x + 2][52] = 0.0F;

      v[t1][x + 2][2] = 0.0F;
    }
    STOP(section6,timers)
  }

  return 0;
}

PetscErrorCode MyMatShellMult(Mat A_matfree, Vec xvec, Vec yvec)
{
  DM da;
  Vec local_xvec;

  PetscScalar**restrict xvec_tmp;
  PetscScalar**restrict yvec_tmp;

  PetscFunctionBegin;
  struct MatContext * ctx;
  PetscCall(MatShellGetContext(A_matfree,&(ctx)));
  PetscCall(MatGetDM(A_matfree,&(da)));
  PetscCall(DMGetLocalVector(da,&(local_xvec)));
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
  yvec_tmp[0][0]= xvec_tmp[0][0];
  PetscCall(DMDAVecRestoreArrayRead(da,local_xvec,&xvec_tmp));
  PetscCall(DMDAVecRestoreArray(da,yvec,&yvec_tmp));
  PetscCall(DMRestoreLocalVector(da,&(local_xvec)));
  PetscFunctionReturn(0);
}

PetscErrorCode preconditioner_callback(Mat A_matfree, Vec yvec)
{
  DM da;

  PetscScalar**restrict yvec_tmp;

  PetscFunctionBegin;
  struct MatContext * ctx;
  PetscCall(MatShellGetContext(A_matfree,&(ctx)));
  PetscCall(MatGetDM(A_matfree,&(da)));
  PetscCall(DMDAVecGetArray(da,yvec,&yvec_tmp));
  for (int x = ctx->x_m; x <= ctx->x_M; x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      yvec_tmp[x][y] = -2.0F*pow(ctx->h_x, -2) - 2.0F*pow(ctx->h_y, -2);
    }
  }
  yvec_tmp[0][0]=1.;
  PetscCall(DMDAVecRestoreArray(da,yvec,&yvec_tmp));
  PetscFunctionReturn(0);
}