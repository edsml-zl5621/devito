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
  float dt;
  float h_x;
  float h_y;
  int time_M;
  int time_m;
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
  double section7;
} ;

PetscErrorCode preconditioner_callback_pn1(Mat A_matfree_pn1, Vec yvec_pn1);
PetscErrorCode MyMatShellMult_pn1(Mat A_matfree_pn1, Vec xvec_pn1, Vec yvec_pn1);

int Kernel(const float nu, struct dataobj *restrict pn1_vec, const float rho, struct dataobj *restrict u1_vec, struct dataobj *restrict v1_vec, const float dt, const float h_x, const float h_y, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct MatContext * ctx, struct profiler * timers)
{
  Mat A_matfree_pn1;
  Vec b_pn1;
  DM da_pn1;
  KSP ksp_pn1;
  PC pc_pn1;
  PetscMPIInt size;
  Vec x_pn1;

  PetscScalar** b_tmp_pn1;
  PetscScalar** sol_tmp_pn1;

  float (*restrict pn1)[pn1_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[pn1_vec->size[1]]) pn1_vec->data;
  float (*restrict u1)[u1_vec->size[1]][u1_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[u1_vec->size[1]][u1_vec->size[2]]) u1_vec->data;
  float (*restrict v1)[v1_vec->size[1]][v1_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[v1_vec->size[1]][v1_vec->size[2]]) v1_vec->data;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(NULL,NULL,NULL,NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&(size)));

  PetscCall(DMDACreate2d(PETSC_COMM_SELF,DM_BOUNDARY_MIRROR,DM_BOUNDARY_MIRROR,DMDA_STENCIL_STAR,51,51,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&(da_pn1)));
  PetscCall(DMSetFromOptions(da_pn1));
  PetscCall(DMSetUp(da_pn1));
  PetscCall(DMSetMatType(da_pn1,MATSHELL));
  PetscCall(DMCreateMatrix(da_pn1,&(A_matfree_pn1)));
  PetscCall(MatShellSetContext(A_matfree_pn1,ctx));
  PetscCall(DMCreateGlobalVector(da_pn1,&(x_pn1)));
  PetscCall(DMCreateGlobalVector(da_pn1,&(b_pn1)));
  PetscCall(KSPCreate(PETSC_COMM_SELF,&(ksp_pn1)));
  PetscCall(KSPSetOperators(ksp_pn1,A_matfree_pn1,A_matfree_pn1));
  PetscCall(KSPSetTolerances(ksp_pn1,1.00000000000000e-7F,PETSC_DEFAULT,PETSC_DEFAULT,2));
  PetscCall(KSPSetType(ksp_pn1,KSPGMRES));
  PetscCall(KSPGetPC(ksp_pn1,&(pc_pn1)));
  PetscCall(PCSetType(pc_pn1,PCJACOBI));
  PetscCall(PCJacobiSetType(pc_pn1,PC_JACOBI_DIAGONAL));
  PetscCall(KSPSetFromOptions(ksp_pn1));
  PetscCall(MatShellSetOperation(A_matfree_pn1,MATOP_GET_DIAGONAL,(void (*)(void))preconditioner_callback_pn1));
  PetscCall(MatShellSetOperation(A_matfree_pn1,MATOP_MULT,(void (*)(void))MyMatShellMult_pn1));
  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {
    START(section0)
    STOP(section0,timers)

    START(section1)
    STOP(section1,timers)

    START(section2)
    STOP(section2,timers)

    START(section3)
    PetscCall(DMDAVecGetArray(da_pn1,b_pn1,&b_tmp_pn1));
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        b_tmp_pn1[x][y] = rho*(-2.5e-1F*pow(-u1[t0][x + 1][y + 2]/h_x + u1[t0][x + 3][y + 2]/h_x, 2) - 2.5e-1F*pow(-v1[t0][x + 2][y + 1]/h_y + v1[t0][x + 2][y + 3]/h_y, 2) + 1.0e+3F*(-5.0e-1F*u1[t0][x + 1][y + 2]/h_x + 5.0e-1F*u1[t0][x + 3][y + 2]/h_x) - 2.0F*(-5.0e-1F*v1[t0][x + 1][y + 2]/h_x + 5.0e-1F*v1[t0][x + 3][y + 2]/h_x)*(-5.0e-1F*u1[t0][x + 2][y + 1]/h_y + 5.0e-1F*u1[t0][x + 2][y + 3]/h_y) + 1.0e+3F*(-5.0e-1F*v1[t0][x + 2][y + 1]/h_y + 5.0e-1F*v1[t0][x + 2][y + 3]/h_y));
      }
    }
    PetscCall(DMDAVecRestoreArray(da_pn1,b_pn1,&b_tmp_pn1));
    PetscCall(KSPSolve(ksp_pn1,b_pn1,x_pn1));
    PetscCall(DMDAVecGetArray(da_pn1,x_pn1,&sol_tmp_pn1));
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        pn1[x + 2][y + 2] = sol_tmp_pn1[x][y];
      }
    }
    PetscCall(DMDAVecRestoreArray(da_pn1,x_pn1,&sol_tmp_pn1));
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {

        u1[t1][x + 2][y + 2] = dt*(nu*(u1[t0][x + 1][y + 2]/pow(h_x, 2) - 2.0F*u1[t0][x + 2][y + 2]/pow(h_x, 2) + u1[t0][x + 3][y + 2]/pow(h_x, 2) + u1[t0][x + 2][y + 1]/pow(h_y, 2) - 2.0F*u1[t0][x + 2][y + 2]/pow(h_y, 2) + u1[t0][x + 2][y + 3]/pow(h_y, 2)) - (-5.0e-1F*u1[t0][x + 1][y + 2]/h_x + 5.0e-1F*u1[t0][x + 3][y + 2]/h_x)*u1[t0][x + 2][y + 2] - (-5.0e-1F*u1[t0][x + 2][y + 1]/h_y + 5.0e-1F*u1[t0][x + 2][y + 3]/h_y)*v1[t0][x + 2][y + 2] + u1[t0][x + 2][y + 2]/dt - (-5.0e-1F*pn1[x + 1][y + 2]/h_x + 5.0e-1F*pn1[x + 3][y + 2]/h_x)/rho);

        v1[t1][x + 2][y + 2] = dt*(nu*(v1[t0][x + 1][y + 2]/pow(h_x, 2) - 2.0F*v1[t0][x + 2][y + 2]/pow(h_x, 2) + v1[t0][x + 3][y + 2]/pow(h_x, 2) + v1[t0][x + 2][y + 1]/pow(h_y, 2) - 2.0F*v1[t0][x + 2][y + 2]/pow(h_y, 2) + v1[t0][x + 2][y + 3]/pow(h_y, 2)) - (-5.0e-1F*v1[t0][x + 1][y + 2]/h_x + 5.0e-1F*v1[t0][x + 3][y + 2]/h_x)*u1[t0][x + 2][y + 2] - (-5.0e-1F*v1[t0][x + 2][y + 1]/h_y + 5.0e-1F*v1[t0][x + 2][y + 3]/h_y)*v1[t0][x + 2][y + 2] + v1[t0][x + 2][y + 2]/dt - (-5.0e-1F*pn1[x + 2][y + 1]/h_y + 5.0e-1F*pn1[x + 2][y + 3]/h_y)/rho);
      }

      u1[t1][x + 2][52] = 1.00000000000000F;
    }
    STOP(section3,timers)

    START(section4)
    for (int y = y_m; y <= y_M; y += 1)
    {
      u1[t1][2][y + 2] = 0.0F;

      u1[t1][52][y + 2] = 0.0F;
    }
    STOP(section4,timers)

    START(section5)
    for (int x = x_m; x <= x_M; x += 1)
    {
      u1[t1][x + 2][2] = 0.0F;
    }
    STOP(section5,timers)

    START(section6)
    for (int y = y_m; y <= y_M; y += 1)
    {
      v1[t1][2][y + 2] = 0.0F;

      v1[t1][52][y + 2] = 0.0F;
    }
    STOP(section6,timers)

    START(section7)
    for (int x = x_m; x <= x_M; x += 1)
    {
      v1[t1][x + 2][52] = 0.0F;

      v1[t1][x + 2][2] = 0.0F;
    }
    STOP(section7,timers)
  }

  return 0;
}

PetscErrorCode preconditioner_callback_pn1(Mat A_matfree_pn1, Vec yvec_pn1)
{
  DM da_pn1;

  PetscScalar** y_pre_pn1;

  PetscFunctionBegin;
  struct MatContext * ctx;
  PetscCall(MatShellGetContext(A_matfree_pn1,&(ctx)));
  PetscCall(MatGetDM(A_matfree_pn1,&(da_pn1)));
  PetscCall(DMDAVecGetArray(da_pn1,yvec_pn1,&y_pre_pn1));
  for (int x = ctx->x_m; x <= ctx->x_M; x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      y_pre_pn1[x][y] = -2.0F*pow(ctx->h_x, -2) - 2.0F*pow(ctx->h_y, -2);
    }
  }
  y_pre_pn1[0][0]=1.;
  PetscCall(DMDAVecRestoreArray(da_pn1,yvec_pn1,&y_pre_pn1));
  PetscFunctionReturn(0);
}

PetscErrorCode MyMatShellMult_pn1(Mat A_matfree_pn1, Vec xvec_pn1, Vec yvec_pn1)
{
  DM da_pn1;
  Vec local_xvec_pn1;

  PetscScalar** xvec_tmp_pn1;
  PetscScalar** y_matvec_pn1;

  PetscFunctionBegin;
  struct MatContext * ctx;
  PetscCall(MatShellGetContext(A_matfree_pn1,&(ctx)));
  PetscCall(MatGetDM(A_matfree_pn1,&(da_pn1)));
  PetscCall(DMGetLocalVector(da_pn1,&(local_xvec_pn1)));
  PetscCall(DMGlobalToLocalBegin(da_pn1,xvec_pn1,INSERT_VALUES,local_xvec_pn1));
  PetscCall(DMGlobalToLocalEnd(da_pn1,xvec_pn1,INSERT_VALUES,local_xvec_pn1));
  PetscCall(DMDAVecGetArrayRead(da_pn1,local_xvec_pn1,&xvec_tmp_pn1));
  PetscCall(DMDAVecGetArray(da_pn1,yvec_pn1,&y_matvec_pn1));
  for (int x = ctx->x_m; x <= ctx->x_M; x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      y_matvec_pn1[x][y] = -2.0F*pow(ctx->h_x, -2)*xvec_tmp_pn1[x][y] + pow(ctx->h_x, -2)*xvec_tmp_pn1[x - 1][y] + pow(ctx->h_x, -2)*xvec_tmp_pn1[x + 1][y] - 2.0F*pow(ctx->h_y, -2)*xvec_tmp_pn1[x][y] + pow(ctx->h_y, -2)*xvec_tmp_pn1[x][y - 1] + pow(ctx->h_y, -2)*xvec_tmp_pn1[x][y + 1];
    }
  }
  y_matvec_pn1[0][0]= xvec_tmp_pn1[0][0];
  PetscCall(DMDAVecRestoreArrayRead(da_pn1,local_xvec_pn1,&xvec_tmp_pn1));
  PetscCall(DMDAVecRestoreArray(da_pn1,yvec_pn1,&y_matvec_pn1));
  PetscCall(DMRestoreLocalVector(da_pn1,&(local_xvec_pn1)));
  PetscFunctionReturn(0);
}