#define _POSIX_C_SOURCE 200809L
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "petscksp.h"
#include "petscdmda.h"

struct MatContext
{
  float h_x;
  float h_y;
  int i0x_ltkn;
  int i0x_rtkn;
  int i0y_ltkn;
  int i0y_rtkn;
  int x_M;
  int x_m;
  int y_M;
  int y_m;
  int x_size;
  int y_size;
  int s_o;
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

int Kernel(struct dataobj *restrict p_vec, struct dataobj *restrict pn_vec, struct profiler * timers, struct MatContext * ctx)
{
  float (*restrict p)[p_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[p_vec->size[1]]) p_vec->data;
  float (*restrict pn)[pn_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[pn_vec->size[1]]) pn_vec->data;

  Vec x;
  Vec b;
  Mat A_matfree;
  KSP ksp;
  PC pc;
  DM da;
  PetscMPIInt size;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(NULL,NULL,NULL,NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(DMDACreate2d(PETSC_COMM_SELF,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,ctx->x_size,ctx->y_size,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateGlobalVector(da,&x));
  PetscCall(DMCreateGlobalVector(da,&b));
  PetscCall(DMSetMatType(da,MATSHELL));
  PetscCall(DMCreateMatrix(da,&A_matfree));
  PetscCall(MatShellSetOperation(A_matfree,MATOP_MULT,(void (*)(void))MyMatShellMult));
  PetscCall(MatShellSetContext(A_matfree,ctx));
  for (int x = ctx->x_m; x <= ctx->x_M; x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      PetscInt position = x*ctx->x_size + y;
      PetscScalar b_val = pn[x + ctx->s_o][y + ctx->s_o];
      PetscCall(VecSetValue(b,position,b_val,ADD_VALUES));
    }
  }
  PetscCall(KSPCreate(PETSC_COMM_SELF,&ksp));
  PetscCall(KSPSetOperators(ksp,A_matfree,A_matfree));
  PetscCall(KSPSetTolerances(ksp,1.00000000000000e-8F,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  PetscCall(KSPSetType(ksp,KSPGMRES));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCNONE));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,b,x));
  PetscScalar ** pn_tmp;
  PetscCall(VecGetArray2d(x,ctx->x_size,ctx->y_size,0,0,&pn_tmp));
  for (int x = ctx->x_m; x <= ctx->x_M; x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      pn[x + ctx->s_o][y + ctx->s_o] = pn_tmp[x][y];
    }
  }
  PetscCall(VecRestoreArray2d(x,ctx->x_size,ctx->y_size,0,0,&pn_tmp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A_matfree));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());

  return 0;
}

PetscErrorCode MyMatShellMult(Mat A_matfree, Vec xvec, Vec yvec)
{
  PetscFunctionBegin;
  struct MatContext * ctx;
  PetscCall(MatShellGetContext(A_matfree,&ctx));
  PetscScalar ** p;
  PetscScalar ** pn;
  DM da;
  Vec local_xvec;
  Vec local_yvec;
  PetscCall(MatGetDM(A_matfree,&da));
  PetscCall(DMGetLocalVector(da,&local_xvec));
  PetscCall(DMGetLocalVector(da,&local_yvec));
  PetscCall(DMGlobalToLocalBegin(da,xvec,INSERT_VALUES,local_xvec));
  PetscCall(DMGlobalToLocalEnd(da,xvec,INSERT_VALUES,local_xvec));
  PetscCall(DMDAVecGetArrayRead(da,local_xvec,&pn));
  PetscCall(DMDAVecGetArray(da,local_yvec,&p));
  //  for parallel code will have to use DMDAGETCORNERS

  for (int i0x = ctx->i0x_ltkn + ctx->x_m; i0x <= -ctx->i0x_rtkn + ctx->x_M; i0x += 1)
  {
    for (int i0y = ctx->i0y_ltkn + ctx->y_m; i0y <= -ctx->i0y_rtkn + ctx->y_M; i0y += 1)
    {
      p[i0x][i0y] = -2.0F*pow(ctx->h_x, -2)*pn[i0x][i0y] + pow(ctx->h_x, -2)*pn[i0x - 1][i0y] + pow(ctx->h_x, -2)*pn[i0x + 1][i0y] - 2.0F*pow(ctx->h_y, -2)*pn[i0x][i0y] + pow(ctx->h_y, -2)*pn[i0x][i0y - 1] + pow(ctx->h_y, -2)*pn[i0x][i0y + 1];
    }
  }
  for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
  {
    p[0][y] = pn[0][y];

    p[4][y] = pn[4][y];
  }
  for (int x = ctx->x_m; x <= ctx->x_M; x += 1)
  {
    p[x][0] = pn[x][0];

    p[x][4] = pn[x][4];
  }
  PetscCall(DMDAVecRestoreArrayRead(da,local_xvec,&pn));
  PetscCall(DMDAVecRestoreArray(da,local_yvec,&p));
  PetscCall(DMLocalToGlobalBegin(da,local_yvec,INSERT_VALUES,yvec));
  PetscCall(DMLocalToGlobalEnd(da,local_yvec,INSERT_VALUES,yvec));
  PetscCall(DMRestoreLocalVector(da,&local_xvec));
  PetscCall(DMRestoreLocalVector(da,&local_yvec));
  PetscFunctionReturn(0);
}