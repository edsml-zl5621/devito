#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "petscksp.h"
#include "petscdmda.h"
#include "petscsnes.h"

struct MatContext
{
  double h_x;
  double h_y;
  int i0x_ltkn;
  int i0x_rtkn;
  int i0y_ltkn;
  int i0y_rtkn;
  int i1y_rtkn;
  int i2y_ltkn;
  int i3x_ltkn;
  int i4x_rtkn;
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
} ;

PetscErrorCode preconditioner_callback_pn(Mat A_matfree_pn, Vec yvec_pn);
PetscErrorCode MyMatShellMult_pn(Mat A_matfree_pn, Vec xvec_pn, Vec yvec_pn);
PetscErrorCode FormFunction(SNES snes, Vec xvec_pn, Vec yvec_pn, void *null);

int Kernel(struct dataobj *restrict pn_vec, struct dataobj *restrict rhs_vec, const int x_M, const int x_m, const int y_M, const int y_m, struct MatContext * ctx, struct profiler * timers)
{
  Mat J;
  Vec b_pn;
  Vec blocal_pn;
  DM da_pn;
  KSP ksp_pn;
  SNES snes_pn;
  PC pc_pn;
  PetscMPIInt size;
  Vec x_pn;
  Vec xlocal_pn;
  PetscScalar* b_one_dim;

  double (*restrict rhs)[rhs_vec->size[1]] __attribute__ ((aligned (64))) = (double (*)[rhs_vec->size[1]]) rhs_vec->data;

  PetscFunctionBeginUser;
  // Temporary initialisation and SERIAL only example
  PetscCall(PetscInitialize(NULL,NULL,NULL,NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_SELF,&(size)));

  PetscCall(SNESCreate(PETSC_COMM_SELF, &snes_pn));
  PetscCall(DMDACreate2d(PETSC_COMM_SELF,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,13,13,PETSC_DECIDE,PETSC_DECIDE,1,2,NULL,NULL,&(da_pn)));
  PetscCall(DMSetUp(da_pn));
  PetscCall(SNESSetDM(snes_pn, da_pn));
  PetscCall(DMSetApplicationContext(da_pn, ctx));
  PetscCall(DMSNESSetFunction(da_pn, FormFunction, NULL));
  PetscCall(DMSetMatType(da_pn, MATSHELL));
  PetscCall(DMCreateMatrix(da_pn,&(J)));

  PetscCall(MatShellSetOperation(J,MATOP_MULT,(void (*)(void))MyMatShellMult_pn));
  PetscCall(SNESSetJacobian(snes_pn, J, J, MatMFFDComputeJacobian, NULL));
  PetscCall(SNESSetType(snes_pn, SNESKSPONLY));

  PetscCall(DMCreateGlobalVector(da_pn,&(x_pn)));
  PetscCall(DMCreateLocalVector(da_pn,&(xlocal_pn)));
  PetscCall(DMCreateGlobalVector(da_pn,&(b_pn)));
  PetscCall(DMCreateLocalVector(da_pn,&(blocal_pn)));
  PetscCall(VecReplaceArray(xlocal_pn,pn_vec->data));

  PetscCall(SNESGetKSP(snes_pn, &ksp_pn));
  PetscCall(KSPSetTolerances(ksp_pn,1.00000000000000e-7F,PETSC_DEFAULT,PETSC_DEFAULT,10000));
  PetscCall(KSPSetType(ksp_pn,KSPGMRES));
  PetscCall(KSPGetPC(ksp_pn,&(pc_pn)));
  PetscCall(PCSetType(pc_pn,PCJACOBI));
  PetscCall(PCJacobiSetType(pc_pn,PC_JACOBI_DIAGONAL));
  PetscCall(KSPSetFromOptions(ksp_pn));
  PetscCall(MatShellSetOperation(J,MATOP_GET_DIAGONAL,(void (*)(void))preconditioner_callback_pn));

  // Setup constant part - rhs - in this example it is just zero everywhere.
  PetscCall(VecGetArray(blocal_pn, &b_one_dim));
  PetscScalar (* b_tmp_pn)[17] = (PetscScalar (*)[17])b_one_dim;
  for (int x = x_m; x <= x_M; x += 1)
  {
    for (int y = y_m; y <= y_M; y += 1)
    {
      b_tmp_pn[x + 2][y + 2] = rhs[x + 2][y + 2];
    }
  }
  // boundary loops - need to set b to zero on essential boundary nodes
  for (int x = ctx->x_m; x <= ctx->x_M; x += 1)
  {
    for (int i1y = 1 - ctx->i1y_rtkn + ctx->y_M; i1y <= ctx->y_M; i1y += 1)
    {
      b_tmp_pn[x + 2][i1y + 2] = 0.;
    }
    for (int i2y = ctx->y_m; i2y <= -1 + ctx->i2y_ltkn + ctx->y_m; i2y += 1)
    {
      b_tmp_pn[x + 2][i2y + 2] = 0.;
    }
  }
  for (int i3x = ctx->x_m; i3x <= -1 + ctx->i3x_ltkn + ctx->x_m; i3x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      b_tmp_pn[i3x + 2][y + 2] = 0.;
    }
  }
  for (int i4x = 1 - ctx->i4x_rtkn + ctx->x_M; i4x <= ctx->x_M; i4x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      b_tmp_pn[i4x + 2][y + 2] = 0.;
    }
  }

  PetscCall(VecRestoreArray(blocal_pn,&b_one_dim));
  // Set up initial guess using user initialised field - satisfies BCs
  // This works because I used VecReplaceArray above.
  PetscCall(DMLocalToGlobal(da_pn,xlocal_pn,INSERT_VALUES,x_pn));
  PetscCall(DMLocalToGlobal(da_pn,blocal_pn,INSERT_VALUES,b_pn));
  PetscCall(SNESSolve(snes_pn,b_pn,x_pn));
  PetscCall(DMGlobalToLocal(da_pn,x_pn,INSERT_VALUES,xlocal_pn));

  PetscCall(VecDestroy(&(x_pn)));
  PetscCall(VecDestroy(&(b_pn)));
  PetscCall(MatDestroy(&(J)));
  PetscCall(SNESDestroy(&(snes_pn)));
  PetscCall(DMDestroy(&(da_pn)));

  return 0;
}

PetscErrorCode preconditioner_callback_pn(Mat A_matfree_pn, Vec yvec_pn)
{
  DM da_pn;
  Vec local_yvec_pn;
  struct MatContext * ctx;
  PetscScalar* y_one_dim;

  PetscFunctionBegin;

  PetscCall(MatGetDM(A_matfree_pn,&(da_pn)));
  PetscCall(DMGetApplicationContext(da_pn, &ctx));

  PetscCall(DMGetLocalVector(da_pn,&(local_yvec_pn)));
  PetscCall(VecGetArray(local_yvec_pn,&y_one_dim));
  PetscScalar (* y_pre_pn)[17] = (PetscScalar (*)[17])y_one_dim;

  // Interior
  for (int i0x = ctx->i0x_ltkn + ctx->x_m; i0x <= -ctx->i0x_rtkn + ctx->x_M; i0x += 1)
  {
    for (int i0y = ctx->i0y_ltkn + ctx->y_m; i0y <= -ctx->i0y_rtkn + ctx->y_M; i0y += 1)
    {
      y_pre_pn[i0x + 2][i0y + 2] = -2.0*pow(ctx->h_x, -2) - 2.0*pow(ctx->h_y, -2);
    }
  }
  // Boundary loops 
  // These loops are the result of using subdomains in Devito.
  for (int x = ctx->x_m; x <= ctx->x_M; x += 1)
  {
    for (int i1y = 1 - ctx->i1y_rtkn + ctx->y_M; i1y <= ctx->y_M; i1y += 1)
    {
      y_pre_pn[x + 2][i1y + 2] = -2.0*pow(ctx->h_x, -2) - 2.0*pow(ctx->h_y, -2);
    }
    for (int i2y = ctx->y_m; i2y <= -1 + ctx->i2y_ltkn + ctx->y_m; i2y += 1)
    {
      y_pre_pn[x + 2][i2y + 2] = -2.0*pow(ctx->h_x, -2) - 2.0*pow(ctx->h_y, -2);
    }
  }
  for (int i3x = ctx->x_m; i3x <= -1 + ctx->i3x_ltkn + ctx->x_m; i3x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      y_pre_pn[i3x + 2][y + 2] = -2.0*pow(ctx->h_x, -2) - 2.0*pow(ctx->h_y, -2);
    }
  }
  for (int i4x = 1 - ctx->i4x_rtkn + ctx->x_M; i4x <= ctx->x_M; i4x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      y_pre_pn[i4x + 2][y + 2] = -2.0*pow(ctx->h_x, -2) - 2.0*pow(ctx->h_y, -2);
    }
  }

  PetscCall(VecRestoreArray(local_yvec_pn, &y_one_dim));
  PetscCall(DMLocalToGlobalBegin(da_pn,local_yvec_pn,INSERT_VALUES,yvec_pn));
  PetscCall(DMLocalToGlobalEnd(da_pn,local_yvec_pn,INSERT_VALUES,yvec_pn));
  PetscFunctionReturn(0);
}


PetscErrorCode FormFunction(SNES snes, Vec xvec_pn, Vec yvec_pn, void *null)
{
  DM da_pn;
  Vec local_xvec_pn, local_yvec_pn;
  struct MatContext * ctx;
  const PetscScalar* x_one_dim;
  PetscScalar* y_one_dim;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &da_pn));
  PetscCall(DMGetApplicationContext(da_pn, &ctx));

  PetscCall(DMGetLocalVector(da_pn,&(local_xvec_pn)));
  PetscCall(DMGlobalToLocalBegin(da_pn,xvec_pn,INSERT_VALUES,local_xvec_pn));
  PetscCall(DMGlobalToLocalEnd(da_pn,xvec_pn,INSERT_VALUES,local_xvec_pn));

  PetscCall(DMGetLocalVector(da_pn,&(local_yvec_pn)));

  PetscCall(VecGetArrayRead(local_xvec_pn,&x_one_dim));
  PetscCall(VecGetArray(local_yvec_pn,&y_one_dim));

  PetscScalar (* xvec_tmp_pn)[17] = (PetscScalar (*)[17])x_one_dim;
  PetscScalar (* y_matvec_pn)[17] = (PetscScalar (*)[17])y_one_dim;

  // Interior
  for (int i0x = ctx->i0x_ltkn + ctx->x_m; i0x <= -ctx->i0x_rtkn + ctx->x_M; i0x += 1)
  {
    for (int i0y = ctx->i0y_ltkn + ctx->y_m; i0y <= -ctx->i0y_rtkn + ctx->y_M; i0y += 1)
    {
      y_matvec_pn[i0x + 2][i0y + 2] = -2.0*pow(ctx->h_x, -2)*xvec_tmp_pn[i0x + 2][i0y + 2] + pow(ctx->h_x, -2)*xvec_tmp_pn[i0x + 1][i0y + 2] + pow(ctx->h_x, -2)*xvec_tmp_pn[i0x + 3][i0y + 2] - 2.0*pow(ctx->h_y, -2)*xvec_tmp_pn[i0x + 2][i0y + 2] + pow(ctx->h_y, -2)*xvec_tmp_pn[i0x + 2][i0y + 1] + pow(ctx->h_y, -2)*xvec_tmp_pn[i0x + 2][i0y + 3];
    }
  }
  // Boundary loops - F is set to zero.
  // These loops are the result of using subdomains in Devito.
  for (int x = ctx->x_m; x <= ctx->x_M; x += 1)
  {
    for (int i1y = 1 - ctx->i1y_rtkn + ctx->y_M; i1y <= ctx->y_M; i1y += 1)
    {
      y_matvec_pn[x + 2][i1y + 2] = 0.;
    }
    for (int i2y = ctx->y_m; i2y <= -1 + ctx->i2y_ltkn + ctx->y_m; i2y += 1)
    {
      y_matvec_pn[x + 2][i2y + 2] = 0.;
    }
  }
  for (int i3x = ctx->x_m; i3x <= -1 + ctx->i3x_ltkn + ctx->x_m; i3x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      y_matvec_pn[i3x + 2][y + 2] = 0.;
    }
  }
  for (int i4x = 1 - ctx->i4x_rtkn + ctx->x_M; i4x <= ctx->x_M; i4x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      y_matvec_pn[i4x + 2][y + 2] = 0.;
    }
  }

  PetscCall(VecRestoreArrayRead(local_xvec_pn, &x_one_dim));
  PetscCall(VecRestoreArray(local_yvec_pn, &y_one_dim));
  PetscCall(DMLocalToGlobalBegin(da_pn,local_yvec_pn,INSERT_VALUES,yvec_pn));
  PetscCall(DMLocalToGlobalEnd(da_pn,local_yvec_pn,INSERT_VALUES,yvec_pn));
  PetscFunctionReturn(0);

}


PetscErrorCode MyMatShellMult_pn(Mat A_matfree_pn, Vec xvec_pn, Vec yvec_pn)
{
  DM da_pn;
  Vec local_xvec_pn;
  Vec local_yvec_pn;
  struct MatContext * ctx;
  const PetscScalar* x_one_dim;
  PetscScalar* y_one_dim;

  PetscFunctionBegin;

  PetscCall(MatGetDM(A_matfree_pn,&(da_pn)));
  PetscCall(DMGetApplicationContext(da_pn, &ctx));

  PetscCall(DMGetLocalVector(da_pn,&(local_xvec_pn)));
  PetscCall(DMGlobalToLocalBegin(da_pn,xvec_pn,INSERT_VALUES,local_xvec_pn));
  PetscCall(DMGlobalToLocalEnd(da_pn,xvec_pn,INSERT_VALUES,local_xvec_pn));

  PetscCall(DMGetLocalVector(da_pn,&(local_yvec_pn)));

  PetscCall(VecGetArrayRead(local_xvec_pn,&x_one_dim));
  PetscCall(VecGetArray(local_yvec_pn,&y_one_dim));

  PetscScalar (* xvec_tmp_pn)[17] = (PetscScalar (*)[17])x_one_dim;
  PetscScalar (* y_matvec_pn)[17] = (PetscScalar (*)[17])y_one_dim;


  // Interior
  for (int i0x = ctx->i0x_ltkn + ctx->x_m; i0x <= -ctx->i0x_rtkn + ctx->x_M; i0x += 1)
  {
    for (int i0y = ctx->i0y_ltkn + ctx->y_m; i0y <= -ctx->i0y_rtkn + ctx->y_M; i0y += 1)
    {
      y_matvec_pn[i0x + 2][i0y + 2] = -2.0*pow(ctx->h_x, -2)*xvec_tmp_pn[i0x + 2][i0y + 2] + pow(ctx->h_x, -2)*xvec_tmp_pn[i0x + 1][i0y + 2] + pow(ctx->h_x, -2)*xvec_tmp_pn[i0x + 3][i0y + 2] - 2.0*pow(ctx->h_y, -2)*xvec_tmp_pn[i0x + 2][i0y + 2] + pow(ctx->h_y, -2)*xvec_tmp_pn[i0x + 2][i0y + 1] + pow(ctx->h_y, -2)*xvec_tmp_pn[i0x + 2][i0y + 3];
    }
  }
  // Boundary loops - just diagonal component on row.
  // These loops are the result of using subdomains in Devito.
  for (int x = ctx->x_m; x <= ctx->x_M; x += 1)
  {
    for (int i1y = 1 - ctx->i1y_rtkn + ctx->y_M; i1y <= ctx->y_M; i1y += 1)
    {
      y_matvec_pn[x + 2][i1y + 2] = -2.0*pow(ctx->h_x, -2)*xvec_tmp_pn[x + 2][i1y + 2] - 2.0*pow(ctx->h_y, -2)*xvec_tmp_pn[x + 2][i1y + 2];
    }
    for (int i2y = ctx->y_m; i2y <= -1 + ctx->i2y_ltkn + ctx->y_m; i2y += 1)
    {
      y_matvec_pn[x + 2][i2y + 2] = -2.0*pow(ctx->h_x, -2)*xvec_tmp_pn[x + 2][i2y + 2] - 2.0*pow(ctx->h_y, -2)*xvec_tmp_pn[x + 2][i2y + 2];
    }
  }
  for (int i3x = ctx->x_m; i3x <= -1 + ctx->i3x_ltkn + ctx->x_m; i3x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      y_matvec_pn[i3x + 2][y + 2] = -2.0*pow(ctx->h_x, -2)*xvec_tmp_pn[i3x + 2][y + 2] - 2.0*pow(ctx->h_y, -2)*xvec_tmp_pn[i3x + 2][y + 2];
    }
  }
  for (int i4x = 1 - ctx->i4x_rtkn + ctx->x_M; i4x <= ctx->x_M; i4x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      y_matvec_pn[i4x + 2][y + 2] = -2.0*pow(ctx->h_x, -2)*xvec_tmp_pn[i4x + 2][y + 2] - 2.0*pow(ctx->h_y, -2)*xvec_tmp_pn[i4x + 2][y + 2];
    }
  }
  PetscCall(VecRestoreArrayRead(local_xvec_pn, &x_one_dim));
  PetscCall(VecRestoreArray(local_yvec_pn, &y_one_dim));
  PetscCall(DMLocalToGlobalBegin(da_pn,local_yvec_pn,INSERT_VALUES,yvec_pn));
  PetscCall(DMLocalToGlobalEnd(da_pn,local_yvec_pn,INSERT_VALUES,yvec_pn));
  PetscFunctionReturn(0);
}