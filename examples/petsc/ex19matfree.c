/* Portions of this code are under:
   Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/
static char help[] = "Nonlinear driven cavity with multigrid in 2d.\n \
  \n\
The 2D driven cavity problem is solved in a velocity-vorticity formulation.\n\
The flow can be driven with the lid or with buoyancy or both:\n\
  -lidvelocity &ltlid&gt, where &ltlid&gt = dimensionless velocity of lid\n\
  -grashof &ltgr&gt, where &ltgr&gt = dimensionless temperature gradent\n\
  -prandtl &ltpr&gt, where &ltpr&gt = dimensionless thermal/momentum diffusity ratio\n\
 -contours : draw contour plots of solution\n\n";
/* in HTML, '&lt' = '<' and '&gt' = '>' */

/*
      See src/ksp/ksp/tutorials/ex45.c
*/

/*F-----------------------------------------------------------------------

    We thank David E. Keyes for contributing the driven cavity discretization within this example code.

    This problem is modeled by the partial differential equation system

\begin{eqnarray}
        - \triangle U - \nabla_y \Omega & = & 0  \\
        - \triangle V + \nabla_x\Omega & = & 0  \\
        - \triangle \Omega + \nabla \cdot ([U*\Omega,V*\Omega]) - GR* \nabla_x T & = & 0  \\
        - \triangle T + PR* \nabla \cdot ([U*T,V*T]) & = & 0
\end{eqnarray}

    in the unit square, which is uniformly discretized in each of x and y in this simple encoding.

    No-slip, rigid-wall Dirichlet conditions are used for $ [U,V]$.
    Dirichlet conditions are used for Omega, based on the definition of
    vorticity: $ \Omega = - \nabla_y U + \nabla_x V$, where along each
    constant coordinate boundary, the tangential derivative is zero.
    Dirichlet conditions are used for T on the left and right walls,
    and insulation homogeneous Neumann conditions are used for T on
    the top and bottom walls.

    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a
    nonlinear system of equations.  Upwinding is used for the divergence
    (convective) terms and central for the gradient (source) terms.

    The Jacobian can be either
      * formed via finite differencing using coloring (the default), or
      * applied matrix-free via the option -snes_mf
        (for larger grid problems this variant may not converge
        without a preconditioner due to ill-conditioning).

  ------------------------------------------------------------------------F*/

/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#if defined(PETSC_APPLE_FRAMEWORK)
  #import <PETSc/petscsnes.h>
  #import <PETSc/petscdmda.h>
#else
  #include <petscsnes.h>
  #include <petscdm.h>
  #include <petscdmda.h>
#endif

/*
   User-defined routines and data structures
*/
typedef struct {
  PetscScalar u, v, omega, temp;
} Field;

typedef struct {
    Mat *submats;
    PetscInt n_submats;
    DM *sub_dms;
    IS *fields;
    SNES snes;
} JacobianContext;

typedef struct {
    IS rows;
    IS cols;
    DM dm;
    Vec omega, uvec, vvec, temp;
} SubmatrixContext;

PetscErrorCode FormFunctionLocal(DMDALocalInfo *, Field **, Field **, void *);

typedef struct {
  PetscReal lidvelocity, prandtl, grashof; /* physical parameters */
  PetscBool draw_contours;                 /* flag - 1 indicates drawing contours */
} AppCtx;

extern PetscErrorCode FormInitialGuess(AppCtx *, DM, Vec);


PetscErrorCode MyMatShellMult(Mat J, Vec X, Vec Y) {
  JacobianContext *jac_ctx;
  Vec sol;
  Vec uprev, vprev, omegaprev, tempprev;
  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J, &jac_ctx));

  PetscCall(SNESGetSolution(jac_ctx->snes, &sol));
  PetscCall(VecGetSubVector(sol, jac_ctx->fields[0], &uprev));
  PetscCall(VecGetSubVector(sol, jac_ctx->fields[1], &vprev));
  PetscCall(VecGetSubVector(sol, jac_ctx->fields[2], &omegaprev));
  PetscCall(VecGetSubVector(sol, jac_ctx->fields[3], &tempprev));

  // dfu/du
  SubmatrixContext *j00_ctx;
  Vec j00X, j00Y;
  PetscCall(MatShellGetContext(jac_ctx->submats[0], &j00_ctx));
  PetscCall(VecGetSubVector(X, j00_ctx->cols, &j00X));
  PetscCall(VecGetSubVector(Y, j00_ctx->rows, &j00Y));
  PetscCall(MatMult(jac_ctx->submats[0], j00X, j00Y));
  PetscCall(VecRestoreSubVector(X, j00_ctx->cols, &j00X));
  PetscCall(VecRestoreSubVector(Y, j00_ctx->rows, &j00Y));

  //dfu/domega
  SubmatrixContext *j02_ctx;
  Vec j02X, j02Y;
  PetscCall(MatShellGetContext(jac_ctx->submats[2], &j02_ctx));
  PetscCall(VecGetSubVector(X, j02_ctx->cols, &j02X));
  PetscCall(VecGetSubVector(Y, j02_ctx->rows, &j02Y));
  PetscCall(MatMult(jac_ctx->submats[2], j02X, j02Y));
  PetscCall(VecRestoreSubVector(X, j02_ctx->cols, &j02X));
  PetscCall(VecRestoreSubVector(Y, j02_ctx->rows, &j02Y));

  //dfv/dv
  SubmatrixContext *j11_ctx;
  Vec j11X, j11Y;
  PetscCall(MatShellGetContext(jac_ctx->submats[5], &j11_ctx));
  PetscCall(VecGetSubVector(X, j11_ctx->cols, &j11X));
  PetscCall(VecGetSubVector(Y, j11_ctx->rows, &j11Y));
  PetscCall(MatMult(jac_ctx->submats[5], j11X, j11Y));
  PetscCall(VecRestoreSubVector(X, j11_ctx->cols, &j11X));
  PetscCall(VecRestoreSubVector(Y, j11_ctx->rows, &j11Y));

  //dfv/domega
  SubmatrixContext *j12_ctx;
  Vec j12X, j12Y;
  PetscCall(MatShellGetContext(jac_ctx->submats[6], &j12_ctx));
  PetscCall(VecGetSubVector(X, j12_ctx->cols, &j12X));
  PetscCall(VecGetSubVector(Y, j12_ctx->rows, &j12Y));
  PetscCall(MatMult(jac_ctx->submats[6], j12X, j12Y));
  PetscCall(VecRestoreSubVector(X, j12_ctx->cols, &j12X));
  PetscCall(VecRestoreSubVector(Y, j12_ctx->rows, &j12Y));

  //dfomega/du
  SubmatrixContext *j20_ctx;
  Vec j20X, j20Y;
  PetscCall(MatShellGetContext(jac_ctx->submats[8], &j20_ctx));
  j20_ctx->omega = omegaprev;
  j20_ctx->uvec = uprev;
  PetscCall(MatShellSetContext(jac_ctx->submats[8], j20_ctx));
  PetscCall(VecGetSubVector(X, j20_ctx->cols, &j20X));
  PetscCall(VecGetSubVector(Y, j20_ctx->rows, &j20Y));
  PetscCall(MatMult(jac_ctx->submats[8], j20X, j20Y));
  PetscCall(VecRestoreSubVector(X, j20_ctx->cols, &j20X));
  PetscCall(VecRestoreSubVector(Y, j20_ctx->rows, &j20Y));

  //dfomega/dv
  SubmatrixContext *j21_ctx;
  Vec j21X, j21Y;
  PetscCall(MatShellGetContext(jac_ctx->submats[9], &j21_ctx));
  j21_ctx->omega = omegaprev;
  j21_ctx->vvec = vprev;
  PetscCall(MatShellSetContext(jac_ctx->submats[9], j21_ctx));
  PetscCall(VecGetSubVector(X, j21_ctx->cols, &j21X));
  PetscCall(VecGetSubVector(Y, j21_ctx->rows, &j21Y));
  PetscCall(MatMult(jac_ctx->submats[9], j21X, j21Y));
  PetscCall(VecRestoreSubVector(X, j21_ctx->cols, &j21X));
  PetscCall(VecRestoreSubVector(Y, j21_ctx->rows, &j21Y));

  //dfomega/domega
  SubmatrixContext *j22_ctx;
  Vec j22X, j22Y;
  PetscCall(MatShellGetContext(jac_ctx->submats[10], &j22_ctx));
  j22_ctx->uvec = uprev;
  j22_ctx->vvec = vprev;
  PetscCall(MatShellSetContext(jac_ctx->submats[10], j22_ctx));
  PetscCall(VecGetSubVector(X, j22_ctx->cols, &j22X));
  PetscCall(VecGetSubVector(Y, j22_ctx->rows, &j22Y));
  PetscCall(MatMult(jac_ctx->submats[10], j22X, j22Y));
  PetscCall(VecRestoreSubVector(X, j22_ctx->cols, &j22X));
  PetscCall(VecRestoreSubVector(Y, j22_ctx->rows, &j22Y));

  //dfomega/dT
  SubmatrixContext *j23_ctx;
  Vec j23X, j23Y;
  PetscCall(MatShellGetContext(jac_ctx->submats[11], &j23_ctx));
  PetscCall(VecGetSubVector(X, j23_ctx->cols, &j23X));
  PetscCall(VecGetSubVector(Y, j23_ctx->rows, &j23Y));
  PetscCall(MatMult(jac_ctx->submats[11], j23X, j23Y));
  PetscCall(VecRestoreSubVector(X, j23_ctx->cols, &j23X));
  PetscCall(VecRestoreSubVector(Y, j23_ctx->rows, &j23Y));

  //dfT/du
  SubmatrixContext *j30_ctx;
  Vec j30X, j30Y;
  PetscCall(MatShellGetContext(jac_ctx->submats[12], &j30_ctx));
  j30_ctx->temp = tempprev;
  PetscCall(MatShellSetContext(jac_ctx->submats[12], j30_ctx));
  PetscCall(VecGetSubVector(X, j30_ctx->cols, &j30X));
  PetscCall(VecGetSubVector(Y, j30_ctx->rows, &j30Y));
  PetscCall(MatMult(jac_ctx->submats[12], j30X, j30Y));
  PetscCall(VecRestoreSubVector(X, j30_ctx->cols, &j30X));
  PetscCall(VecRestoreSubVector(Y, j30_ctx->rows, &j30Y));

  //dfT/dv
  SubmatrixContext *j31_ctx;
  Vec j31X, j31Y;
  PetscCall(MatShellGetContext(jac_ctx->submats[13], &j31_ctx));  
  j31_ctx->temp = tempprev;
  j31_ctx->vvec = vprev;  
  PetscCall(MatShellSetContext(jac_ctx->submats[13], j31_ctx));
  PetscCall(VecGetSubVector(X, j31_ctx->cols, &j31X));
  PetscCall(VecGetSubVector(Y, j31_ctx->rows, &j31Y));
  PetscCall(MatMult(jac_ctx->submats[13], j31X, j31Y));
  PetscCall(VecRestoreSubVector(X, j31_ctx->cols, &j31X));
  PetscCall(VecRestoreSubVector(Y, j31_ctx->rows, &j31Y));

  //dfT/dT
  SubmatrixContext *j33_ctx;
  Vec j33X, j33Y;
  PetscCall(MatShellGetContext(jac_ctx->submats[15], &j33_ctx));
  j33_ctx->uvec = uprev;
  j33_ctx->vvec = vprev;
  PetscCall(MatShellSetContext(jac_ctx->submats[15], j33_ctx));
  PetscCall(VecGetSubVector(X, j33_ctx->cols, &j33X));
  PetscCall(VecGetSubVector(Y, j33_ctx->rows, &j33Y));
  PetscCall(MatMult(jac_ctx->submats[15], j33X, j33Y));
  PetscCall(VecRestoreSubVector(X, j33_ctx->cols, &j33X));
  PetscCall(VecRestoreSubVector(Y, j33_ctx->rows, &j33Y));

  PetscCall(VecRestoreSubVector(sol, jac_ctx->fields[0], &uprev));
  PetscCall(VecRestoreSubVector(sol, jac_ctx->fields[1], &vprev));
  PetscCall(VecRestoreSubVector(sol, jac_ctx->fields[2], &omegaprev));
  PetscCall(VecRestoreSubVector(sol, jac_ctx->fields[3], &tempprev));

  PetscFunctionReturn(0);
}



// dfu/du
static PetscErrorCode MySubMatMult_J00(Mat J00, Vec X, Vec Y)
{
  PetscScalar *x_array_vec, *y_array_vec;
  SubmatrixContext *sub_ctx;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hx, hy, dhx, dhy, hxdhy, hydhx;
  PetscScalar uxx, uyy;
  Vec xlocal, ylocal;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J00, &sub_ctx));

  da = sub_ctx->dm;

  PetscCall(DMDAGetLocalInfo(da, &info));

  dhx   = (PetscReal)(info.mx - 1);
  dhy   = (PetscReal)(info.my - 1);
  hx    = 1.0 / dhx;
  hy    = 1.0 / dhy;
  hxdhy = hx * dhy;
  hydhx = hy * dhx;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J00, &n, NULL));

  PetscCall(DMGetLocalVector(da, &xlocal));
  PetscCall(DMGetLocalVector(da, &ylocal));
  PetscCall(DMGlobalToLocal(da, X, INSERT_VALUES, xlocal));
  PetscCall(VecGetArray(xlocal, &x_array_vec));
  PetscCall(VecGetArray(ylocal, &y_array_vec));

  PetscScalar (* x_array)[info.gxm] = (PetscScalar (*)[info.gxm]) x_array_vec;
  PetscScalar (* y_array)[info.gxm] = (PetscScalar (*)[info.gxm]) y_array_vec;

  if (yints == 0) {
    j     = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i] = x_array[j][i];
      }
    }
 if (yinte == info.my) {
    j     = info.my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i]     = x_array[j][i];
    }
  }
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]     = x_array[j][i];
    }
  }
  /* Test whether we are on the right edge of the global array */
  if (xinte == info.mx) {
    i     = info.mx - 1;
    xinte = xinte - 1;
    /* right edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]     = x_array[j][i];
    }
  }
  for (j = yints; j < yinte; j++) {
    for (i = xints; i < xinte; i++) {
      uxx       = (2.0 * x_array[j][i] - x_array[j][i-1] - x_array[j][i+1]) * hydhx;
      uyy       = (2.0 * x_array[j][i] - x_array[j-1][i] - x_array[j+1][i]) * hxdhy;
      y_array[j][i] = uxx + uyy;
    }
  }
  PetscCall(VecRestoreArray(xlocal, &x_array_vec));
  PetscCall(VecRestoreArray(ylocal, &y_array_vec));
  PetscCall(DMLocalToGlobal(da, ylocal, INSERT_VALUES, Y));
  PetscFunctionReturn(0);
}

//dfu/domega
static PetscErrorCode MySubMatMult_J02(Mat J02, Vec X, Vec Y)
{
  SubmatrixContext *sub_ctx;
  PetscScalar *x_array_vec, *y_array_vec;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec xlocal, ylocal;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hx, dhx;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J02, &sub_ctx));

  da = sub_ctx->dm;
  PetscCall(DMDAGetLocalInfo(da, &info));

  dhx   = (PetscReal)(info.mx - 1);
  hx    = 1.0 / dhx;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J02, &n, NULL));
  PetscCall(DMGetLocalVector(da, &ylocal));
  PetscCall(DMGetLocalVector(da, &xlocal));
  PetscCall(DMGlobalToLocal(da, X, INSERT_VALUES, xlocal));
  PetscCall(VecGetArray(xlocal, &x_array_vec));
  PetscCall(VecGetArray(ylocal, &y_array_vec));

  PetscScalar (* x_array)[info.gxm] = (PetscScalar (*)[info.gxm]) x_array_vec;
  PetscScalar (* y_array)[info.gxm] = (PetscScalar (*)[info.gxm]) y_array_vec;

  if (yints == 0) {
    j     = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i] = 0.0;
    }
  }
  if (yinte == info.my) {
    j     = info.my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i]     = 0.0;
    }
  }
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]     = 0.0;
    }
  }
  /* Test whether we are on the right edge of the global array */
  if (xinte == info.mx) {
    i     = info.mx - 1;
    xinte = xinte - 1;
    /* right edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]     = 0.0;
    }
  }
  for (j = yints; j < yinte; j++) {
    for (i = xints; i < xinte; i++) {
      y_array[j][i] = -.5 * (x_array[j+1][i] - x_array[j-1][i]) * hx;
    }
  }
  PetscCall(VecRestoreArray(xlocal, &x_array_vec));
  PetscCall(VecRestoreArray(ylocal, &y_array_vec));
  PetscCall(DMLocalToGlobal(da, ylocal, ADD_VALUES, Y));
  PetscFunctionReturn(0);
}


// dfv/domega
static PetscErrorCode MySubMatMult_J12(Mat J12, Vec X, Vec Y)
{
  SubmatrixContext *sub_ctx;
  PetscScalar *x_array_vec, *y_array_vec;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec xlocal, ylocal;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hy, dhy;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J12, &sub_ctx));

  da = sub_ctx->dm;

  PetscCall(DMDAGetLocalInfo(da, &info));

  dhy   = (PetscReal)(info.my - 1);
  hy    = 1.0 / dhy;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J12, &n, NULL));
  PetscCall(DMGetLocalVector(da, &xlocal));
  PetscCall(DMGetLocalVector(da, &ylocal));
  PetscCall(DMGlobalToLocal(da, X, INSERT_VALUES, xlocal));
  PetscCall(VecGetArray(xlocal, &x_array_vec));
  PetscCall(VecGetArray(ylocal, &y_array_vec));

  PetscScalar (* x_array)[info.gxm] = (PetscScalar (*)[info.gxm]) x_array_vec;
  PetscScalar (* y_array)[info.gxm] = (PetscScalar (*)[info.gxm]) y_array_vec;


  if (yints == 0) {
    j     = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i] = 0.0;
    }
  }
  if (yinte == info.my) {
    j     = info.my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i]     = 0.0;
    }
  }
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]     = 0.0;
    }
  }
  /* Test whether we are on the right edge of the global array */
  if (xinte == info.mx) {
    i     = info.mx - 1;
    xinte = xinte - 1;
    /* right edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]     = 0.0;
    }
  }
  for (j = yints; j < yinte; j++) {
    for (i = xints; i < xinte; i++) {
      y_array[j][i] = .5 * (x_array[j][i+1] - x_array[j][i-1]) * hy;
    }
  }
  PetscCall(VecRestoreArray(xlocal, &x_array_vec));
  PetscCall(VecRestoreArray(ylocal, &y_array_vec));
  PetscCall(DMLocalToGlobal(da, ylocal, ADD_VALUES, Y));
  PetscFunctionReturn(0);
}


static PetscErrorCode MySubMatMult_J20(Mat J20, Vec X, Vec Y)
{
  PetscScalar *x_array_vec, *y_array_vec;
  PetscScalar *omega_array_vec, *u_array_vec;
  SubmatrixContext *sub_ctx;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec ylocal, xlocal;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hy, dhy;
  PetscScalar vx;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J20, &sub_ctx));

  da = sub_ctx->dm;
  PetscCall(DMDAGetLocalInfo(da, &info));

  dhy   = (PetscReal)(info.my - 1);
  hy    = 1.0 / dhy;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J20, &n, NULL));

  PetscCall(DMGetLocalVector(da, &ylocal));
  PetscCall(DMGetLocalVector(da, &xlocal));
  PetscCall(DMGlobalToLocal(da, X, INSERT_VALUES, xlocal));
  PetscCall(VecGetArray(xlocal, &x_array_vec));
  PetscCall(VecGetArray(ylocal, &y_array_vec));
  PetscCall(VecGetArray(sub_ctx->omega, &omega_array_vec));
  PetscCall(VecGetArray(sub_ctx->uvec, &u_array_vec));

  PetscScalar (* u_array)[info.gxm] = (PetscScalar (*)[info.gxm]) u_array_vec;
  PetscScalar (* omega_array)[info.gxm] = (PetscScalar (*)[info.gxm]) omega_array_vec;
  PetscScalar (* x_array)[info.gxm] = (PetscScalar (*)[info.gxm]) x_array_vec;
  PetscScalar (* y_array)[info.gxm] = (PetscScalar (*)[info.gxm]) y_array_vec;

  if (yints == 0) {
    j     = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i] = (x_array[j + 1][i] - x_array[j][i]) * dhy;
    }
  }
  if (yinte == info.my) {
    j     = info.my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
        y_array[j][i]     = (x_array[j][i] - x_array[j-1][i]) * dhy;
        }
  }
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]     = 0.0;
    }
  }
  /* Test whether we are on the right edge of the global array */
  if (xinte == info.mx) {
    i     = info.mx - 1;
    xinte = xinte - 1;
    /* right edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]     = 0.0;
    }
  }
  for (j = yints; j < yinte; j++) {
    for (i = xints; i < xinte; i++) {
      vx = u_array[j][i];

      if (PetscRealPart(vx) > 0.0) y_array[j][i] = (omega_array[j][i] - omega_array[j][i-1]) * hy * x_array[j][i];
      else y_array[j][i] = (omega_array[j][i+1] - omega_array[j][i]) * hy * x_array[j][i];
    }
  }
  PetscCall(VecRestoreArray(sub_ctx->omega, omega_array_vec));
  PetscCall(VecRestoreArray(sub_ctx->uvec, u_array_vec));
  PetscCall(VecRestoreArray(xlocal, &x_array_vec));
  PetscCall(VecRestoreArray(ylocal, &y_array_vec));
  PetscCall(DMLocalToGlobal(da, ylocal, INSERT_VALUES, Y));
  PetscFunctionReturn(0);
}


static PetscErrorCode MySubMatMult_J21(Mat J21, Vec X, Vec Y)
{
  PetscScalar *x_array_vec, *y_array_vec;
  PetscScalar *v_array_vec, *omega_array_vec;
  SubmatrixContext *sub_ctx;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec xlocal, ylocal;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hx, dhx;
  PetscScalar vy;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J21, &sub_ctx));

  da = sub_ctx->dm;
  PetscCall(DMDAGetLocalInfo(da, &info));

  dhx   = (PetscReal)(info.mx - 1);
  hx    = 1.0 / dhx;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J21, &n, NULL));
  
  PetscCall(DMGetLocalVector(da, &ylocal));
  PetscCall(DMGetLocalVector(da, &xlocal));
  PetscCall(DMGlobalToLocal(da, X, INSERT_VALUES, xlocal));
  PetscCall(VecGetArray(xlocal, &x_array_vec));
  PetscCall(VecGetArray(sub_ctx->omega, &omega_array_vec));
  PetscCall(VecGetArray(sub_ctx->vvec, &v_array_vec));
  PetscCall(VecGetArray(ylocal, &y_array_vec));

  PetscScalar (* v_array)[info.gxm] = (PetscScalar (*)[info.gxm]) v_array_vec;
  PetscScalar (* omega_array)[info.gxm] = (PetscScalar (*)[info.gxm]) omega_array_vec;
  PetscScalar (* x_array)[info.gxm] = (PetscScalar (*)[info.gxm]) x_array_vec;
  PetscScalar (* y_array)[info.gxm] = (PetscScalar (*)[info.gxm]) y_array_vec;

  if (yints == 0) {
    j     = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i] = 0.0;

    }
  }
  if (yinte == info.my) {
    j     = info.my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i]     = 0.0;
    }
  }
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i] = - (x_array[j][i+1] - x_array[j][i]) * dhx;
    }
  }
  /* Test whether we are on the right edge of the global array */
  if (xinte == info.mx) {
    i     = info.mx - 1;
    xinte = xinte - 1;
    /* right edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]= -(x_array[j][i] - x_array[j][i-1]) * dhx;
    }
  }
  for (j = yints; j < yinte; j++) {
    for (i = xints; i < xinte; i++) {
      vy = v_array[j][i];

      if (PetscRealPart(vy) > 0.0) y_array[j][i] = (omega_array[j][i] - omega_array[j-1][i]) * hx * x_array[j][i];
      else y_array[j][i] = (omega_array[j+1][i] - omega_array[j][i]) * hx * x_array[j][i];

    }
  }
  PetscCall(VecRestoreArray(sub_ctx->omega, omega_array_vec));
  PetscCall(VecRestoreArray(sub_ctx->vvec, v_array_vec));
  PetscCall(VecRestoreArray(xlocal, &x_array_vec));
  PetscCall(VecRestoreArray(ylocal, &y_array_vec));
  PetscCall(DMLocalToGlobal(da, ylocal, ADD_VALUES, Y));
  PetscFunctionReturn(0);
}


static PetscErrorCode MySubMatMult_J22(Mat J22, Vec X, Vec Y)
{
  SubmatrixContext *sub_ctx;
  PetscScalar *y_array_vec, *x_array_vec;
  PetscScalar *u_array_vec, *v_array_vec;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec xlocal, ylocal;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hx, hy, dhx, dhy, hxdhy, hydhx;
  PetscScalar uxx, uyy, vx, vy, avx, avy, vxp, vxm, vyp, vym;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J22, &sub_ctx));

  da = sub_ctx->dm;

  PetscCall(DMDAGetLocalInfo(da, &info));

  dhx   = (PetscReal)(info.mx - 1);
  dhy   = (PetscReal)(info.my - 1);
  hx    = 1.0 / dhx;
  hy    = 1.0 / dhy;
  hxdhy = hx * dhy;
  hydhx = hy * dhx;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J22, &n, NULL));

  PetscCall(DMGetLocalVector(da, &ylocal));
  PetscCall(DMGetLocalVector(da, &xlocal));
  PetscCall(DMGlobalToLocal(da, X, INSERT_VALUES, xlocal));
  PetscCall(VecGetArray(xlocal, &x_array_vec));
  PetscCall(VecGetArray(ylocal, &y_array_vec));
  PetscCall(VecGetArray(sub_ctx->uvec, &u_array_vec));
  PetscCall(VecGetArray(sub_ctx->vvec, &v_array_vec));

  PetscScalar (* u_array)[info.gxm] = (PetscScalar (*)[info.gxm]) u_array_vec;
  PetscScalar (* v_array)[info.gxm] = (PetscScalar (*)[info.gxm]) v_array_vec;
  PetscScalar (* x_array)[info.gxm] = (PetscScalar (*)[info.gxm]) x_array_vec;
  PetscScalar (* y_array)[info.gxm] = (PetscScalar (*)[info.gxm]) y_array_vec;

  if (yints == 0) {
    j     = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i] = x_array[j][i];
    }
  }
  if (yinte == info.my) {
    j     = info.my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i]     = x_array[j][i];
    }
  }
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i] = x_array[j][i];
    }
  }
  /* Test whether we are on the right edge of the global array */
  if (xinte == info.mx) {
    i     = info.mx - 1;
    xinte = xinte - 1;
    /* right edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]= x_array[j][i];
    }
  }
  for (j = yints; j < yinte; j++) {
    for (i = xints; i < xinte; i++) {
      vx = u_array[j][i];
      avx = PetscAbsScalar(vx);
      vxp = .5 * (vx + avx);
      vxm = .5 * (vx - avx);
      vy = v_array[j][i];
      avy = PetscAbsScalar(vy);
      vyp = .5 * (vy + avy);
      vym = .5 * (vy - avy);

      uxx           = (2.0 * x_array[j][i] - x_array[j][i-1] - x_array[j][i+1]) * hydhx;
      uyy           = (2.0 * x_array[j][i] - x_array[j-1][i] - x_array[j+1][i]) * hxdhy;

      y_array[j][i] = uxx + uyy + (vxp * (x_array[j][i] - x_array[j][i-1]) + vxm * (x_array[j][i+1] - x_array[j][i])) * hy + (vyp * (x_array[j][i] - x_array[j-1][i]) + vym * (x_array[j+1][i] - x_array[j][i])) * hx;
    }
  }

  PetscCall(VecRestoreArray(sub_ctx->uvec, u_array_vec));
  PetscCall(VecRestoreArray(sub_ctx->vvec, v_array_vec));
  PetscCall(VecRestoreArray(xlocal, &x_array_vec));
  PetscCall(VecRestoreArray(ylocal, &y_array_vec));
  PetscCall(DMLocalToGlobal(da, ylocal, ADD_VALUES, Y));
  PetscFunctionReturn(0);
}


// domega/dT
static PetscErrorCode MySubMatMult_J23(Mat J23, Vec X, Vec Y)
{
  PetscScalar *x_array_vec, *y_array_vec;
  SubmatrixContext *sub_ctx;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec xlocal, ylocal;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hy, dhy;
  PetscReal   grashof;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J23, &sub_ctx));

  da = sub_ctx->dm;
  PetscCall(DMDAGetLocalInfo(da, &info));

  grashof = 1.0;

  dhy   = (PetscReal)(info.my - 1);
  hy    = 1.0 / dhy;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J23, &n, NULL));

  PetscCall(DMGetLocalVector(da, &ylocal));
  PetscCall(DMGetLocalVector(da, &xlocal));
  PetscCall(DMGlobalToLocal(da, X, INSERT_VALUES, xlocal));
  PetscCall(VecGetArray(xlocal, &x_array_vec));
  PetscCall(VecGetArray(ylocal, &y_array_vec));

  PetscScalar (* x_array)[info.gxm] = (PetscScalar (*)[info.gxm]) x_array_vec;
  PetscScalar (* y_array)[info.gxm] = (PetscScalar (*)[info.gxm]) y_array_vec;

  if (yints == 0) {
    j     = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i] = 0;
    }
  }
  if (yinte == info.my) {
      j     = info.my - 1;
      yinte = yinte - 1;
      /* top edge */
      for (i = info.xs; i < info.xs + info.xm; i++) {
        y_array[j][i]     = 0;
      }
    }
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i] = 0;
    }
  }
  /* Test whether we are on the right edge of the global array */
  if (xinte == info.mx) {
    i     = info.mx - 1;
    xinte = xinte - 1;
    /* right edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]= 0;
    }
  }
  for (j = yints; j < yinte; j++) {
    for (i = xints; i < xinte; i++) {
      y_array[j][i] = - .5 * grashof * (x_array[j][i+1] - x_array[j][i-1]) * hy;
    }
  }
  
  PetscCall(VecRestoreArray(xlocal, &x_array_vec));
  PetscCall(VecRestoreArray(ylocal, &y_array_vec));
  PetscCall(DMLocalToGlobal(da, ylocal, ADD_VALUES, Y));
  PetscFunctionReturn(0);
}


// df(T)/du
static PetscErrorCode MySubMatMult_J30(Mat J30, Vec X, Vec Y)
{
  PetscScalar *x_array_vec, *y_array_vec;
  PetscScalar *temp_array_vec;
  SubmatrixContext *sub_ctx;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hy, dhy;
  PetscReal   prandtl;
  PetscScalar vx, avx, vxp, vxm;
  Vec xlocal, ylocal;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J30, &sub_ctx));

  da = sub_ctx->dm;
  PetscCall(DMDAGetLocalInfo(da, &info));

  prandtl = 1.0;
  dhy   = (PetscReal)(info.my - 1);
  hy    = 1.0 / dhy;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J30, &n, NULL));
  PetscCall(DMGetLocalVector(da, &ylocal));
  PetscCall(DMGetLocalVector(da, &xlocal));
  PetscCall(DMGlobalToLocal(da, X, INSERT_VALUES, xlocal));
  PetscCall(VecGetArray(xlocal, &x_array_vec));
  PetscCall(VecGetArray(ylocal, &y_array_vec));
  PetscCall(VecGetArray(sub_ctx->temp, &temp_array_vec));

  PetscScalar (* temp_array)[info.gxm] = (PetscScalar (*)[info.gxm]) temp_array_vec;
  PetscScalar (* x_array)[info.gxm] = (PetscScalar (*)[info.gxm]) x_array_vec;
  PetscScalar (* y_array)[info.gxm] = (PetscScalar (*)[info.gxm]) y_array_vec;

  if (yints == 0) {
    j     = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i] = 0;
    }
  }
 if (yinte == info.my) {
    j     = info.my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i]     = 0;
    }
  }
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i] = 0;
    }
  }
  /* Test whether we are on the right edge of the global array */
  if (xinte == info.mx) {
    i     = info.mx - 1;
    xinte = xinte - 1;
    /* right edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]= 0;
    }
  }
  for (j = yints; j < yinte; j++) {
    for (i = xints; i < xinte; i++) {
      vx  = x_array[j][i];
      avx = PetscAbsScalar(vx);
      vxp = .5 * (vx + avx);
      vxm = .5 * (vx - avx);
      y_array[j][i] = prandtl * (vxp * (temp_array[j][i] - temp_array[j][i - 1]) + vxm * (temp_array[j][i+1] - temp_array[j][i])) * hy;
    }
  }

  PetscCall(VecRestoreArray(xlocal, &x_array_vec));
  PetscCall(VecRestoreArray(ylocal, &y_array_vec));
  PetscCall(VecRestoreArray(sub_ctx->temp, &temp_array_vec));
  PetscCall(DMLocalToGlobal(da, ylocal, INSERT_VALUES, Y));
  PetscFunctionReturn(0);
}


static PetscErrorCode MySubMatMult_J31(Mat J31, Vec X, Vec Y)
{
  PetscScalar *x_array_vec, *y_array_vec;
  PetscScalar *temp_array_vec, *v_array_vec;
  SubmatrixContext *sub_ctx;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec xlocal, ylocal;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hx, dhx;
  PetscReal   prandtl;
  PetscScalar vy;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J31, &sub_ctx));

  da = sub_ctx->dm;
  PetscCall(DMDAGetLocalInfo(da, &info));

  prandtl = 1.0;

  dhx   = (PetscReal)(info.mx - 1);
  hx    = 1.0 / dhx;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J31, &n, NULL));
  PetscCall(DMGetLocalVector(da, &ylocal));
  PetscCall(DMGetLocalVector(da, &xlocal));
  PetscCall(DMGlobalToLocal(da, X, INSERT_VALUES, xlocal));
  PetscCall(VecGetArray(xlocal, &x_array_vec));
  PetscCall(VecGetArray(ylocal, &y_array_vec));
  PetscCall(VecGetArray(sub_ctx->temp, &temp_array_vec));
  PetscCall(VecGetArray(sub_ctx->vvec, &v_array_vec));

  PetscScalar (* temp_array)[info.gxm] = (PetscScalar (*)[info.gxm]) temp_array_vec;
  PetscScalar (* v_array)[info.gxm] = (PetscScalar (*)[info.gxm]) v_array_vec;
  PetscScalar (* x_array)[info.gxm] = (PetscScalar (*)[info.gxm]) x_array_vec;
  PetscScalar (* y_array)[info.gxm] = (PetscScalar (*)[info.gxm]) y_array_vec;

  if (yints == 0) {
    j     = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i] = 0;
    }
  }
  if (yinte == info.my) {
      j     = info.my - 1;
      yinte = yinte - 1;
      /* top edge */
      for (i = info.xs; i < info.xs + info.xm; i++) {
        y_array[j][i]     = 0;
      }
    }
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i] = 0;
    }
  }
  /* Test whether we are on the right edge of the global array */
  if (xinte == info.mx) {
    i     = info.mx - 1;
    xinte = xinte - 1;
    /* right edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]= 0;
    }
  }
  for (j = yints; j < yinte; j++) {
    for (i = xints; i < xinte; i++) {
      vy = v_array[j][i];

      if (PetscRealPart(vy) > 0.0) y_array[j][i] = prandtl * (temp_array[j][i] - temp_array[j-1][i]) * hx * x_array[j][i];
      else y_array[j][i] = prandtl * (temp_array[j+1][i] - temp_array[j][i]) * hx * x_array[j][i];

    }
  }

  PetscCall(VecRestoreArray(sub_ctx->temp, &temp_array_vec));
  PetscCall(VecRestoreArray(sub_ctx->vvec, &v_array_vec));
  PetscCall(VecRestoreArray(xlocal, &x_array_vec));
  PetscCall(VecRestoreArray(ylocal, &y_array_vec));
  PetscCall(DMLocalToGlobal(da, ylocal, ADD_VALUES, Y));
  PetscFunctionReturn(0);
}


static PetscErrorCode MySubMatMult_J33(Mat J33, Vec X, Vec Y)
{
  PetscScalar *x_array_vec, *y_array_vec;
  PetscScalar *v_array_vec, *u_array_vec;
  SubmatrixContext *sub_ctx;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec xlocal, ylocal;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hx, hy, dhx, dhy, hxdhy, hydhx;
  PetscReal   prandtl;
  PetscScalar uxx, uyy, vx, vy, avx, avy, vxp, vxm, vyp, vym;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J33, &sub_ctx));

  da = sub_ctx->dm;
  PetscCall(DMDAGetLocalInfo(da, &info));

  prandtl = 1.0;

  dhx   = (PetscReal)(info.mx - 1);
  dhy   = (PetscReal)(info.my - 1);
  hx    = 1.0 / dhx;
  hy    = 1.0 / dhy;
  hxdhy = hx * dhy;
  hydhx = hy * dhx;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J33, &n, NULL));
  PetscCall(DMGetLocalVector(da, &ylocal));
  PetscCall(DMGetLocalVector(da, &xlocal));
  PetscCall(DMGlobalToLocal(da, X, INSERT_VALUES, xlocal));
  PetscCall(VecGetArray(xlocal, &x_array_vec));
  PetscCall(VecGetArray(ylocal, &y_array_vec));
  PetscCall(VecGetArray(sub_ctx->uvec, &u_array_vec));
  PetscCall(VecGetArray(sub_ctx->vvec, &v_array_vec));

  PetscScalar (* u_array)[info.gxm] = (PetscScalar (*)[info.gxm]) u_array_vec;
  PetscScalar (* v_array)[info.gxm] = (PetscScalar (*)[info.gxm]) v_array_vec;
  PetscScalar (* x_array)[info.gxm] = (PetscScalar (*)[info.gxm]) x_array_vec;
  PetscScalar (* y_array)[info.gxm] = (PetscScalar (*)[info.gxm]) y_array_vec;


  if (yints == 0) {
    j     = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      y_array[j][i] = x_array[j][i] - x_array[j+1][i];

    }
  }
  if (yinte == info.my) {
      j     = info.my - 1;
      yinte = yinte - 1;
      /* top edge */
      for (i = info.xs; i < info.xs + info.xm; i++) {
        y_array[j][i]     = x_array[j][i] - x_array[j-1][i];
      }
    }
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i] = x_array[j][i];
    }
  }
  /* Test whether we are on the right edge of the global array */
  if (xinte == info.mx) {
    i     = info.mx - 1;
    xinte = xinte - 1;
    /* right edge */
    for (j = info.ys; j < info.ys + info.ym; j++) {
      y_array[j][i]= x_array[j][i];
    }
  }
  for (j = yints; j < yinte; j++) {
    for (i = xints; i < xinte; i++) {
      vx  = u_array[j][i];
      avx = PetscAbsScalar(vx);
      vxp = .5 * (vx + avx);
      vxm = .5 * (vx - avx);
      vy  = v_array[j][i];
      avy = PetscAbsScalar(vy);
      vyp = .5 * (vy + avy);
      vym = .5 * (vy - avy);

      uxx          = (2.0 * x_array[j][i] - x_array[j][i-1] - x_array[j][i+1]) * hydhx;
      uyy          = (2.0 * x_array[j][i] - x_array[j-1][i] - x_array[j+1][i]) * hxdhy;
      y_array[j][i] = uxx + uyy + prandtl * ((vxp * (x_array[j][i] - x_array[j][i-1]) + vxm * (x_array[j][i+1] - x_array[j][i])) * hy + (vyp * (x_array[j][i] - x_array[j-1][i]) + vym * (x_array[j+1][i] - x_array[j][i])) * hx);
    }
  }

  PetscCall(VecRestoreArray(sub_ctx->uvec, &u_array_vec));
  PetscCall(VecRestoreArray(sub_ctx->vvec, &v_array_vec));
  PetscCall(VecRestoreArray(xlocal, &x_array_vec));
  PetscCall(VecRestoreArray(ylocal, &y_array_vec));
  PetscCall(DMLocalToGlobal(da, ylocal, ADD_VALUES, Y));
  PetscFunctionReturn(0);
}


static PetscErrorCode MatCreateSubMatrices_Shell(Mat mat, PetscInt n, IS irow[], IS icol[], MatReuse scall, Mat **submat)
{
    PetscFunctionBegin;
    PetscCall(PetscMalloc1(n, submat));
    DM dm;
    PetscCall(MatGetDM(mat, &dm));
    JacobianContext *jac_ctx;
    PetscCall(MatShellGetContext(mat, &jac_ctx));

    PetscInt M, N;
    PetscCall(DMDAGetInfo(dm, NULL, &M, &N, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));

    for (PetscInt i = 0; i < n; ++i) {
        Mat block;
        PetscCall(MatCreate(PETSC_COMM_SELF, &block));
        PetscCall(MatSetSizes(block, M*N, M*N, PETSC_DECIDE, PETSC_DECIDE));
        PetscCall(MatSetType(block, MATSHELL));
        SubmatrixContext *sub_ctx;
        PetscCall(PetscMalloc1(1, &sub_ctx));
        sub_ctx->dm = jac_ctx->sub_dms[i/4];

        PetscInt row_idx = i / 4;
        PetscInt col_idx = i % 4;

        sub_ctx->rows = irow[row_idx];
        sub_ctx->cols = icol[col_idx];

        PetscCall(MatShellSetContext(block, sub_ctx));
        PetscCall(MatSetUp(block));
        (*submat)[i] = block;
    }
    //dfu/du
    PetscCall(MatShellSetOperation((*submat)[0], MATOP_MULT, (void (*)(void))MySubMatMult_J00));

    //dfu/domega
    PetscCall(MatShellSetOperation((*submat)[2], MATOP_MULT, (void (*)(void))MySubMatMult_J02));

    //dfv/dv -> can use the same callback as dfu/du
    PetscCall(MatShellSetOperation((*submat)[5], MATOP_MULT, (void (*)(void))MySubMatMult_J00));

    //dfv/domega
    PetscCall(MatShellSetOperation((*submat)[6], MATOP_MULT, (void (*)(void))MySubMatMult_J12));

    //dfomega/du
    PetscCall(MatShellSetOperation((*submat)[8], MATOP_MULT, (void (*)(void))MySubMatMult_J20));

    //dfomega/dv
    PetscCall(MatShellSetOperation((*submat)[9], MATOP_MULT, (void (*)(void))MySubMatMult_J21));

    //dfomega/domega
    PetscCall(MatShellSetOperation((*submat)[10], MATOP_MULT, (void (*)(void))MySubMatMult_J22));

    //domega/dT
    PetscCall(MatShellSetOperation((*submat)[11], MATOP_MULT, (void (*)(void))MySubMatMult_J23));

    //dfT/du
    PetscCall(MatShellSetOperation((*submat)[12], MATOP_MULT, (void (*)(void))MySubMatMult_J30));

    //dfT/dv
    PetscCall(MatShellSetOperation((*submat)[13], MATOP_MULT, (void (*)(void))MySubMatMult_J31));

    //df(T)/domega is zero

    //df(T)/dT
    PetscCall(MatShellSetOperation((*submat)[15], MATOP_MULT, (void (*)(void))MySubMatMult_J33));

    PetscFunctionReturn(0);
}


int main(int argc, char **argv)
{
  AppCtx   user; /* user-defined work context */
  PetscInt mx, my;
  MPI_Comm comm;
  SNES     snes;
  DM       da;
  Vec      x;
  Mat J;
  IS          *fields;
  DM          *sub_dms;
  PetscInt n=16;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_SELF;
  PetscCall(SNESCreate(comm, &snes));

  /*
      Create distributed array object to manage parallel grid and vectors
      for principal unknowns (x) and governing residuals (f)
  */
  PetscCall(DMDACreate2d(PETSC_COMM_SELF, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 4, 1, 0, 0, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMSetMatType(da, MATSHELL));
  PetscCall(SNESSetDM(snes, da));

  PetscCall(DMDAGetInfo(da, 0, &mx, &my, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
  /*
     Problem parameters (velocity of lid, prandtl, and grashof numbers)
  */
  user.lidvelocity = 1.0 / (mx * my);
  user.prandtl     = 1.0;
  user.grashof     = 1.0;

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lidvelocity", &user.lidvelocity, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-prandtl", &user.prandtl, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-grashof", &user.grashof, NULL));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-contours", &user.draw_contours));

  PetscCall(DMDASetFieldName(da, 0, "x_velocity"));
  PetscCall(DMDASetFieldName(da, 1, "y_velocity"));
  PetscCall(DMDASetFieldName(da, 2, "Omega"));
  PetscCall(DMDASetFieldName(da, 3, "temperature"));

  PetscCall(DMSetApplicationContext(da, &user));
  PetscCall(DMDASNESSetFunctionLocal(da, INSERT_VALUES, (PetscErrorCode (*)(DMDALocalInfo *, void *, void *, void *))FormFunctionLocal, &user));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateMatrix(da, &J));
  PetscCall(DMCreateFieldDecomposition(da, NULL, NULL, &fields, &sub_dms));

  PetscCall(MatShellSetOperation(J, MATOP_CREATE_SUBMATRICES, (void (*)(void))MatCreateSubMatrices_Shell));

  JacobianContext * ctx;
  PetscCall(PetscMalloc1(1, &ctx));
  ctx->n_submats = n;
  PetscCall(PetscMalloc1(n, &ctx->submats));
  PetscCall(PetscMalloc1(4, &ctx->fields));

  ctx->sub_dms = sub_dms;
  ctx->snes = snes;
  ctx->fields = fields;

  PetscCall(MatSetDM(J, da));
  PetscCall(MatShellSetContext(J, ctx));
  PetscCall(MatCreateSubMatrices(J, ctx->n_submats, ctx->fields, ctx->fields, MAT_INITIAL_MATRIX, &ctx->submats));

  PetscCall(SNESSetJacobian(snes,J,J,MatMFFDComputeJacobian,NULL));
  PetscCall(MatShellSetOperation(J,MATOP_MULT,(void (*)(void))MyMatShellMult));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(FormInitialGuess(&user, da, x));

  PetscCall(SNESSolve(snes, NULL, x));

  PetscCall(VecDestroy(&x));
  PetscCall(DMDestroy(&da));
  PetscCall(SNESDestroy(&snes));
  PetscCall(MatDestroy(&J));

  for (PetscInt i = 0; i < 4; ++i) {
    PetscCall(ISDestroy(&fields[i]));
    PetscCall(DMDestroy(&sub_dms[i]));
  }
  PetscCall(PetscFree(fields));
  PetscCall(PetscFree(sub_dms));

  for (PetscInt i = 0; i < ctx->n_submats; ++i) {
    PetscCall(MatDestroy(&ctx->submats[i]));
  }
  PetscCall(PetscFree(ctx->submats));
  PetscCall(PetscFree(ctx));
  PetscCall(PetscFinalize());
  return 0;
}

/* ------------------------------------------------------------------- */

/*
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
*/
PetscErrorCode FormInitialGuess(AppCtx *user, DM da, Vec X)
{
  PetscInt  i, j, mx, xs, ys, xm, ym;
  PetscReal grashof, dx;
  Field   **x;

  PetscFunctionBeginUser;
  grashof = user->grashof;

  PetscCall(DMDAGetInfo(da, 0, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  dx = 1.0 / (mx - 1);

  /*
     Get local grid boundaries (for 2-dimensional DMDA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
  */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  PetscCall(DMDAVecGetArrayWrite(da, X, &x));

  /*
     Compute initial guess over the locally owned part of the grid
     Initial condition is motionless fluid and equilibrium temperature
  */
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      x[j][i].u     = 0.0;
      x[j][i].v     = 0.0;
      x[j][i].omega = 0.0;
      x[j][i].temp  = (grashof > 0) * i * dx;
    }
  }

  /*
     Restore vector
  */
  PetscCall(DMDAVecRestoreArrayWrite(da, X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, Field **x, Field **f, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hx, hy, dhx, dhy, hxdhy, hydhx;
  PetscReal   grashof, prandtl, lid;
  PetscScalar u, uxx, uyy, vx, vy, avx, avy, vxp, vxm, vyp, vym;

  PetscFunctionBeginUser;
  grashof = user->grashof;
  prandtl = user->prandtl;
  lid     = user->lidvelocity;

  /*
     Define mesh intervals ratios for uniform grid.

     Note: FD formulae below are normalized by multiplying through by
     local volume element (i.e. hx*hy) to obtain coefficients O(1) in two dimensions.

  */
  dhx   = (PetscReal)(info->mx - 1);
  dhy   = (PetscReal)(info->my - 1);
  hx    = 1.0 / dhx;
  hy    = 1.0 / dhy;
  hxdhy = hx * dhy;
  hydhx = hy * dhx;

  xints = info->xs;
  xinte = info->xs + info->xm;
  yints = info->ys;
  yinte = info->ys + info->ym;

  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j     = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i = info->xs; i < info->xs + info->xm; i++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega + (x[j + 1][i].u - x[j][i].u) * dhy;
      f[j][i].temp  = x[j][i].temp - x[j + 1][i].temp;
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == info->my) {
    j     = info->my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i = info->xs; i < info->xs + info->xm; i++) {
      f[j][i].u     = x[j][i].u - lid;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega + (x[j][i].u - x[j - 1][i].u) * dhy;
      f[j][i].temp  = x[j][i].temp - x[j - 1][i].temp;
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i     = 0;
    xints = xints + 1;
    /* left edge */
    for (j = info->ys; j < info->ys + info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i + 1].v - x[j][i].v) * dhx;
      f[j][i].temp  = x[j][i].temp;
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info->mx) {
    i     = info->mx - 1;
    xinte = xinte - 1;
    /* right edge */
    for (j = info->ys; j < info->ys + info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i].v - x[j][i - 1].v) * dhx;
      f[j][i].temp  = x[j][i].temp - (PetscReal)(grashof > 0);
    }
  }

  /* Compute over the interior points */
  for (j = yints; j < yinte; j++) {
    for (i = xints; i < xinte; i++) {
      /*
       convective coefficients for upwinding
      */
      vx  = x[j][i].u;
      avx = PetscAbsScalar(vx);
      vxp = .5 * (vx + avx);
      vxm = .5 * (vx - avx);
      vy  = x[j][i].v;
      avy = PetscAbsScalar(vy);
      vyp = .5 * (vy + avy);
      vym = .5 * (vy - avy);

      /* U velocity */
      u         = x[j][i].u;
      uxx       = (2.0 * u - x[j][i - 1].u - x[j][i + 1].u) * hydhx;
      uyy       = (2.0 * u - x[j - 1][i].u - x[j + 1][i].u) * hxdhy;
      f[j][i].u = uxx + uyy - .5 * (x[j + 1][i].omega - x[j - 1][i].omega) * hx;

      /* V velocity */
      u         = x[j][i].v;
      uxx       = (2.0 * u - x[j][i - 1].v - x[j][i + 1].v) * hydhx;
      uyy       = (2.0 * u - x[j - 1][i].v - x[j + 1][i].v) * hxdhy;
      f[j][i].v = uxx + uyy + .5 * (x[j][i + 1].omega - x[j][i - 1].omega) * hy;

      /* Omega */
      u             = x[j][i].omega;
      uxx           = (2.0 * u - x[j][i - 1].omega - x[j][i + 1].omega) * hydhx;
      uyy           = (2.0 * u - x[j - 1][i].omega - x[j + 1][i].omega) * hxdhy;
      f[j][i].omega = uxx + uyy + (vxp * (u - x[j][i - 1].omega) + vxm * (x[j][i + 1].omega - u)) * hy + (vyp * (u - x[j - 1][i].omega) + vym * (x[j + 1][i].omega - u)) * hx - .5 * grashof * (x[j][i + 1].temp - x[j][i - 1].temp) * hy;

      /* Temperature */
      u            = x[j][i].temp;
      uxx          = (2.0 * u - x[j][i - 1].temp - x[j][i + 1].temp) * hydhx;
      uyy          = (2.0 * u - x[j - 1][i].temp - x[j + 1][i].temp) * hxdhy;
      f[j][i].temp = uxx + uyy + prandtl * ((vxp * (u - x[j][i - 1].temp) + vxm * (x[j][i + 1].temp - u)) * hy + (vyp * (u - x[j - 1][i].temp) + vym * (x[j + 1][i].temp - u)) * hx);
    }
  }

  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  PetscCall(PetscLogFlops(84.0 * info->ym * info->xm));
  PetscFunctionReturn(PETSC_SUCCESS);
}