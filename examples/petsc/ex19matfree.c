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
    IS *rows;
    IS *cols;
    DM *sub_dms;
    IS *fields;
    SNES snes;
} JacobianContext;

typedef struct {
    Vec sol;
    PetscInt index;
    IS rows;
    IS cols;
    DM dm;
    Vec omega, uvec, vvec, temp;
    SNES snes;
} SubmatrixContext;

PetscErrorCode FormFunctionLocal(DMDALocalInfo *, Field **, Field **, void *);

typedef struct {
  PetscReal lidvelocity, prandtl, grashof; /* physical parameters */
  PetscBool draw_contours;                 /* flag - 1 indicates drawing contours */
} AppCtx;

extern PetscErrorCode FormInitialGuess(AppCtx *, DM, Vec);
extern PetscErrorCode NonlinearGS(SNES, Vec, Vec, void *);


PetscErrorCode MyMatShellMult(Mat J, Vec X, Vec Y) {
  JacobianContext *jac_ctx;
  DM da, da_u, da_v, da_omega, da_temp;
  Vec u, v, omega, temp;
  Vec uprev, vprev, omegaprev, tempprev;
  Vec yu0, yu2;
  Vec yv1, yv2;
  Vec yomega0, yomega1, yomega2, yomega3;
  Vec yT0, yT1, yT3;
  SNES snes;
  Vec sol;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J, &jac_ctx));
  snes = jac_ctx->snes;
  PetscCall(SNESGetSolution(snes, &sol));

  // get the sub_dms
  da_u = jac_ctx->sub_dms[0];
  da_v = jac_ctx->sub_dms[1];
  da_omega = jac_ctx->sub_dms[2];
  da_temp = jac_ctx->sub_dms[3];

  PetscCall(SNESGetDM(snes, &da));

  PetscCall(DMCreateGlobalVector(da_u, &u));
  PetscCall(DMCreateGlobalVector(da_v, &v));
  PetscCall(DMCreateGlobalVector(da_omega, &omega));
  PetscCall(DMCreateGlobalVector(da_temp, &temp));

  PetscCall(DMCreateGlobalVector(da_u, &uprev));
  PetscCall(DMCreateGlobalVector(da_v, &vprev));
  PetscCall(DMCreateGlobalVector(da_omega, &omegaprev));
  PetscCall(DMCreateGlobalVector(da_temp, &tempprev));

  PetscCall(DMCreateGlobalVector(da_u, &yu0));
  PetscCall(DMCreateGlobalVector(da_omega, &yu2));

  PetscCall(DMCreateGlobalVector(da_v, &yv1));
  PetscCall(DMCreateGlobalVector(da_omega, &yv2));

  PetscCall(DMCreateGlobalVector(da_u, &yomega0));
  PetscCall(DMCreateGlobalVector(da_v, &yomega1));
  PetscCall(DMCreateGlobalVector(da_omega, &yomega2));
  PetscCall(DMCreateGlobalVector(da_temp, &yomega3));

  PetscCall(DMCreateGlobalVector(da_u, &yT0));
  PetscCall(DMCreateGlobalVector(da_v, &yT1));
  PetscCall(DMCreateGlobalVector(da_temp, &yT3));

  VecScatter scatter_u, scatter_v, scatter_omega, scatter_temp;

  PetscCall(VecScatterCreate(X, jac_ctx->fields[0], u, NULL, &scatter_u));
  PetscCall(VecScatterCreate(X, jac_ctx->fields[1], v, NULL, &scatter_v));
  PetscCall(VecScatterCreate(X, jac_ctx->fields[2], omega, NULL, &scatter_omega));
  PetscCall(VecScatterCreate(X, jac_ctx->fields[3], temp, NULL, &scatter_temp));

  PetscCall(VecScatterBegin(scatter_u, X, u, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_u, X, u, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(scatter_omega, X, omega, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_omega, X, omega, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(scatter_v, X, v, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_v, X, v, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(scatter_temp, X, temp, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_temp, X, temp, INSERT_VALUES, SCATTER_FORWARD));

  // previous newton iteration solution
  PetscCall(VecScatterBegin(scatter_u, sol, uprev, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_u, sol, uprev, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(scatter_omega, sol, omegaprev, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_omega, sol, omegaprev, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(scatter_v, sol, vprev, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_v, sol, vprev, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(scatter_temp, sol, tempprev, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter_temp, sol, tempprev, INSERT_VALUES, SCATTER_FORWARD));

  //dfu/du
  PetscCall(MatMult(jac_ctx->submats[0], u, yu0));
  //dfu/domega
  PetscCall(MatMult(jac_ctx->submats[2], omega, yu2));
  // Yu
  PetscCall(VecAXPY(yu0, 1.0, yu2));

  //dfv/dv
  PetscCall(MatMult(jac_ctx->submats[5], v, yv1));
  //dfv/domega
  PetscCall(MatMult(jac_ctx->submats[6], omega, yv2));
  // Yv
  PetscCall(VecAXPY(yv1, 1.0, yv2));

  //dfomega/du
  SubmatrixContext *j20_ctx;
  PetscCall(MatShellGetContext(jac_ctx->submats[8], &j20_ctx));
  j20_ctx->omega = omegaprev;
  j20_ctx->uvec = uprev;
  PetscCall(MatShellSetContext(jac_ctx->submats[8], j20_ctx));
  PetscCall(MatMult(jac_ctx->submats[8], u, yomega0));

  //dfomega/dv
  SubmatrixContext *j21_ctx;
  PetscCall(MatShellGetContext(jac_ctx->submats[9], &j21_ctx));
  j21_ctx->omega = omegaprev;
  j21_ctx->vvec = vprev;
  PetscCall(MatShellSetContext(jac_ctx->submats[9], j21_ctx));
  PetscCall(MatMult(jac_ctx->submats[9], v, yomega1));

  //dfomega/domega
  SubmatrixContext *j22_ctx;
  PetscCall(MatShellGetContext(jac_ctx->submats[10], &j22_ctx));
  j22_ctx->uvec = uprev;
  j22_ctx->vvec = vprev;
  PetscCall(MatShellSetContext(jac_ctx->submats[10], j22_ctx));
  PetscCall(MatMult(jac_ctx->submats[10], omega, yomega2));

  //dfomega/dT
  PetscCall(MatMult(jac_ctx->submats[11], temp, yomega3));

  //Yomega
  PetscCall(VecAXPY(yomega0, 1.0, yomega1));
  PetscCall(VecAXPY(yomega0, 1.0, yomega2));
  PetscCall(VecAXPY(yomega0, 1.0, yomega3));

  //dfT/du
  SubmatrixContext *j30_ctx;
  PetscCall(MatShellGetContext(jac_ctx->submats[12], &j30_ctx));
  j30_ctx->temp = tempprev;
  PetscCall(MatShellSetContext(jac_ctx->submats[12], j30_ctx));
  PetscCall(MatMult(jac_ctx->submats[12], u, yT0));

  //dfT/dv
  SubmatrixContext *j31_ctx;
  PetscCall(MatShellGetContext(jac_ctx->submats[13], &j31_ctx));
  j31_ctx->temp = tempprev;
  j31_ctx->vvec = vprev;
  PetscCall(MatShellSetContext(jac_ctx->submats[13], j31_ctx));
  PetscCall(MatMult(jac_ctx->submats[13], v, yT1));

  //dfT/dT
  SubmatrixContext *j33_ctx;
  PetscCall(MatShellGetContext(jac_ctx->submats[15], &j33_ctx));
  j33_ctx->uvec = uprev;
  j33_ctx->vvec = vprev;
  PetscCall(MatMult(jac_ctx->submats[15], temp, yT3));

  //YT
  PetscCall(VecAXPY(yT0, 1.0, yT1));
  PetscCall(VecAXPY(yT0, 1.0, yT3));

  PetscCall(VecScatterBegin(scatter_u, yu0, Y, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(scatter_u, yu0, Y, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterBegin(scatter_v, yv1, Y, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(scatter_v, yv1, Y, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterBegin(scatter_omega, yomega0, Y, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(scatter_omega, yomega0, Y, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterBegin(scatter_temp, yT0, Y, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(scatter_temp, yT0, Y, INSERT_VALUES, SCATTER_REVERSE));

  PetscFunctionReturn(0);
}


//dfu/du
static PetscErrorCode MySubMatMult_J00(Mat J00, Vec X, Vec Y)
{
  const PetscScalar **x_array;
  SubmatrixContext *sub_ctx;
  PetscScalar **y_array;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hx, hy, dhx, dhy, hxdhy, hydhx;
  PetscScalar u, uxx, uyy;

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

  PetscCall(DMDAVecGetArrayRead(da, X, &x_array));
  PetscCall(DMDAVecGetArray(da, Y, &y_array));

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
      u         = x_array[j][i];
      uxx       = (2.0 * u - x_array[j][i-1] - x_array[j][i+1]) * hydhx;
      uyy       = (2.0 * u - x_array[j-1][i] - x_array[j+1][i]) * hxdhy;
      y_array[j][i] = uxx + uyy;
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(da, X, &x_array));
  PetscCall(DMDAVecRestoreArray(da, Y, &y_array));
  PetscFunctionReturn(0);
}

//dfu/domega
static PetscErrorCode MySubMatMult_J02(Mat J00, Vec X, Vec Y)
{
  const PetscScalar **x_array;
  SubmatrixContext *sub_ctx;
  PetscScalar **y_array;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hx, dhx;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J00, &sub_ctx));

  da = sub_ctx->dm;
  PetscCall(DMDAGetLocalInfo(da, &info));

  dhx   = (PetscReal)(info.mx - 1);
  hx    = 1.0 / dhx;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J00, &n, NULL));
  PetscCall(DMDAVecGetArrayRead(da, X, &x_array));
  PetscCall(DMDAVecGetArray(da, Y, &y_array));

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
  PetscCall(DMDAVecRestoreArrayRead(da, X, &x_array));
  PetscCall(DMDAVecRestoreArray(da, Y, &y_array));
  PetscFunctionReturn(0);
}

//dfv/domega
static PetscErrorCode MySubMatMult_J12(Mat J00, Vec X, Vec Y)
{
  const PetscScalar **x_array;
  SubmatrixContext *sub_ctx;
  PetscScalar **y_array;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hy, dhy;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J00, &sub_ctx));

  da = sub_ctx->dm;

  PetscCall(DMDAGetLocalInfo(da, &info));

  dhy   = (PetscReal)(info.my - 1);
  hy    = 1.0 / dhy;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J00, &n, NULL));
  PetscCall(DMDAVecGetArrayRead(da, X, &x_array));
  PetscCall(DMDAVecGetArray(da, Y, &y_array));

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
  PetscCall(DMDAVecRestoreArrayRead(da, X, &x_array));
  PetscCall(DMDAVecRestoreArray(da, Y, &y_array));
  PetscFunctionReturn(0);
}


static PetscErrorCode MySubMatMult_J20(Mat J00, Vec X, Vec Y)
{
  const PetscScalar **x_array, **omega_array, **u_array;
  SubmatrixContext *sub_ctx;
  PetscScalar **y_array;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec omega, uprev;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hy, dhy;
  PetscScalar vx;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J00, &sub_ctx));

  da = sub_ctx->dm;
  omega = sub_ctx->omega;
  uprev = sub_ctx->uvec;
  PetscCall(DMDAGetLocalInfo(da, &info));

  dhy   = (PetscReal)(info.my - 1);
  hy    = 1.0 / dhy;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J00, &n, NULL));

  PetscCall(DMDAVecGetArrayRead(da, X, &x_array));
  PetscCall(DMDAVecGetArrayRead(da, omega, &omega_array));
  PetscCall(DMDAVecGetArrayRead(da, uprev, &u_array));
  PetscCall(DMDAVecGetArray(da, Y, &y_array));

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
  PetscCall(DMDAVecRestoreArrayRead(da, X, &x_array));
  PetscCall(DMDAVecRestoreArrayRead(da, omega, &omega_array));
  PetscCall(DMDAVecRestoreArrayRead(da, uprev, &u_array));
  PetscCall(DMDAVecRestoreArray(da, Y, &y_array));
  PetscFunctionReturn(0);
}


static PetscErrorCode MySubMatMult_J21(Mat J00, Vec X, Vec Y)
{
  const PetscScalar **x_array, **omega_array, **v_array;
  SubmatrixContext *sub_ctx;
  PetscScalar **y_array;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec omega, vprev;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hx, dhx;
  PetscScalar vy;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J00, &sub_ctx));

  da = sub_ctx->dm;
  omega = sub_ctx->omega;
  vprev = sub_ctx->vvec;
  PetscCall(DMDAGetLocalInfo(da, &info));

  dhx   = (PetscReal)(info.mx - 1);
  hx    = 1.0 / dhx;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J00, &n, NULL));
  
  PetscCall(DMDAVecGetArrayRead(da, X, &x_array));
  PetscCall(DMDAVecGetArrayRead(da, omega, &omega_array));
  PetscCall(DMDAVecGetArrayRead(da, vprev, &v_array));
  PetscCall(DMDAVecGetArray(da, Y, &y_array));

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
  PetscCall(DMDAVecRestoreArrayRead(da, X, &x_array));
  PetscCall(DMDAVecRestoreArrayRead(da, omega, &omega_array));
  PetscCall(DMDAVecRestoreArrayRead(da, vprev, v_array));
  PetscCall(DMDAVecRestoreArray(da, Y, &y_array));
  PetscFunctionReturn(0);
}


static PetscErrorCode MySubMatMult_J22(Mat J00, Vec X, Vec Y)
{
  const PetscScalar **x_array, **u_array, **v_array;
  SubmatrixContext *sub_ctx;
  PetscScalar **y_array;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec uvec, vvec;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hx, hy, dhx, dhy, hxdhy, hydhx;
  PetscScalar uxx, uyy, vx, vy, avx, avy, vxp, vxm, vyp, vym;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J00, &sub_ctx));

  da = sub_ctx->dm;
  uvec = sub_ctx->uvec;
  vvec = sub_ctx->vvec;

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
  PetscCall(DMDAVecGetArrayRead(da, X, &x_array));
  PetscCall(DMDAVecGetArrayRead(da, uvec, &u_array));
  PetscCall(DMDAVecGetArrayRead(da, vvec, &v_array));
  PetscCall(DMDAVecGetArray(da, Y, &y_array));

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

  PetscCall(DMDAVecRestoreArrayRead(da, X, &x_array));
  PetscCall(DMDAVecRestoreArrayRead(da, uvec, &u_array));
  PetscCall(DMDAVecRestoreArrayRead(da, vvec, &v_array));
  PetscCall(DMDAVecRestoreArray(da, Y, &y_array));
  PetscFunctionReturn(0);
}


// domega/dT
static PetscErrorCode MySubMatMult_J23(Mat J00, Vec X, Vec Y)
{
  const PetscScalar **x_array;
  SubmatrixContext *sub_ctx;
  PetscScalar **y_array;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hy, dhy;
  PetscReal   grashof;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J00, &sub_ctx));

  da = sub_ctx->dm;
  PetscCall(DMDAGetLocalInfo(da, &info));

  grashof = 1.0;

  dhy   = (PetscReal)(info.my - 1);
  hy    = 1.0 / dhy;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J00, &n, NULL));

  PetscCall(DMDAVecGetArrayRead(da, X, &x_array));
  PetscCall(DMDAVecGetArray(da, Y, &y_array));
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
  PetscCall(DMDAVecRestoreArrayRead(da, X, &x_array));
  PetscCall(DMDAVecRestoreArray(da, Y, &y_array));
  PetscFunctionReturn(0);
}


// df(T)/du
static PetscErrorCode MySubMatMult_J30(Mat J00, Vec X, Vec Y)
{
  const PetscScalar **x_array, **temp_array;
  SubmatrixContext *sub_ctx;
  PetscScalar **y_array;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec temp;
  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hy, dhy;
  PetscReal   prandtl;
  PetscScalar vx, avx, vxp, vxm;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J00, &sub_ctx));

  da = sub_ctx->dm;
  temp = sub_ctx->temp;
  PetscCall(DMDAGetLocalInfo(da, &info));

  prandtl = 1.0;
  dhy   = (PetscReal)(info.my - 1);
  hy    = 1.0 / dhy;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J00, &n, NULL));
  PetscCall(DMDAVecGetArrayRead(da, X, &x_array));
  PetscCall(DMDAVecGetArrayRead(da, temp, &temp_array));
  PetscCall(DMDAVecGetArray(da, Y, &y_array));

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
      //edit this
      y_array[j][i] = prandtl * (vxp * (temp_array[j][i] - temp_array[j][i - 1]) + vxm * (temp_array[j][i+1] - temp_array[j][i])) * hy;
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(da, X, &x_array));
  PetscCall(DMDAVecRestoreArrayRead(da, temp, &temp_array));
  PetscCall(DMDAVecRestoreArray(da, Y, &y_array));
  PetscFunctionReturn(0);
}


static PetscErrorCode MySubMatMult_J31(Mat J00, Vec X, Vec Y)
{
  const PetscScalar **x_array, **temp_array, **v_array;
  SubmatrixContext *sub_ctx;
  PetscScalar **y_array;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec temp, vprev;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hx, dhx;
  PetscReal   prandtl;
  PetscScalar vy;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J00, &sub_ctx));

  da = sub_ctx->dm;
  temp = sub_ctx->temp;
  vprev = sub_ctx->vvec;
  PetscCall(DMDAGetLocalInfo(da, &info));

  prandtl = 1.0;

  dhx   = (PetscReal)(info.mx - 1);
  hx    = 1.0 / dhx;

  xints = info.xs;
  xinte = info.xs + info.xm;
  yints = info.ys;
  yinte = info.ys + info.ym;

  PetscCall(MatGetLocalSize(J00, &n, NULL));
  PetscCall(DMDAVecGetArrayRead(da, X, &x_array));
  PetscCall(DMDAVecGetArrayRead(da, vprev, &v_array));
  PetscCall(DMDAVecGetArrayRead(da, temp, &temp_array));
  PetscCall(DMDAVecGetArray(da, Y, &y_array));

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

  PetscCall(DMDAVecRestoreArrayRead(da, X, &x_array));
  PetscCall(DMDAVecRestoreArrayRead(da, temp, &temp_array));
  PetscCall(DMDAVecRestoreArrayRead(da, vprev, &v_array));
  PetscCall(DMDAVecRestoreArray(da, Y, &y_array));
  PetscFunctionReturn(0);
}


static PetscErrorCode MySubMatMult_J33(Mat J00, Vec X, Vec Y)
{
  const PetscScalar **x_array, **u_array, **v_array;
  SubmatrixContext *sub_ctx;
  PetscScalar **y_array;
  PetscInt    n;
  DM da;
  DMDALocalInfo info;
  Vec uvec, vvec;

  PetscInt    xints, xinte, yints, yinte, i, j;
  PetscReal   hx, hy, dhx, dhy, hxdhy, hydhx;
  PetscReal   prandtl;
  PetscScalar uxx, uyy, vx, vy, avx, avy, vxp, vxm, vyp, vym;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(J00, &sub_ctx));

  da = sub_ctx->dm;
  uvec = sub_ctx->uvec;
  vvec = sub_ctx->vvec;
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

  PetscCall(MatGetLocalSize(J00, &n, NULL));

  PetscCall(DMDAVecGetArrayRead(da, X, &x_array));
  PetscCall(DMDAVecGetArrayRead(da, uvec, &u_array));
  PetscCall(DMDAVecGetArrayRead(da, vvec, &v_array));
  PetscCall(DMDAVecGetArray(da, Y, &y_array));

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
  PetscCall(DMDAVecRestoreArrayRead(da, X, &x_array));
  PetscCall(DMDAVecRestoreArrayRead(da, uvec, &u_array));
  PetscCall(DMDAVecRestoreArrayRead(da, vvec, &v_array));
  PetscCall(DMDAVecRestoreArray(da, Y, &y_array));
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
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

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