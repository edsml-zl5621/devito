#define _POSIX_C_SOURCE 200809L

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

PetscErrorCode MyMatShellMult(Mat A_matfree, Vec xvec, Vec yvec);

int Kernel(struct dataobj *restrict pn_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, const float dt, const float h_x, const float h_y, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, struct MatContext * ctx)
{
  Mat A_matfree;

  float (*restrict pn)[pn_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[pn_vec->size[1]]) pn_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  float (*restrict v)[v_vec->size[1]][v_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[v_vec->size[1]][v_vec->size[2]]) v_vec->data;

  PetscCall(MatShellSetContext(A_matfree,ctx));
  PetscCall(MatShellSetOperation(A_matfree,MATOP_MULT,(void (*)(void))MyMatShellMult));
  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        u[t1][x + 2][y + 2] = dt*(-5.0e-1F*pn[x + 1][y + 2]/h_x + 5.0e-1F*pn[x + 3][y + 2]/h_x - (-5.0e-1F*u[t0][x + 2][y + 1]/h_y + 5.0e-1F*u[t0][x + 2][y + 3]/h_y)*v[t0][x + 2][y + 2] + u[t0][x + 2][y + 2]/dt);

        v[t1][x + 2][y + 2] = dt*(-(-5.0e-1F*v[t0][x + 1][y + 2]/h_x + 5.0e-1F*v[t0][x + 3][y + 2]/h_x)*u[t0][x + 2][y + 2] - 5.0e-1F*pn[x + 2][y + 1]/h_y + 5.0e-1F*pn[x + 2][y + 3]/h_y + v[t0][x + 2][y + 2]/dt);
      }
    }
  }

  return 0;
}

PetscErrorCode MyMatShellMult(Mat A_matfree, Vec xvec, Vec yvec)
{
  PetscScalar**restrict xvec_tmp;
  PetscScalar**restrict yvec_tmp;

  struct MatContext * ctx;
  PetscCall(MatShellGetContext(A_matfree,&ctx));

  for (int x = ctx->x_m; x <= ctx->x_M; x += 1)
  {
    for (int y = ctx->y_m; y <= ctx->y_M; y += 1)
    {
      yvec_tmp[x][y] = pow(ctx->h_x, -2)*xvec_tmp[x + 1][y + 2] - 2.0F*pow(ctx->h_x, -2)*xvec_tmp[x + 2][y + 2] + pow(ctx->h_x, -2)*xvec_tmp[x + 3][y + 2] + pow(ctx->h_y, -2)*xvec_tmp[x + 2][y + 1] - 2.0F*pow(ctx->h_y, -2)*xvec_tmp[x + 2][y + 2] + pow(ctx->h_y, -2)*xvec_tmp[x + 2][y + 3];
    }
  }
}
