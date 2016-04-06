#include <parflow.h>
#include <parflow_p4est_2d.h>
#include <parflow_p4est_3d.h>

/*
 * A globals structure muss exist prior calling this function
*/
parflow_p4est_grid_t *
parflow_p4est_grid_new (int nx, int ny, int nz)
{
  parflow_p4est_grid_t *pfgrid;

  if (nz == 1) {
    pfgrid = parflow_p4est_grid_2d_new (nx, ny);
  }
  else {
    pfgrid = parflow_p4est_grid_3d_new (nx, ny, nz);
  }
  return pfgrid;
}

void
parflow_p4est_grid_destroy (parflow_p4est_grid_t * pfgrid)
{
  int                 dim = PARFLOW_P4EST_GET_GRID_DIM (pfgrid);

  if (dim == 2) {
    parflow_p4est_grid_2d_destroy (pfgrid);
  }
  else {
    parflow_p4est_grid_3d_destroy (pfgrid);
  }
}
