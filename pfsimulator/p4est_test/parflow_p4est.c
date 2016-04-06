#include <parflow.h>
#include <parflow_p4est_2d.h>
#include <parflow_p4est_3d.h>

/*A globals structure muss exist prior calling this function */
parflow_p4est_grid_t *
parflow_p4est_grid_new ()
{
  parflow_p4est_grid_t *pfgrid;
  int                 NX, NY, NZ;

  pfgrid = NULL;
  NX = GetIntDefault ("ComputationalGrid.NX", 1);
  NY = GetIntDefault ("ComputationalGrid.NY", 1);
  NZ = GetIntDefault ("ComputationalGrid.NZ", 1);

  if (NZ == 1) {
    pfgrid = parflow_p4est_grid_2d_new (NX, NY);
  }
  else {
    pfgrid = parflow_p4est_grid_3d_new (NX, NY, NZ);
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
