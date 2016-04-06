#include <parflow.h>
#include <parflow_p4est_2d.h>
#include <parflow_p4est_3d.h>

/*
 * A globals structure muss exist prior calling this function
*/
parflow_p4est_grid_t *
parflow_p4est_grid_new (int Px, int Py, int Pz)
{
  parflow_p4est_grid_t *pfgrid;

  if (Pz == 1) {
    pfgrid = parflow_p4est_grid_2d_new (Px, Py);
  }
  else {
    pfgrid = parflow_p4est_grid_3d_new (Px, Py, Pz);
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
    P4EST_ASSERT (dim == 3);
    parflow_p4est_grid_3d_destroy (pfgrid);
  }
}

void
parflow_p4est_qcoord_to_vertex (parflow_p4est_grid_t * pfgrid,
                                p4est_topidx_t treeid,
                                p4est_quadrant_t *quad, double v[3])
{
  int                 dim = PARFLOW_P4EST_GET_GRID_DIM (pfgrid);

  if (dim == 2) {
    parflow_p4est_qcoord_to_vertex_2d (pfgrid, treeid, quad, v);
  }
  else {
    P4EST_ASSERT (dim == 3);
    parflow_p4est_qcoord_to_vertex_3d (pfgrid, treeid, quad, v);
  }
}

int
parflow_p4est_quad_owner_rank (p4est_quadrant_t *quad)
{
  return quad->p.piggy1.owner_rank;
}
