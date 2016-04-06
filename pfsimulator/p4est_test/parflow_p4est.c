#include <parflow.h>
#include <parflow_p4est_2d.h>
#include <parflow_p4est_3d.h>

typedef struct parflow_p4est_grid
{
  int                 dim;
  union
  {
    parflow_p4est_grid_2d_t  *p4;
    parflow_p4est_grid_3d_t  *p8;
  }p;
}
parflow_p4est_grid_t;

/*
 * A globals structure muss exist prior calling this function
*/
parflow_p4est_grid_t *
parflow_p4est_grid_new (int Px, int Py, int Pz)
{
  parflow_p4est_grid_t *pfgrid;

  pfgrid = P4EST_ALLOC_ZERO (parflow_p4est_grid_t, 1);
  if (Pz == 1) {
    pfgrid->dim = 2;
    pfgrid->p.p4 = parflow_p4est_grid_2d_new (Px, Py);
  }
  else {
    pfgrid->dim = 3;
    pfgrid->p.p8 = parflow_p4est_grid_3d_new (Px, Py, Pz);
  }
  return pfgrid;
}

void
parflow_p4est_grid_destroy (parflow_p4est_grid_t * pfgrid)
{
  int                 dim = PARFLOW_P4EST_GET_GRID_DIM (pfgrid);

  if (dim == 2) {
    parflow_p4est_grid_2d_destroy (pfgrid->p.p4);
  }
  else {
    P4EST_ASSERT (dim == 3);
    parflow_p4est_grid_3d_destroy (pfgrid->p.p8);
  }
  P4EST_FREE (pfgrid);
}

#if 0
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

p4est_topidx_t
parflow_p4est_gquad_owner_tree (p4est_quadrant_t   *quad){
  return quad->p.piggy3.which_tree;
}
#endif
