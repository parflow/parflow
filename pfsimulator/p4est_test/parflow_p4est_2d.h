#ifndef PARLOW_P4EST_2D_H
#define PARLOW_P4EST_2D_H

#include "parflow_p4est.h"
#include <p4est_lnodes.h>

typedef struct parflow_p4est_grid_2d
{
  int                 dim;
  p4est_t            *forest;
  p4est_connectivity_t *connect;
  p4est_ghost_t      *ghost;
}
parflow_p4est_grid_2d_t;

parflow_p4est_grid_2d_t *parflow_p4est_grid_2d_new (int Px, int Py);

void                parflow_p4est_grid_2d_destroy (parflow_p4est_grid_2d_t *
                                                   pfgrid);

void                parflow_p4est_qcoord_to_vertex_2d (parflow_p4est_grid_t *
                                                       pfgrid,
                                                       p4est_topidx_t treeid,
                                                       p4est_quadrant_t *
                                                       quad, double v[3]);

#endif // !PARLOW_P4EST_2D_H
