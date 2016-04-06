#ifndef PARLOW_P4EST_3D_H
#define PARLOW_P4EST_3D_H

#include "parflow_p4est.h"
#include <p8est_lnodes.h>

typedef struct parflow_p4est_grid_3d
{
  int                 dim;
  p8est_t            *forest;
  p8est_connectivity_t *connect;
  p8est_ghost_t      *ghost;
}
parflow_p4est_grid_3d_t;

parflow_p4est_grid_3d_t *parflow_p4est_grid_3d_new (int Px, int Py, int Pz);

void                parflow_p4est_grid_3d_destroy (parflow_p4est_grid_3d_t *
                                                   pfgrid);

void                parflow_p4est_qcoord_to_vertex_3d (parflow_p4est_grid_t *
                                                       pfgrid,
                                                       p4est_topidx_t treeid,
                                                       p4est_quadrant_t *
                                                       quad, double v[3]);
#endif // !PARLOW_P4EST_3D_H
