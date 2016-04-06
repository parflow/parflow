#ifndef PARLOW_P4EST_H
#define PARLOW_P4EST_H

#ifndef P4_TO_P8
#include <p4est.h>
#include <p4est_connectivity.h>
#else
#include <p8est.h>
#include <p8est_connectivity.h>
#endif

typedef struct parflow_p4est_grid
{
  int                 dim;
  p4est_t            *forest;
  p4est_connectivity_t *connect;
}
parflow_p4est_grid_t;

/*Accesor macros*/
#define PARFLOW_P4EST_GET_GRID_DIM(pfgrid) ((pfgrid)->dim)

/*Functions*/
parflow_p4est_grid_t *parflow_p4est_grid_new (int Px, int Py, int Pz);

void                parflow_p4est_grid_destroy (parflow_p4est_grid_t *
                                                pfgrid);

void                parflow_p4est_qcoord_to_vertex (parflow_p4est_grid_t *
                                                    pfgrid,
                                                    p4est_topidx_t treeid,
                                                    p4est_quadrant_t * quad,
                                                    double v[3]);

int                 parflow_p4est_quad_owner_rank (p4est_quadrant_t *quad);

#endif // !PARLOW_P4EST_H
