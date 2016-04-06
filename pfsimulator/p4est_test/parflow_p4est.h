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
parflow_p4est_grid_t *parflow_p4est_grid_new (int nx, int ny, int nz);

void                parflow_p4est_grid_destroy (parflow_p4est_grid_t *
                                                pfgrid);

#endif // !PARLOW_P4EST_H
