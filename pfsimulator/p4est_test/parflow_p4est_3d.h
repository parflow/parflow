#ifndef PARLOW_P4EST_3D_H
#define PARLOW_P4EST_3D_H

#include "parflow_p4est.h"

parflow_p4est_grid_t *parflow_p4est_grid_3d_new (int NX, int NY, int NZ);

void
               parflow_p4est_grid_3d_destroy (parflow_p4est_grid_t * pfgrid);

#endif // !PARLOW_P4EST_3D_H
