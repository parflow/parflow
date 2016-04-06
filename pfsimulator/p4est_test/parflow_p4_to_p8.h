#ifndef PARFLOW_P4_TO_P8_H
#define PARFLOW_P4_TO_P8_H

#include <p4est_to_p8est.h>

#define parflow_p4est_grid_2d_t           parflow_p4est_grid_3d_t
#define parflow_p4est_quad_iter_2d_t      parflow_p4est_quad_iter_3d_t
#define parflow_p4est_ghost_iter_2d_t     parflow_p4est_ghost_iter_3d_t

#define parflow_p4est_grid_2d_new         parflow_p4est_grid_3d_new
#define parflow_p4est_grid_2d_destroy     parflow_p4est_grid_3d_destroy
#define parflow_p4est_qcoord_to_vertex_2d parflow_p4est_qcoord_to_vertex_3d

#endif /* !PARFLOW_P4_TO_P8_H */
