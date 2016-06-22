#ifndef PARFLOW_P4_TO_P8_H
#define PARFLOW_P4_TO_P8_H

#include <p4est_to_p8est.h>

#define parflow_p4est_grid_2d_t           parflow_p4est_grid_3d_t
#define parflow_p4est_qiter_2d_t          parflow_p4est_qiter_3d_t
#define parflow_p4est_giter_2d_t          parflow_p4est_giter_3d_t

#define parflow_p4est_grid_2d_new           parflow_p4est_grid_3d_new
#define parflow_p4est_grid_2d_destroy       parflow_p4est_grid_3d_destroy
#define parflow_p4est_grid_2d_mesh_init     parflow_p4est_grid_3d_mesh_init
#define parflow_p4est_grid_2d_mesh_destroy  parflow_p4est_grid_3d_mesh_destroy
#define parflow_p4est_get_zneigh_2d         parflow_p4est_get_zneigh_3d
#define parflow_p4est_qcoord_to_vertex_2d   parflow_p4est_qcoord_to_vertex_3d

#define parflow_p4est_qiter_init_2d       parflow_p4est_qiter_init_3d
#define parflow_p4est_qiter_isvalid_2d    parflow_p4est_qiter_isvalid_3d
#define parflow_p4est_qiter_next_2d       parflow_p4est_qiter_next_3d
#define parflow_p4est_qiter_destroy_2d    parflow_p4est_qiter_destroy_3d
#define parflow_p4est_get_quad_data_2d    parflow_p4est_get_quad_data_3d
#define parflow_p4est_get_ghost_data_2d   parflow_p4est_get_ghost_data_3d
#define parflow_p4est_get_projection_owner_2d parflow_p4est_get_projection_owner_3d

#endif                          /* !PARFLOW_P4_TO_P8_H */
