#ifndef PARLOW_P4EST_3D_H
#define PARLOW_P4EST_3D_H

#include "parflow_p4est.h"
#include <p8est_lnodes.h>
#include <p8est_extended.h>

SC_EXTERN_C_BEGIN;

typedef struct parflow_p4est_grid_3d {

    int             dim;
    int             Tx,Ty,Tz;    /*Number of trees in each coordinate direction*/
    p8est_t        *forest;
    p8est_connectivity_t *connect;
    p8est_ghost_t  *ghost;
    sc_array_t     *ghost_data;

    p8est_mesh_t   *mesh;       /* Allocated only during ParFlow grid
                                   initialization and destroyed inmediatly
                                   afterwards */
} parflow_p4est_grid_3d_t;

typedef struct parflow_p4est_qiter_3d {

    /** Fields used by both types of iterators */
    parflow_p4est_iter_type_t itype; /* Flag identifiying iterator type*/
    p8est_connectivity_t *connect;
    p4est_topidx_t  which_tree;     /* owner tree of the current quadrant */
    p8est_quadrant_t *quad;         /* current quadrant */
    int             owner_rank; /* processor owning current quadrant */
    int             local_idx;  /* number of subgrid relative to the owner processor */

    /** Fields used only for quadrant iterator */
    p8est_t        *forest;
    p8est_tree_t   *tree;
    sc_array_t     *tquadrants;
    int             Q;          /* quadrants in this tree */
    int             q;          /* index of current quad in this tree */

    /** Fields used only for ghost iterator */
    p8est_ghost_t  *ghost;
    sc_array_t     *ghost_layer;
    int             G;          /* ghosts quadrants in this layer */
    int             g;          /* index of current quad in this layer */

} parflow_p4est_qiter_3d_t;

parflow_p4est_grid_3d_t *parflow_p4est_grid_3d_new(int Px, int Py, int Pz);

void            parflow_p4est_grid_3d_destroy(parflow_p4est_grid_3d_t *
                                              pfgrid);

void            parflow_p4est_grid_3d_mesh_init(parflow_p4est_grid_3d_t *
                                                pfgrid);

void            parflow_p4est_grid_3d_mesh_destroy(parflow_p4est_grid_3d_t *
                                                   pfgrid);


void            parflow_p4est_get_zneigh_3d(Subgrid * subgrid,
                                            parflow_p4est_qiter_3d_t * qiter,
                                            parflow_p4est_grid_3d_t * pfgrid);

void            parflow_p4est_qcoord_to_vertex_3d(p8est_connectivity_t *
                                                  connect,
                                                  p4est_topidx_t treeid,
                                                  p8est_quadrant_t *
                                                  quad, double v[3]);

parflow_p4est_qiter_3d_t
    * parflow_p4est_qiter_init_3d(parflow_p4est_grid_3d_t * pfg,
                                  parflow_p4est_iter_type_t itype);

int             parflow_p4est_qiter_isvalid_3d(parflow_p4est_qiter_3d_t *
                                               qit_3d);

parflow_p4est_qiter_3d_t
    * parflow_p4est_qiter_next_3d(parflow_p4est_qiter_3d_t * qit_3d);

parflow_p4est_quad_data_t
    * parflow_p4est_get_quad_data_3d(parflow_p4est_qiter_3d_t * qit_3d);


parflow_p4est_ghost_data_t
    * parflow_p4est_get_ghost_data_3d(parflow_p4est_grid_3d_t *pfg,
                                      parflow_p4est_qiter_3d_t * qit_3d);

SC_EXTERN_C_END;

#endif                          /* !PARLOW_P4EST_3D_H */
