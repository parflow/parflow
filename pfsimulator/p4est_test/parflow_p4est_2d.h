#ifndef PARLOW_P4EST_2D_H
#define PARLOW_P4EST_2D_H

#include "parflow_p4est.h"
#include <p4est_lnodes.h>

typedef struct parflow_p4est_grid_2d {

    int             dim;
    p4est_t        *forest;
    p4est_connectivity_t *connect;
    p4est_ghost_t  *ghost;

} parflow_p4est_grid_2d_t;

typedef struct parflow_p4est_qiter_2d {

    p4est_t        *forest;
    p4est_connectivity_t *connect;
    p4est_topidx_t  tt;
    p4est_tree_t   *tree;
    sc_array_t     *tquadrants;
    int             owner_rank; /* processor owning current quadrant */
    int             Q;          /* quadrants in this tree */
    int             q;          /* index of current quad in this tree */
    p4est_quadrant_t *quad;     /* current quadrant */

} parflow_p4est_qiter_2d_t;

typedef struct parflow_p4est_giter_2d {

    p4est_ghost_t  *ghost;
    sc_array_t     *ghost_layer;
    p4est_connectivity_t *connect;
    int             owner_rank; /* processor owning current quadrant */
    int             G;          /* ghosts quadrants in this layer */
    int             g;          /* index of current quad in this layer */
    p4est_quadrant_t *quad;     /* current quadrant */

} parflow_p4est_giter_2d_t;

parflow_p4est_grid_2d_t *parflow_p4est_grid_2d_new(int Px, int Py);

void            parflow_p4est_grid_2d_destroy(parflow_p4est_grid_2d_t *
                                              pfgrid);

void            parflow_p4est_qcoord_to_vertex_2d(p4est_connectivity_t *
                                                  connect,
                                                  p4est_topidx_t treeid,
                                                  p4est_quadrant_t *
                                                  quad, double v[3]);

parflow_p4est_qiter_2d_t
    * parflow_p4est_qiter_init_2d(parflow_p4est_grid_2d_t * pfg);

int             parflow_p4est_qiter_isvalid_2d(parflow_p4est_qiter_2d_t *
                                               qit_2d);

void            parflow_p4est_qiter_next_2d(parflow_p4est_qiter_2d_t *
                                            qit_2d);

void            parflow_p4est_qiter_destroy_2d(parflow_p4est_qiter_2d_t *
                                               qit_2d);

#endif                          // !PARLOW_P4EST_2D_H
