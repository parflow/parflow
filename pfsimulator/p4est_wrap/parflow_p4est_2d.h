#ifndef PARLOW_P4EST_2D_H
#define PARLOW_P4EST_2D_H

#include "parflow_p4est.h"
#include <p4est_lnodes.h>
#include <p4est_extended.h>

SC_EXTERN_C_BEGIN;

typedef struct parflow_p4est_grid_2d {
  int dim;
  int initial_level;
  int Tx, Ty;                   /*Number of trees in each coordinate direction*/
  p4est_t        *forest;
  p4est_connectivity_t *connect;
  p4est_ghost_t  *ghost;
  sc_array_t     *ghost_data;

  p4est_mesh_t   *mesh;         /* Allocated only during ParFlow grid
                                 * initialization and destroyed inmediatly
                                 * afterwards */

  p4est_topidx_t *lexic_to_tree;   /* lexicographical order to brick order */
  p4est_topidx_t *tree_to_lexic;   /* brick order to lexicographical order */
} parflow_p4est_grid_2d_t;

typedef struct parflow_p4est_qiter_2d {
  /** Fields used by both types of iterators */
  parflow_p4est_iter_type_t itype;     /* Flag identifiying iterator type */
  p4est_connectivity_t *connect;
  p4est_topidx_t which_tree;           /* owner tree of the current quadrant */
  p4est_quadrant_t *quad;              /* current quadrant */
  int owner_rank;               /* processor owning current quadrant */
  int local_idx;                /* number of subgrid relative to the owner processor */

  /** Fields used only for local quadrant iterator */
  p4est_t        *forest;
  p4est_tree_t   *tree;
  sc_array_t     *tquadrants;
  int Q;                        /* quadrants in this tree */
  int q;                        /* index of current quad in this tree */

  /** Fields used only for ghost iterator */
  p4est_ghost_t  *ghost;
  sc_array_t     *ghost_layer;
  int G;                        /* ghosts quadrants in this layer */
  int g;                        /* index of current quad in this layer */
} parflow_p4est_qiter_2d_t;

parflow_p4est_grid_2d_t *parflow_p4est_grid_2d_new(int Px, int Py);

void            parflow_p4est_grid_2d_destroy(parflow_p4est_grid_2d_t *
                                              pfgrid);

void            parflow_p4est_grid_2d_mesh_init(parflow_p4est_grid_2d_t *
                                                pfgrid);

void            parflow_p4est_grid_2d_mesh_destroy(parflow_p4est_grid_2d_t *
                                                   pfgrid);

void            parflow_p4est_get_zneigh_2d(Subgrid * subgrid);

void            parflow_p4est_quad_to_vertex_2d(p4est_connectivity_t *
                                                  connect,
                                                  p4est_topidx_t treeid,
                                                  p4est_quadrant_t *
                                                  quad, double v[3]);

void            parflow_p4est_parent_to_vertex_2d(p4est_connectivity_t *
                                                  connect,
                                                  p4est_topidx_t treeid,
                                                  p4est_quadrant_t *
                                                  quad, double pv[3]);
parflow_p4est_qiter_2d_t
* parflow_p4est_qiter_init_2d(parflow_p4est_grid_2d_t * pfg,
                              parflow_p4est_iter_type_t itype);

int             parflow_p4est_qiter_isvalid_2d(parflow_p4est_qiter_2d_t *
                                               qit_2d);
parflow_p4est_qiter_2d_t
* parflow_p4est_qiter_next_2d(parflow_p4est_qiter_2d_t * qit_2d);

parflow_p4est_quad_data_t
* parflow_p4est_get_quad_data_2d(parflow_p4est_qiter_2d_t * qit_2d);

parflow_p4est_ghost_data_t
* parflow_p4est_get_ghost_data_2d(parflow_p4est_grid_2d_t *  pfg,
                                  parflow_p4est_qiter_2d_t * qit_2d);

void parflow_p4est_get_projection_info_2d(Subgrid *subgrid, int info[2]);

void            parflow_p4est_nquads_per_rank_2d(parflow_p4est_grid_2d_t *pfg,
                                                 int *                    quads_per_rank);

void            parflow_p4est_get_brick_coord_2d(Subgrid *                subgrid,
                                                 parflow_p4est_grid_2d_t *pfg,
                                                 p4est_gloidx_t           bcoord[3]);

int             parflow_p4est_check_neigh_2d(Subgrid *sfine, Subgrid *scoarse,
                                             parflow_p4est_grid_2d_t * pfg);
SC_EXTERN_C_END;

#endif                          /* !PARLOW_P4EST_2D_H */
