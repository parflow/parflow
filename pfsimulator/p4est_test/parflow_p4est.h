#ifndef PARLOW_P4EST_H
#define PARLOW_P4EST_H

#include <parflow.h>
#include <sc_containers.h>

typedef Subregion Subgrid;
typedef struct parflow_p4est_grid parflow_p4est_grid_t;
typedef struct parflow_p4est_qiter parflow_p4est_qiter_t;

typedef enum parflow_p4est_iter_type {
    PARFLOW_P4EST_QUAD = 0x01,
    PARFLOW_P4EST_GHOST = 0x02
} parflow_p4est_iter_type_t;

typedef struct parflow_p4est_quad_data {

    Subgrid *pf_subgrid;

} parflow_p4est_quad_data_t;

typedef parflow_p4est_quad_data_t parflow_p4est_ghost_data_t;

/*
typedef struct parflow_p4est_ghost_data {

    sc_array_t     *ghost_subgrids;

} parflow_p4est_ghost_data_t;
*/

/*
 * Functions
 */
parflow_p4est_grid_t *parflow_p4est_grid_new(int Px, int Py, int Pz);

void            parflow_p4est_grid_destroy(parflow_p4est_grid_t * pfgrid);

parflow_p4est_qiter_t *parflow_p4est_qiter_init(parflow_p4est_grid_t *
                                                pfg,
                                                parflow_p4est_iter_type_t
                                                itype);

parflow_p4est_qiter_t *parflow_p4est_qiter_next(parflow_p4est_qiter_t *
                                                qiter);

void            parflow_p4est_qiter_qcorner(parflow_p4est_qiter_t * qiter,
                                            double v[3]);

parflow_p4est_quad_data_t
    * parflow_p4est_get_quad_data(parflow_p4est_qiter_t * qiter);

int             parflow_p4est_qiter_get_owner_rank(parflow_p4est_qiter_t *
                                                   qiter);

parflow_p4est_ghost_data_t
   * parflow_p4est_get_ghost_data(parflow_p4est_grid_t * pfgrid,
                                  parflow_p4est_qiter_t * qiter);

#endif                          // !PARLOW_P4EST_H
