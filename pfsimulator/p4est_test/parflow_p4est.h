#ifndef PARLOW_P4EST_H
#define PARLOW_P4EST_H

typedef struct parflow_p4est_grid parflow_p4est_grid_t;
typedef struct parflow_p4est_qiter parflow_p4est_qiter_t;

typedef enum parflow_p4est_iter_type {
    PARFLOW_P4EST_QUAD,
    PARFLOW_P4EST_GHOST
} parflow_p4est_iter_type_t;

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

void            parflow_p4est_qiter_set_data(parflow_p4est_qiter_t *
                                             qiter, void *user_data);

void           *parflow_p4est_qiter_get_data(parflow_p4est_qiter_t *
                                             qiter);

int             parflow_p4est_qiter_get_owner_rank(parflow_p4est_qiter_t *
                                                   qiter);

#endif                          // !PARLOW_P4EST_H
