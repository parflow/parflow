#include <parflow.h>
#include <parflow_p4est_2d.h>
#include <parflow_p4est_3d.h>
#include <sc_functions.h>
#include <sc_containers.h>

struct parflow_p4est_grid {

    int             dim;
    union {
        parflow_p4est_grid_2d_t *p4;
        parflow_p4est_grid_3d_t *p8;
    } p;

};

struct parflow_p4est_qiter {

    int             dim;
    union {
        parflow_p4est_qiter_2d_t *qiter_2d;
        parflow_p4est_qiter_3d_t *qiter_3d;
    } q;

};

/*
 * Accesor macros
 */
#define PARFLOW_P4EST_GET_GRID_DIM(pfgrid) ((pfgrid)->dim)
#define PARFLOW_P4EST_GET_QITER_DIM(qiter) ((qiter)->dim)

/*
 * A globals structure muss exist prior calling this function
 */
parflow_p4est_grid_t *
parflow_p4est_grid_new(int Px, int Py, int Pz)
{
    parflow_p4est_grid_t *pfgrid;

    pfgrid = P4EST_ALLOC_ZERO(parflow_p4est_grid_t, 1);
    if (Pz == 1) {
        pfgrid->dim = 2;
        pfgrid->p.p4 = parflow_p4est_grid_2d_new(Px, Py);
    } else {
        pfgrid->dim = 3;
        pfgrid->p.p8 = parflow_p4est_grid_3d_new(Px, Py, Pz);
    }
    return pfgrid;
}

void
parflow_p4est_grid_destroy(parflow_p4est_grid_t * pfgrid)
{
    int             dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

    if (dim == 2) {
        parflow_p4est_grid_2d_destroy(pfgrid->p.p4);
    } else {
        P4EST_ASSERT(dim == 3);
        parflow_p4est_grid_3d_destroy(pfgrid->p.p8);
    }
    P4EST_FREE(pfgrid);
}

parflow_p4est_qiter_t *
parflow_p4est_qiter_init(parflow_p4est_grid_t * pfgrid,
                         parflow_p4est_iter_type_t itype)
{
    parflow_p4est_qiter_t *qiter = NULL;
    parflow_p4est_qiter_2d_t *qit_2d;
    parflow_p4est_qiter_3d_t *qit_3d;
    int             dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

    if (dim == 2) {
        qit_2d = parflow_p4est_qiter_init_2d(pfgrid->p.p4, itype);
        if (qit_2d) {
            qiter = P4EST_ALLOC(parflow_p4est_qiter_t, 1);
            qiter->dim = dim;
            qiter->q.qiter_2d = qit_2d;
        }
    } else {
        P4EST_ASSERT(dim == 3);
        qit_3d = parflow_p4est_qiter_init_3d(pfgrid->p.p8, itype);
        if (qit_3d) {
            qiter = P4EST_ALLOC(parflow_p4est_qiter_t, 1);
            qiter->dim = dim;
            qiter->q.qiter_3d = qit_3d;
        }
    }

    return qiter;
}

parflow_p4est_qiter_t *
parflow_p4est_qiter_next(parflow_p4est_qiter_t * qiter)
{
    int             dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

    if (dim == 2) {
        if (!parflow_p4est_qiter_next_2d(qiter->q.qiter_2d)) {
            P4EST_FREE(qiter);
            return NULL;
        }
    } else {
        P4EST_ASSERT(dim == 3);
        if (!parflow_p4est_qiter_next_3d(qiter->q.qiter_3d)) {
            P4EST_FREE(qiter);
            return NULL;
        }
    }

    P4EST_ASSERT(qiter);
    return qiter;
}

void
parflow_p4est_qiter_qcorner(parflow_p4est_qiter_t * qiter, double v[3])
{
    int             k;
    int             dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);
    int             level;

    if (dim == 2) {
        parflow_p4est_qcoord_to_vertex_2d(qiter->q.qiter_2d->connect,
                                          qiter->q.qiter_2d->which_tree,
                                          qiter->q.qiter_2d->quad, v);
        level = qiter->q.qiter_2d->quad->level;
    } else {
        P4EST_ASSERT(dim == 3);
        parflow_p4est_qcoord_to_vertex_3d(qiter->q.qiter_3d->connect,
                                          qiter->q.qiter_3d->which_tree,
                                          qiter->q.qiter_3d->quad, v);
        level = qiter->q.qiter_3d->quad->level;
    }

    for (k = 0; k < 3; ++k) {
        v[k] *= sc_intpow(2, level);
    }
}

parflow_p4est_quad_data_t *
parflow_p4est_qiter_get_data(parflow_p4est_qiter_t * qiter)
{
    int             dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

    if (dim == 2) {
        return parflow_p4est_qiter_get_data_2d(qiter->q.qiter_2d);
    } else {
        P4EST_ASSERT(dim == 3);
        return parflow_p4est_qiter_get_data_3d(qiter->q.qiter_3d);
    }
}

parflow_p4est_ghost_data_t *
parflow_p4est_get_ghost_data(parflow_p4est_grid_t * pfgrid)
{
    int             dim = PARFLOW_P4EST_GET_QITER_DIM(pfgrid);

    if (dim == 2) {
        return pfgrid->p.p4->ghost_data;
    } else {
        P4EST_ASSERT(dim == 3);
        return pfgrid->p.p8->ghost_data;
    }
}

int
parflow_p4est_qiter_get_owner_rank(parflow_p4est_qiter_t * qiter)
{
    int             dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

    if (dim == 2) {
        return qiter->q.qiter_2d->owner_rank;
    } else {
        P4EST_ASSERT(dim == 3);
        return qiter->q.qiter_3d->owner_rank;
    }
}

int
parflow_p4est_qiter_get_ghost_idx(parflow_p4est_qiter_t * qiter)
{
    int             g;
    int             dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

    if (dim == 2) {
        g = qiter->q.qiter_2d->g;
    } else {
        P4EST_ASSERT(dim == 3);
        g = qiter->q.qiter_3d->g;
    }

    P4EST_ASSERT(g >= 0);

    return g;
}
