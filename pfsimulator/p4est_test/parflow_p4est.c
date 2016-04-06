#include <parflow.h>
#include <parflow_p4est_2d.h>
#include <parflow_p4est_3d.h>

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
parflow_p4est_qiter_init(parflow_p4est_grid_t * pfgrid)
{
    parflow_p4est_qiter_t *qiter;
    int             dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

    qiter = P4EST_ALLOC(parflow_p4est_qiter_t, 1);
    qiter->dim = dim;
    if (dim == 2) {
        qiter->q.qiter_2d = parflow_p4est_qiter_init_2d(pfgrid->p.p4);
    } else {
        P4EST_ASSERT(dim == 3);
        qiter->q.qiter_3d = parflow_p4est_qiter_init_3d(pfgrid->p.p8);
    }

    return qiter;
}

int
parflow_p4est_qiter_isvalid(parflow_p4est_qiter_t * qiter)
{
    int             dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

    if (dim == 2) {
        return parflow_p4est_qiter_isvalid_2d(qiter->q.qiter_2d);
    } else {
        P4EST_ASSERT(dim == 3);
        return parflow_p4est_qiter_isvalid_3d(qiter->q.qiter_3d);
    }
}

void
parflow_p4est_qiter_next(parflow_p4est_qiter_t * qiter)
{
    int             dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

    if (dim == 2) {
        parflow_p4est_qiter_next_2d(qiter->q.qiter_2d);
    } else {
        P4EST_ASSERT(dim == 3);
        parflow_p4est_qiter_next_3d(qiter->q.qiter_3d);
    }
}

void
parflow_p4est_qiter_destroy(parflow_p4est_qiter_t * qiter)
{
    int             dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

    if (dim == 2) {
        parflow_p4est_qiter_destroy_2d(qiter->q.qiter_2d);
    } else {
        P4EST_ASSERT(dim == 3);
        parflow_p4est_qiter_destroy_3d(qiter->q.qiter_3d);
    }
    P4EST_FREE(qiter);
}

void
parflow_p4est_qiter_qcorner(parflow_p4est_qiter_t * qiter, double v[3])
{
    int             k;
    int             dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);
    double          level_factor;

    if (dim == 2) {
        parflow_p4est_qcoord_to_vertex_2d(qiter->q.qiter_2d->connect,
                                          qiter->q.qiter_2d->tt,
                                          qiter->q.qiter_2d->quad, v);
        level_factor = qiter->q.qiter_2d->quad->level;
    } else {
        P4EST_ASSERT(dim == 3);
        parflow_p4est_qcoord_to_vertex_3d(qiter->q.qiter_3d->connect,
                                          qiter->q.qiter_3d->tt,
                                          qiter->q.qiter_3d->quad, v);
        level_factor = qiter->q.qiter_3d->quad->level;
    }

    for (k = 0; k < 3; ++k) {
        v[k] *= level_factor;
    }
}

#if 0
p4est_topidx_t
parflow_p4est_gquad_owner_tree(p4est_quadrant_t * quad)
{
    return quad->p.piggy3.which_tree;
}
#endif
