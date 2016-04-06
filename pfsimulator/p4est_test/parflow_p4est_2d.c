#include <parflow.h>
#include "parflow_p4est_math.h"
#ifndef P4_TO_P8
#include "parflow_p4est_2d.h"
#include <p4est_vtk.h>
#else
#include "parflow_p4est_3d.h"
#include <p8est_vtk.h>
#endif

static int
parflow_p4est_refine_fn(p4est_t * p4est, p4est_topidx_t which_tree,
                        p4est_quadrant_t * quadrant)
{
    return 1;
}

parflow_p4est_grid_2d_t *
parflow_p4est_grid_2d_new(int Px, int Py
#ifdef P4_TO_P8
                          , int Pz
#endif
    )
{
    int             g, gt;
    int             tx, ty;
#ifdef P4_TO_P8
    int             tz;
#endif
    int             level, refine_level, balance;
    parflow_p4est_grid_2d_t *pfg;

    pfg = P4EST_ALLOC_ZERO(parflow_p4est_grid_2d_t, 1);
    tx = pfmax(Px, 1);
    ty = pfmax(Py, 1);
    gt = gcd(tx, ty);
#ifdef P4_TO_P8
    tz = pfmax(Pz, 1);
    gt = gcd(gt, tz);
#endif
    g = powtwo_div(gt);

    refine_level = (int) log2((double) g);
    balance = refine_level > 0;

    /*
     * Create connectivity structure
     */
    pfg->connect = p4est_connectivity_new_brick(tx / g, ty / g
#ifdef P4_TO_P8
                                                , tz / g, 0
#endif
                                                , 0, 0);

    pfg->forest = p4est_new(amps_CommWorld, pfg->connect, 0, NULL, NULL);

    /*
     * Refine to get a grid with same number of elements as parflow
     * old Grid structure and resdistribute quadrants among mpi_comm
     */
    for (level = 0; level < refine_level; ++level) {
        p4est_refine(pfg->forest, 0, parflow_p4est_refine_fn, NULL);
        p4est_partition(pfg->forest, 0, NULL);
    }

    /*
     * After refine, call 2:1 balance and redistribute new quadrants
     * among the mpi communicator
     */
    if (balance) {
        p4est_balance(pfg->forest, P4EST_CONNECT_FACE, NULL);
        p4est_partition(pfg->forest, 0, NULL);
    }

    /*
     * allocate ghost storage 
     */
    pfg->ghost = p4est_ghost_new(pfg->forest, P4EST_CONNECT_FACE);

    // p4est_vtk_write_file (pfg->forest, NULL, P4EST_STRING "_pfbrick");

    return pfg;
}

void
parflow_p4est_grid_2d_destroy(parflow_p4est_grid_2d_t * pfg)
{
    p4est_destroy(pfg->forest);
    p4est_connectivity_destroy(pfg->connect);
    p4est_ghost_destroy(pfg->ghost);
    P4EST_FREE(pfg);
}

/* START: Quadrant iterator routines */

void
parflow_p4est_qiter_init_2d(parflow_p4est_qiter_2d_t * qiter,
                            parflow_p4est_grid_2d_t * pfg)
{
    memset(qiter, 0, sizeof(parflow_p4est_qiter_2d_t));

    qiter->forest = pfg->forest;
    qiter->tt = qiter->forest->first_local_tree;
    if (qiter->tt <= qiter->forest->last_local_tree) {
        P4EST_ASSERT(qiter->tt >= 0);
        qiter->tree =
            p4est_tree_array_index(qiter->forest->trees, qiter->tt);
        qiter->tquadrants = &qiter->tree->quadrants;
        qiter->Q = (int) qiter->tquadrants->elem_count;
        P4EST_ASSERT(qiter->Q > 0);
        qiter->quad = p4est_quadrant_array_index(qiter->tquadrants,
                                                 (size_t) qiter->q);
    }
}

int
parflow_p4est_qiter_isvalid_2d(parflow_p4est_qiter_2d_t * qiter)
{
    return (qiter->q < qiter->Q);
}

void
parflow_p4est_qiter_next_2d(parflow_p4est_qiter_2d_t * qiter)
{

    P4EST_ASSERT(parflow_p4est_quad_iter_isvalid(qiter));

    if (++qiter->q == qiter->Q) {
        if (++qiter->tt <= qiter->forest->last_local_tree) {
            qiter->tree =
                p4est_tree_array_index(qiter->forest->trees, qiter->tt);
            qiter->tquadrants = &qiter->tree->quadrants;
            qiter->Q = (int) qiter->tquadrants->elem_count;
            qiter->q = 0;
            qiter->quad = p4est_quadrant_array_index(qiter->tquadrants,
                                                     (size_t) qiter->q);
        } else {
            memset(qiter, 0, sizeof(parflow_p4est_qiter_2d_t));
            return;
        }
    } else {
        qiter->quad =
            p4est_quadrant_array_index(qiter->tquadrants, qiter->q);
    }

    P4EST_ASSERT(parflow_p4est_qiter_isvalid(qiter));
}
/* END: Quadrant iterator routines */


/* START: Ghost iterator routines */
void
parflow_p4est_giter_init_2d(parflow_p4est_giter_2d_t * giter,
                            parflow_p4est_grid_2d_t * pfg)
{
    memset(qiter, 0, sizeof(parflow_p4est_giter_2d_t));

    giter->ghost = pfg->ghost;
    giter->ghost_layer = &giter->ghost->ghosts;
    giter->G = (int) giter->ghost_layer->elem_count;
    P4EST_ASSERT(Q >= 0);
    if (giter->g < giter->G) {
        P4EST_ASSERT(giter->g >= 0);
        giter->quad =
            p4est_quadrant_array_index(giter->ghost_layer,
                                       (size_t) giter->g);
        giter->level = pow (2., giter->quad->level);
        p4est_qcoord_to_vertex(pfg->connect,
                               parflow_p4est_gquad_owner_tree(giter->quad),
                               giter->quad->x, giter->quad->y,
#ifdef P4_TO_P8
                               giter->quad->z,
#endif
                               giter->v);
        // TODO: Get owner rank

    }
    P4EST_ASSERT(parflow_p4est_giter_isvalid(giter));
}

int
parflow_p4est_giter_isvalid_2d(parflow_p4est_giter_2d_t * giter)
{
    return (giter->g < giter->G);
}

void
parflow_p4est_giter_next_2d(parflow_p4est_giter_2d_t * giter)
{

    P4EST_ASSERT(parflow_p4est_giter_isvalid(giter));

    if (++giter->g < giter->G) {

        giter->quad =
            p4est_quadrant_array_index(giter->ghost_layer,
                                       (size_t) giter->g);
        giter->level = pow (2., giter->quad->level);
        p4est_qcoord_to_vertex(pfg->connect,
                               parflow_p4est_gquad_owner_tree(giter->quad),
                               giter->quad->x, giter->quad->y,
#ifdef P4_TO_P8
                               giter->quad->z,
#endif
                               giter->v);
        // TODO: get owner rank
    } else {

    }
}
/* END: Ghost iterator routines */

#if 0
void
parflow_p4est_qcoord_to_vertex_2d(parflow_p4est_grid_t * pfgrid,
                                  p4est_topidx_t treeid,
                                  p4est_quadrant_t * quad, double v[3])
{

    p4est_qcoord_to_vertex(pfgrid->connect, treeid, quad->x, quad->y,
#ifdef P4_TO_P8
                           quad->z,
#endif
                           v);
}
#endif
