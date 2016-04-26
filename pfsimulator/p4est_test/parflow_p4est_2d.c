#include <parflow.h>
#include "parflow_p4est_math.h"
#ifndef P4_TO_P8
#include "parflow_p4est_2d.h"
#include <p4est_vtk.h>
#else
#include "parflow_p4est_3d.h"
#include <p8est_vtk.h>
#endif

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
    int             initial_level;
    size_t          quad_data_size;
    parflow_p4est_grid_2d_t *pfg;

    pfg = P4EST_ALLOC_ZERO(parflow_p4est_grid_2d_t, 1);
    tx = pfmax(Px, 1);
    ty = pfmax(Py, 1);
    gt = gcd(tx, ty);
#ifdef P4_TO_P8
    tz = pfmax(Pz, 1);
    gt = gcd(gt, tz);
#endif
    initial_level = powtwo_div(gt);
    g = 1 << initial_level;
    quad_data_size = sizeof(parflow_p4est_quad_data_t);

    /*
     * Create connectivity structure
     */
    pfg->connect = p4est_connectivity_new_brick(tx / g, ty / g
#ifdef P4_TO_P8
                                                , tz / g, 0
#endif
                                                , 0, 0);

    /*
     * Create p4est structure
     */
    pfg->forest = p4est_new_ext(amps_CommWorld, pfg->connect,
                                0, initial_level, 1,
                                quad_data_size, NULL, NULL);

    /*
     * allocate ghost storage 
     */
    pfg->ghost = p4est_ghost_new(pfg->forest, P4EST_CONNECT_FACE);
    pfg->ghost_data = sc_array_new_size(sizeof(parflow_p4est_ghost_data_t),
                                        pfg->ghost->ghosts.elem_count);

    // p4est_vtk_write_file (pfg->forest, NULL, P4EST_STRING "_pfbrick");

    return pfg;
}

void
parflow_p4est_grid_2d_destroy(parflow_p4est_grid_2d_t * pfg)
{
    p4est_ghost_destroy(pfg->ghost);
    sc_array_destroy(pfg->ghost_data);
    p4est_destroy(pfg->forest);
    p4est_connectivity_destroy(pfg->connect);
    P4EST_FREE(pfg);
}

/*
 * START: Quadrant iterator routines 
 */
static parflow_p4est_qiter_2d_t *
parflow_p4est_qiter_info_2d(parflow_p4est_qiter_2d_t * qit_2d)
{
    int             rank;

    P4EST_ASSERT(qit_2d != NULL);
    if (qit_2d->itype & PARFLOW_P4EST_QUAD) {
        qit_2d->quad =
            p4est_quadrant_array_index(qit_2d->tquadrants,
                                       (size_t) qit_2d->q);
    } else {
        P4EST_ASSERT(qit_2d->itype & PARFLOW_P4EST_GHOST);
        qit_2d->quad =
            p4est_quadrant_array_index(qit_2d->ghost_layer,
                                       (size_t) qit_2d->g);
        qit_2d->which_tree = qit_2d->quad->p.piggy3.which_tree;
        qit_2d->local_idx = qit_2d->quad->p.piggy3.local_num;
        /** Update owner rank **/
        rank = 0;
        while (qit_2d->ghost->proc_offsets[rank + 1] <= qit_2d->g) {
            ++rank;
            P4EST_ASSERT(rank < qit_2d->ghost->mpisize);
        }
        qit_2d->owner_rank = rank;
    }

    return qit_2d;
}

parflow_p4est_qiter_2d_t *
parflow_p4est_qiter_init_2d(parflow_p4est_grid_2d_t * pfg,
                            parflow_p4est_iter_type_t itype)
{
    parflow_p4est_qiter_2d_t *qit_2d;

    /** This processor is empty */
    if (pfg->forest->local_num_quadrants == 0) {
        P4EST_ASSERT(pfg->forest->first_local_tree == -1);
        P4EST_ASSERT(pfg->forest->last_local_tree == -2);
        P4EST_ASSERT( (int) pfg->ghost->ghosts.elem_count == 0);
        return NULL;
    }

    qit_2d = P4EST_ALLOC_ZERO(parflow_p4est_qiter_2d_t, 1);
    qit_2d->itype = itype;
    qit_2d->connect = pfg->connect;

    if (itype & PARFLOW_P4EST_QUAD) {

       /** Populate necesary fields **/
        qit_2d->forest = pfg->forest;
        qit_2d->which_tree = qit_2d->forest->first_local_tree;
        qit_2d->owner_rank = qit_2d->forest->mpirank;
        qit_2d->tree =
            p4est_tree_array_index(qit_2d->forest->trees,
                                   qit_2d->which_tree);
        qit_2d->tquadrants = &qit_2d->tree->quadrants;
        qit_2d->Q = (int) qit_2d->tquadrants->elem_count;

        /** Populate ghost fields with invalid values **/
        qit_2d->G = -1;
        qit_2d->g = -1;
    } else {
        P4EST_ASSERT(itype & PARFLOW_P4EST_GHOST);

        /** Populate necesary fields **/
        qit_2d->ghost = pfg->ghost;
        qit_2d->ghost_layer = &qit_2d->ghost->ghosts;
        qit_2d->G = (int) qit_2d->ghost_layer->elem_count;
        P4EST_ASSERT(qit_2d->G >= 0);

        /** There are no quadrants in this ghost layer,
         **  we are done. */
        if (qit_2d->g == qit_2d->G) {
            P4EST_FREE(qit_2d);
            return NULL;
        }

        /** Populate quad fields with invalid values **/
        qit_2d->Q = -1;
        qit_2d->q = -1;
    }

    P4EST_ASSERT(qit_2d != NULL);

    /** Complete iterator information */
    return parflow_p4est_qiter_info_2d(qit_2d);
}

parflow_p4est_qiter_2d_t *
parflow_p4est_qiter_next_2d(parflow_p4est_qiter_2d_t * qit_2d)
{
    P4EST_ASSERT(qit_2d != NULL);
    if (qit_2d->itype & PARFLOW_P4EST_QUAD) {

        /** Update local index**/
        ++qit_2d->local_idx;

        /** We visited all local quadrants in current tree */
        if (++qit_2d->q == qit_2d->Q) {

            if (++qit_2d->which_tree <= qit_2d->forest->last_local_tree) {

                /** Reset quadrant counter to skip to the next tree */
                qit_2d->q = 0;

                /** Update interator information to next tree **/
                qit_2d->tree =
                    p4est_tree_array_index(qit_2d->forest->trees,
                                           qit_2d->which_tree);
                qit_2d->tquadrants = &qit_2d->tree->quadrants;
                qit_2d->Q = (int) qit_2d->tquadrants->elem_count;

            } else {

                /** We visited all local trees. We are done, free
                 ** iterator and return null ptr */
                P4EST_FREE(qit_2d);
                return NULL;
            }
        }
    } else {
        P4EST_ASSERT(qit_2d->itype & PARFLOW_P4EST_GHOST);

        /** We visited all local quadrants in the ghost layer.
         ** We are done, deallocate iterator and return null ptr */
        if (++qit_2d->g == qit_2d->G) {
            P4EST_FREE(qit_2d);
            return NULL;
        }
    }

    P4EST_ASSERT(qit_2d != NULL);

    /** Update iterator information */
    return parflow_p4est_qiter_info_2d(qit_2d);
}

parflow_p4est_quad_data_t *
parflow_p4est_get_quad_data_2d(parflow_p4est_qiter_2d_t * qit_2d)
{
    return (parflow_p4est_quad_data_t *) qit_2d->quad->p.user_data;
}

parflow_p4est_ghost_data_t *
parflow_p4est_get_ghost_data_2d(parflow_p4est_grid_2d_t *pfg,
                                parflow_p4est_qiter_2d_t * qit_2d)
{
    sc_array_t *gdata = pfg->ghost_data;

    P4EST_ASSERT(qit_2d->itype & PARFLOW_P4EST_GHOST);

    return (parflow_p4est_ghost_data_t *) sc_array_index_int(gdata, qit_2d->g);
}

/*
 * END: Quadrant iterator routines 
 */

void
parflow_p4est_qcoord_to_vertex_2d(p4est_connectivity_t * connect,
                                  p4est_topidx_t treeid,
                                  p4est_quadrant_t * quad, double v[3])
{

    p4est_qcoord_to_vertex(connect, treeid, quad->x, quad->y,
#ifdef P4_TO_P8
                           quad->z,
#endif
                           v);
}
