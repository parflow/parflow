#include <parflow.h>
#include "parflow_p4est_math.h"
#include <sc_functions.h>
#ifndef P4_TO_P8
#include "parflow_p4est_2d.h"
#include <p4est_vtk.h>
#else
#include "parflow_p4est_3d.h"
#include <p8est_bits.h>
#include <p8est_vtk.h>
#endif

static p4est_topidx_t
parflow_p4est_lexicord(int Tx, int Ty, int vv[3])
{
  return (vv[2] * Ty + vv[1]) * Tx + vv[0];
}

#ifdef P4_TO_P8
static uint64_t
parflow_p4est_morton(int w[2])
{
  int i;
  uint64_t xx = (uint64_t)w[0];
  uint64_t yy = (uint64_t)w[1];
  uint64_t one = (uint64_t)1;
  int bits = (sizeof(uint64_t) * CHAR_BIT) / P4EST_DIM;
  uint64_t m = 0;

  for (i = 0; i < bits; ++i)
  {
    m |= ((xx & (one << i)) << (i));
    m |= ((yy & (one << i)) << (i + 1));
  }

  return m;
}
#endif

parflow_p4est_grid_2d_t *
parflow_p4est_grid_2d_new(int Px, int Py
#ifdef P4_TO_P8
                          , int Pz
#endif
                          )
{
  int i;
  int g, gt;
  int tx, ty;

#ifdef P4_TO_P8
  int tz;
#endif
  int vv[3];
  p4est_topidx_t tt, num_trees, lidx;
  double v[3];
  int initial_level;
  size_t quad_data_size;
  parflow_p4est_grid_2d_t *pfg;

  pfg = P4EST_ALLOC_ZERO(parflow_p4est_grid_2d_t, 1);
  tx = pfmax(Px, 1);
  ty = pfmax(Py, 1);
  gt = parflow_p4est_gcd(tx, ty);
#ifdef P4_TO_P8
  tz = pfmax(Pz, 1);
  gt = parflow_p4est_gcd(gt, tz);
#endif
  initial_level = parflow_p4est_powtwo_div(gt);
  g = 1 << initial_level;
  quad_data_size = sizeof(parflow_p4est_quad_data_t);

  pfg->Tx = tx / g;
  pfg->Ty = ty / g;
#ifdef P4_TO_P8
  pfg->Tz = tz / g;
#endif

  BeginTiming(P4ESTimingIndex);

  /** Create connectivity structure */
  pfg->connect = p4est_connectivity_new_brick(pfg->Tx, pfg->Ty
#ifdef P4_TO_P8
                                              , pfg->Tz, 0
#endif
                                              , 0, 0);

  /** Create p4est structure */
  pfg->forest = p4est_new_ext(amps_CommWorld, pfg->connect,
                              0, initial_level, 1,
                              quad_data_size, NULL, NULL);

  /** allocate ghost storage */
  pfg->ghost = p4est_ghost_new(pfg->forest, P4EST_CONNECT_CORNER);

  EndTiming(P4ESTimingIndex);

  pfg->ghost_data =
    sc_array_new_count(sizeof(parflow_p4est_ghost_data_t),
                       pfg->ghost->ghosts.elem_count);

  num_trees = pfg->Tx * pfg->Ty;
#ifdef P4_TO_P8
  num_trees *= pfg->Tz;
#endif

  /** Compute permutation transforming lexicographical to brick order
   *  and its inverse */
  pfg->lexic_to_tree = P4EST_ALLOC_ZERO(p4est_topidx_t, num_trees);
  pfg->tree_to_lexic =
    P4EST_ALLOC_ZERO(p4est_topidx_t, P4EST_DIM * num_trees);

  for (tt = 0; tt < num_trees; ++tt)
  {
    pfg->lexic_to_tree[tt] = -1;
    for (i = 0; i < P4EST_DIM; ++i)
    {
      pfg->tree_to_lexic[P4EST_DIM * tt + i] = -1;
    }
  }

  for (tt = 0; tt < num_trees; ++tt)
  {
    p4est_qcoord_to_vertex(pfg->connect, tt, 0, 0
#ifdef P4_TO_P8
                           , 0
#endif
                           , v);

    vv[0] = (int)v[0];
    vv[1] = (int)v[1];
    vv[2] = (int)v[2];

    lidx = parflow_p4est_lexicord(pfg->Tx, pfg->Ty, vv);
    P4EST_ASSERT(lidx >= 0 && lidx < num_trees);
    pfg->lexic_to_tree[lidx] = tt;

    for (i = 0; i < P4EST_DIM; ++i)
    {
      pfg->tree_to_lexic[P4EST_DIM * tt + i] = vv[i];
    }
  }

  return pfg;
}

void
parflow_p4est_grid_2d_destroy(parflow_p4est_grid_2d_t * pfg)
{
  /** Mesh structure must have been freed before with
   * parflow_p4est_grid_2d_mesh_destroy */
  P4EST_ASSERT(pfg->mesh == NULL);

  p4est_ghost_destroy(pfg->ghost);
  sc_array_destroy(pfg->ghost_data);
  p4est_destroy(pfg->forest);
  p4est_connectivity_destroy(pfg->connect);
  P4EST_FREE(pfg->lexic_to_tree);
  P4EST_FREE(pfg->tree_to_lexic);
  P4EST_FREE(pfg);
}

void
parflow_p4est_grid_2d_mesh_init(parflow_p4est_grid_2d_t * pfgrid)
{
  BeginTiming(P4ESTimingIndex);
  pfgrid->mesh = p4est_mesh_new(pfgrid->forest, pfgrid->ghost,
                                P4EST_CONNECT_FACE);
  EndTiming(P4ESTimingIndex);
}

void
parflow_p4est_grid_2d_mesh_destroy(parflow_p4est_grid_2d_t * pfgrid)
{
  if (pfgrid->mesh != NULL)
  {
    p4est_mesh_destroy(pfgrid->mesh);
    pfgrid->mesh = NULL;
  }
}

void
parflow_p4est_get_zneigh_2d(Subgrid * subgrid
#ifdef P4_TO_P8
                            , parflow_p4est_qiter_2d_t * qiter,
                            parflow_p4est_grid_2d_t * pfgrid
#endif
                            )
{
  int z_neighs[] = { -1, -1 };

#ifdef P4_TO_P8
  p4est_mesh_t   *mesh = pfgrid->mesh;
  p4est_locidx_t K = mesh->local_num_quadrants;
  p4est_locidx_t G = mesh->ghost_num_quadrants;
  int8_t qtof;
  p4est_locidx_t qtoq;
  int f, lidx;
  int faces[] = { 4, 5 };              /** -z face = 4, +z face = 5 */

  P4EST_ASSERT(qiter->itype == PARFLOW_P4EST_QUAD);
  lidx = qiter->local_idx;

  /** Inspect mesh structure to get neighborhod information **/
  for (f = 0; f < 2; ++f)
  {
    qtoq = mesh->quad_to_quad[P4EST_FACES * lidx + faces[f]];
    P4EST_ASSERT(qtoq >= 0);
    qtof = mesh->quad_to_face[P4EST_FACES * lidx + faces[f]];

    if (qtoq == lidx && qtof == faces[f])
    {
      /** face lies on the domain boundary, nothing to do **/
    }
    else
    {
      if (qtoq >= K)
      {
        /** face neighbor is on a different processor,
         *  then qtoq contains its local index in the ghost layer */
        P4EST_ASSERT((qtoq - K) < G);
        z_neighs[f] = qtoq;
      }
      else
      {
        /** face neighbor is on the same processor,
         * then qtoq contains its local index */
        P4EST_ASSERT(qtoq < K);
        z_neighs[f] = qtoq;
      }
    }
  }
#endif

  subgrid->minus_z_neigh = z_neighs[0];
  subgrid->plus_z_neigh = z_neighs[1];
}

/*
 * START: Quadrant iterator routines
 */

/** Complete iterator information */
static parflow_p4est_qiter_2d_t *
parflow_p4est_qiter_info_2d(parflow_p4est_qiter_2d_t * qit_2d)
{
  int rank;

  P4EST_ASSERT(qit_2d != NULL);
  if (qit_2d->itype & PARFLOW_P4EST_QUAD)
  {
    qit_2d->quad =
      p4est_quadrant_array_index(qit_2d->tquadrants,
                                 (size_t)qit_2d->q);
  }
  else
  {
    P4EST_ASSERT(qit_2d->itype & PARFLOW_P4EST_GHOST);
    qit_2d->quad =
      p4est_quadrant_array_index(qit_2d->ghost_layer,
                                 (size_t)qit_2d->g);
    qit_2d->which_tree = qit_2d->quad->p.piggy3.which_tree;
    qit_2d->local_idx = qit_2d->quad->p.piggy3.local_num;
    /** Update owner rank **/
    rank = 0;
    while (qit_2d->ghost->proc_offsets[rank + 1] <= qit_2d->g)
    {
      ++rank;
      P4EST_ASSERT(rank < qit_2d->ghost->mpisize);
    }
    qit_2d->owner_rank = rank;
  }

  return qit_2d;
}

/** Allocate and initialize interatior information */
parflow_p4est_qiter_2d_t *
parflow_p4est_qiter_init_2d(parflow_p4est_grid_2d_t * pfg,
                            parflow_p4est_iter_type_t itype)
{
  parflow_p4est_qiter_2d_t *qit_2d;

  /** This processor is empty */
  if (pfg->forest->local_num_quadrants == 0)
  {
    P4EST_ASSERT(pfg->forest->first_local_tree == -1);
    P4EST_ASSERT(pfg->forest->last_local_tree == -2);
    P4EST_ASSERT((int)pfg->ghost->ghosts.elem_count == 0);
    return NULL;
  }

  qit_2d = P4EST_ALLOC_ZERO(parflow_p4est_qiter_2d_t, 1);
  qit_2d->itype = itype;
  qit_2d->connect = pfg->connect;

  if (itype & PARFLOW_P4EST_QUAD)
  {
    /** Populate necesary fields **/
    qit_2d->forest = pfg->forest;
    qit_2d->which_tree = qit_2d->forest->first_local_tree;
    qit_2d->owner_rank = qit_2d->forest->mpirank;
    qit_2d->tree =
      p4est_tree_array_index(qit_2d->forest->trees,
                             qit_2d->which_tree);
    qit_2d->tquadrants = &qit_2d->tree->quadrants;
    qit_2d->Q = (int)qit_2d->tquadrants->elem_count;

    /** Populate ghost fields with invalid values **/
    qit_2d->G = -1;
    qit_2d->g = -1;
  }
  else
  {
    P4EST_ASSERT(itype & PARFLOW_P4EST_GHOST);

    /** Populate necesary fields **/
    qit_2d->ghost = pfg->ghost;
    qit_2d->ghost_layer = &qit_2d->ghost->ghosts;
    qit_2d->G = (int)qit_2d->ghost_layer->elem_count;
    P4EST_ASSERT(qit_2d->G >= 0);

    /** There are no quadrants in this ghost layer,
    **  we are done. */
    if (qit_2d->g == qit_2d->G)
    {
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

/** Advance the iterator */
parflow_p4est_qiter_2d_t *
parflow_p4est_qiter_next_2d(parflow_p4est_qiter_2d_t * qit_2d)
{
  P4EST_ASSERT(qit_2d != NULL);
  if (qit_2d->itype & PARFLOW_P4EST_QUAD)
  {
    /** Update local index**/
    ++qit_2d->local_idx;

    /** We visited all local quadrants in current tree */
    if (++qit_2d->q == qit_2d->Q)
    {
      if (++qit_2d->which_tree <= qit_2d->forest->last_local_tree)
      {
        /** Reset quadrant counter to skip to the next tree */
        qit_2d->q = 0;

        /** Update interator information to next tree **/
        qit_2d->tree =
          p4est_tree_array_index(qit_2d->forest->trees,
                                 qit_2d->which_tree);
        qit_2d->tquadrants = &qit_2d->tree->quadrants;
        qit_2d->Q = (int)qit_2d->tquadrants->elem_count;
      }
      else
      {
        /** We visited all local trees. We are done, free
        ** iterator and return null ptr */
        P4EST_FREE(qit_2d);
        return NULL;
      }
    }
  }
  else
  {
    P4EST_ASSERT(qit_2d->itype & PARFLOW_P4EST_GHOST);

    /** We visited all local quadrants in the ghost layer.
    ** We are done, deallocate iterator and return null ptr */
    if (++qit_2d->g == qit_2d->G)
    {
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
  return (parflow_p4est_quad_data_t*)qit_2d->quad->p.user_data;
}

parflow_p4est_ghost_data_t *
parflow_p4est_get_ghost_data_2d(parflow_p4est_grid_2d_t *  pfg,
                                parflow_p4est_qiter_2d_t * qit_2d)
{
  sc_array_t     *gdata = pfg->ghost_data;

  P4EST_ASSERT(qit_2d->itype & PARFLOW_P4EST_GHOST);

  return (parflow_p4est_ghost_data_t*)sc_array_index_int(gdata,
                                                         qit_2d->g);
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

void
parflow_p4est_get_projection_info_2d(Subgrid * subgrid
#ifdef P4_TO_P8
                                     , int zl_idx,
                                     parflow_p4est_grid_2d_t * pfg
#endif
                                     , int info[2])
{
#ifdef P4_TO_P8
  double v[3];
  int face;
  int vv[3], w[2];
  int lidx;
  int qlen;
  p4est_locidx_t which_quad;
  p4est_tree_t   *tree;
  p4est_topidx_t tt = (int32_t)subgrid->owner_tree;
  p4est_quadrant_t *quad;
  p4est_topidx_t num_trees = pfg->Tx * pfg->Ty * pfg->Tz;
  p4est_topidx_t tp;
  p4est_quadrant_t proj;

  P4EST_QUADRANT_INIT(&proj);

  P4EST_ASSERT(pfg->forest->first_local_tree <= tt &&
               tt <= pfg->forest->last_local_tree);
  tree = p4est_tree_array_index(pfg->forest->trees, tt);

  /** Grab the quadrant which this subgrid is attached to */
  which_quad = SubgridLocIdx(subgrid) - tree->quadrants_offset;
  quad =
    p4est_quadrant_array_index(&tree->quadrants, (size_t)which_quad);

  /** Compute its coordinates relative to tree vertex */
  p4est_qcoord_to_vertex(pfg->connect, tt, quad->x, quad->y, quad->z, v);

  qlen = 1 << quad->level;

  /** Project such coordinates in the desired z level and figure out the
   * tree owning the projection */
  w[0] = (int)(v[0] * qlen);
  w[1] = (int)(v[1] * qlen);

  vv[0] = w[0] / qlen;
  vv[1] = w[1] / qlen;
  vv[2] = zl_idx / qlen;

  lidx = parflow_p4est_lexicord(pfg->Tx, pfg->Ty, vv);
  P4EST_ASSERT(lidx >= 0 && lidx < num_trees);
  tp = pfg->lexic_to_tree[lidx];
  P4EST_ASSERT(tp >= 0);

  /** Provide face direction to search function */
  face = v[2] > zl_idx ? 4 : 5;

  /** Construct a quadrant matching the coordinates of the desired
   * projection */
  proj.level = quad->level;
  proj.x = quad->x;
  proj.y = quad->y;
  proj.z = (p4est_qcoord_t)
           (zl_idx % qlen) * P4EST_QUADRANT_LEN(proj.level);

  /** Owner of projected subgrid is the owner of the temporay quadrant */
  info[0] = p4est_quadrant_find_owner(pfg->forest, tp, face, &proj);
  P4EST_ASSERT(info[0] >= 0 && info[0] < GlobalsNumProcs);

  /** Use the morton code of the projected quadrant as tag */
  info[1] = (int)parflow_p4est_morton(w);
  P4EST_ASSERT(info[1] < MPI_TAG_UB);
#else
  info[0] = SubgridProcess(subgrid);
  info[1] = SubgridLocIdx(subgrid);
#endif
}

void
parflow_p4est_nquads_per_rank_2d(parflow_p4est_grid_2d_t * pfg,
                                 int *                     quads_per_rank)
{
  int ig;
  int mpisize = amps_Size(amps_CommWorld);
  p4est_gloidx_t *gfq = pfg->forest->global_first_quadrant;

  for (ig = 0; ig < mpisize; ig++)
  {
    quads_per_rank[ig] = (int)(gfq[ig + 1] - gfq[ig]);
  }
}

void
parflow_p4est_get_brick_coord_2d(Subgrid *                 subgrid,
                                 parflow_p4est_grid_2d_t * pfg,
                                 p4est_gloidx_t            bcoord[P4EST_DIM])
{
  int k;
  int rank = amps_Rank(amps_CommWorld);
  double v[3];
  p4est_qcoord_t qcoord[3];
  p4est_topidx_t which_tree;
  p4est_locidx_t which_quad, which_ghost;
  p4est_tree_t   *tree;
  p4est_quadrant_t *quad;

  which_tree = SubgridOwnerTree(subgrid);

  if (rank == subgrid->process)
  {
    /** We own the subgrid, fetch local quadrant associated to it */
    tree = p4est_tree_array_index(pfg->forest->trees, which_tree);
    which_quad = SubgridLocIdx(subgrid) - tree->quadrants_offset;
    P4EST_ASSERT(0 <= which_quad &&
                 which_quad <
                 (p4est_locidx_t)tree->quadrants.elem_count);
    quad =
      p4est_quadrant_array_index(&tree->quadrants,
                                 (size_t)which_quad);
  }
  else
  {
    /** We do not own the subgrid, we fetch ghost quadrant
     * associated  to it */
    which_ghost = SubgridGhostIdx(subgrid);
    P4EST_ASSERT(which_ghost >= 0 &&
                 which_ghost <
                 (p4est_locidx_t)pfg->ghost->ghosts.elem_count);
    quad =
      p4est_quadrant_array_index(&pfg->ghost->ghosts,
                                 (size_t)which_ghost);
  }

  parflow_p4est_qcoord_to_vertex_2d(pfg->connect, which_tree, quad, v);

  qcoord[0] = quad->x;
  qcoord[1] = quad->y;
#ifdef P4_TO_P8
  qcoord[2] = quad->z;
#endif

  for (k = 0; k < P4EST_DIM; k++)
  {
    bcoord[k] = pfg->tree_to_lexic[P4EST_DIM * which_tree + k] *
                (p4est_gloidx_t)(1 << P4EST_MAXLEVEL) + qcoord[k];
  }
}
