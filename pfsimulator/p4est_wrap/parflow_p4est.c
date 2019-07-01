#include <parflow.h>
#include "parflow_p4est_2d.h"
#include "parflow_p4est_3d.h"
#include <sc_functions.h>

struct parflow_p4est_grid {
  int dim;
  union {
    parflow_p4est_grid_2d_t *p4;
    parflow_p4est_grid_3d_t *p8;
  } p;
};

struct parflow_p4est_qiter {
  int dim;
  int initial_level;
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

int parflow_p4est_dim (parflow_p4est_grid_t *pfgrid)
{
    return PARFLOW_P4EST_GET_GRID_DIM(pfgrid);
}

/*
 * A globals structure must exist prior calling this function
 */
void
parflow_p4est_sg_param_init(parflow_p4est_sg_param_t *sp)
{
  int t;

  sp->N[0] = GetIntDefault("ComputationalGrid.NX", 1);
  sp->N[1] = GetIntDefault("ComputationalGrid.NY", 1);
  sp->N[2] = GetIntDefault("ComputationalGrid.NZ", 1);

  sp->m[0] = GetIntDefault("ComputationalSubgrid.MX", 1);
  sp->m[1] = GetIntDefault("ComputationalSubgrid.MY", 1);
  sp->m[2] = GetIntDefault("ComputationalSubgrid.MZ", 1);

  sp->dim = (sp->N[2] > 1) ? 3 : 2;

  for (t = 0; t < 3; ++t)
  {
    sp->l[t] = sp->N[t] % sp->m[t];
  }
}

void
parflow_p4est_sg_param_update(parflow_p4est_qiter_t *   qiter,
                              parflow_p4est_sg_param_t *sp)
{
  int t, p;
  int offset;
  double v[3], pv[3];

  parflow_p4est_qiter_qcorner(qiter, v, pv);
  for (t = 0; t < 3; ++t)
  {
    sp->icorner[t] = (int)v[t];
    sp->pcorner[t] = (int)pv[t];
    sp->p[t] = (sp->icorner[t] < sp->l[t]) ? sp->m[t] + 1 : sp->m[t];
    p = (sp->pcorner[t] < sp->l[t]) ? sp->m[t] + 1 : sp->m[t];
    offset = (sp->icorner[t] >= sp->l[t]) ? sp->l[t] : 0;
    sp->icorner[t] = sp->icorner[t] * sp->p[t] + offset;
    offset = (sp->pcorner[t] >= sp->l[t]) ? sp->l[t] : 0;
    sp->pcorner[t] = sp->pcorner[t] * p + offset;
    /*TODO: This assertion should be active for the uniform case only*/
    //P4EST_ASSERT(sp->icorner[t] + sp->p[t] <= sp->N[t]);
  }
}

/*
 * A globals structure must exist prior calling this function
 */
parflow_p4est_grid_t *
parflow_p4est_grid_new(int Px, int Py, int Pz)
{
  parflow_p4est_grid_t *pfgrid;

  pfgrid = P4EST_ALLOC_ZERO(parflow_p4est_grid_t, 1);
  if (Pz == 1)
  {
    pfgrid->dim = 2;
    pfgrid->p.p4 = parflow_p4est_grid_2d_new(Px, Py);
  }
  else
  {
    pfgrid->dim = 3;
    pfgrid->p.p8 = parflow_p4est_grid_3d_new(Px, Py, Pz);
  }
  return pfgrid;
}

void
parflow_p4est_grid_destroy(parflow_p4est_grid_t * pfgrid)
{
  int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

  if (dim == 2)
  {
    parflow_p4est_grid_2d_destroy(pfgrid->p.p4);
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    parflow_p4est_grid_3d_destroy(pfgrid->p.p8);
  }
  P4EST_FREE(pfgrid);
}

void
parflow_p4est_grid_mesh_init(parflow_p4est_grid_t *pfgrid)
{
  int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

  if (dim == 2)
  {
    parflow_p4est_grid_2d_mesh_init(pfgrid->p.p4);
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    parflow_p4est_grid_3d_mesh_init(pfgrid->p.p8);
  }
}

void
parflow_p4est_grid_mesh_destroy(parflow_p4est_grid_t *pfgrid)
{
  int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

  if (dim == 2)
  {
    parflow_p4est_grid_2d_mesh_destroy(pfgrid->p.p4);
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    parflow_p4est_grid_3d_mesh_destroy(pfgrid->p.p8);
  }
}

int
parflow_p4est_get_nmirrors (parflow_p4est_grid_t * pfgrid)
{
  int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

  if (dim == 2)
  {
    return (int) pfgrid->p.p4->ghost->mirrors.elem_count;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return (int) pfgrid->p.p4->ghost->mirrors.elem_count;
  }
}

void
parflow_p4est_get_zneigh(Subgrid *               subgrid,
                         parflow_p4est_qiter_t * qiter,
                         parflow_p4est_grid_t *  pfgrid)
{
  int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

  if (dim == 2)
  {
    parflow_p4est_get_zneigh_2d(subgrid);
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    parflow_p4est_get_zneigh_3d(subgrid,
                                qiter->q.qiter_3d, pfgrid->p.p8);
  }
}

int
parflow_p4est_get_initial_level(parflow_p4est_grid_t * pfgrid)
{
  int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

  if (dim == 2)
  {
    return pfgrid->p.p4->initial_level;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return pfgrid->p.p8->initial_level;
  }
}

parflow_p4est_qiter_t *
parflow_p4est_qiter_init(parflow_p4est_grid_t *    pfgrid,
                         parflow_p4est_iter_type_t itype)
{
  parflow_p4est_qiter_t *qiter = NULL;
  parflow_p4est_qiter_2d_t *qit_2d;
  parflow_p4est_qiter_3d_t *qit_3d;
  int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

  if (dim == 2)
  {
    qit_2d = parflow_p4est_qiter_init_2d(pfgrid->p.p4, itype);
    if (qit_2d)
    {
      qiter = P4EST_ALLOC(parflow_p4est_qiter_t, 1);
      qiter->dim = dim;
      qiter->initial_level = parflow_p4est_get_initial_level(pfgrid);
      qiter->q.qiter_2d = qit_2d;
    }
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    qit_3d = parflow_p4est_qiter_init_3d(pfgrid->p.p8, itype);
    if (qit_3d)
    {
      qiter = P4EST_ALLOC(parflow_p4est_qiter_t, 1);
      qiter->dim = dim;
      qiter->initial_level = parflow_p4est_get_initial_level(pfgrid);
      qiter->q.qiter_3d = qit_3d;
    }
  }

  return qiter;
}

parflow_p4est_qiter_t *
parflow_p4est_qiter_next(parflow_p4est_qiter_t * qiter)
{
  int dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

  if (dim == 2)
  {
    if (!parflow_p4est_qiter_next_2d(qiter->q.qiter_2d))
    {
      P4EST_FREE(qiter);
      return NULL;
    }
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    if (!parflow_p4est_qiter_next_3d(qiter->q.qiter_3d))
    {
      P4EST_FREE(qiter);
      return NULL;
    }
  }

  P4EST_ASSERT(qiter);
  return qiter;
}

void
parflow_p4est_qiter_qcorner(parflow_p4est_qiter_t * qiter,
                            double v[3], double pv[3])
{
  int k;
  int dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);
  int level;

  if (dim == 2)
  {
    parflow_p4est_quad_to_vertex_2d(qiter->q.qiter_2d->connect,
                                    qiter->q.qiter_2d->which_tree,
                                    qiter->q.qiter_2d->quad, v);
    parflow_p4est_parent_to_vertex_2d(qiter->q.qiter_2d->connect,
                                      qiter->q.qiter_2d->which_tree,
                                      qiter->q.qiter_2d->quad,
                                      qiter->initial_level, pv);

    level = qiter->q.qiter_2d->quad->level;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    parflow_p4est_quad_to_vertex_3d(qiter->q.qiter_3d->connect,
                                    qiter->q.qiter_3d->which_tree,
                                    qiter->q.qiter_3d->quad, v);
    parflow_p4est_parent_to_vertex_3d(qiter->q.qiter_3d->connect,
                                      qiter->q.qiter_3d->which_tree,
                                      qiter->q.qiter_3d->quad,
                                      qiter->initial_level, pv);
    level = qiter->q.qiter_3d->quad->level;
  }

  for (k = 0; k < 3; ++k)
  {
    v[k] *= sc_intpow(2, level);
    pv[k] *= sc_intpow(2, level > 0 ? level - 1 : level);
  }
}

parflow_p4est_qcorner(parflow_p4est_qiter_t * qiter,
                      double v[3], double pv[3])
{
  int k;
  int dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);
  int level;

  if (dim == 2)
  {
    parflow_p4est_quad_to_vertex_2d(qiter->q.qiter_2d->connect,
                                    qiter->q.qiter_2d->which_tree,
                                    qiter->q.qiter_2d->quad, v);
    parflow_p4est_parent_to_vertex_2d(qiter->q.qiter_2d->connect,
                                      qiter->q.qiter_2d->which_tree,
                                      qiter->q.qiter_2d->quad,
                                      qiter->initial_level, pv);

    level = qiter->q.qiter_2d->quad->level;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    parflow_p4est_quad_to_vertex_3d(qiter->q.qiter_3d->connect,
                                    qiter->q.qiter_3d->which_tree,
                                    qiter->q.qiter_3d->quad, v);
    parflow_p4est_parent_to_vertex_3d(qiter->q.qiter_3d->connect,
                                      qiter->q.qiter_3d->which_tree,
                                      qiter->q.qiter_3d->quad,
                                      qiter->initial_level, pv);
    level = qiter->q.qiter_3d->quad->level;
  }

  for (k = 0; k < 3; ++k)
  {
    v[k] *= sc_intpow(2, level);
    pv[k] *= sc_intpow(2, level > 0 ? level - 1 : level);
  }
}

parflow_p4est_quad_data_t *
parflow_p4est_get_quad_data(parflow_p4est_qiter_t * qiter)
{
  int dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

  if (dim == 2)
  {
    return parflow_p4est_get_quad_data_2d(qiter->q.qiter_2d);
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return parflow_p4est_get_quad_data_3d(qiter->q.qiter_3d);
  }
}

int
parflow_p4est_qiter_get_owner_rank(parflow_p4est_qiter_t * qiter)
{
  int dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

  if (dim == 2)
  {
    return qiter->q.qiter_2d->owner_rank;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return qiter->q.qiter_3d->owner_rank;
  }
}

int
parflow_p4est_qiter_get_local_idx(parflow_p4est_qiter_t * qiter)
{
  int dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

  if (dim == 2)
  {
    return qiter->q.qiter_2d->local_idx;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return qiter->q.qiter_3d->local_idx;
  }
}

int
parflow_p4est_qiter_get_tree(parflow_p4est_qiter_t * qiter)
{
  int dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

  if (dim == 2)
  {
    return qiter->q.qiter_2d->which_tree;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return qiter->q.qiter_3d->which_tree;
  }
}

int
parflow_p4est_qiter_get_idx_in_tree(parflow_p4est_qiter_t * qiter)
{
  int dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

  if (dim == 2)
  {
    return qiter->q.qiter_2d->q;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return qiter->q.qiter_3d->q;
  }
}

int
parflow_p4est_qiter_get_ghost_idx(parflow_p4est_qiter_t * qiter)
{
  int dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

  if (dim == 2)
  {
    return qiter->q.qiter_2d->g;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return qiter->q.qiter_3d->g;
  }
}

int
parflow_p4est_qiter_is_mirror (parflow_p4est_qiter_t * qiter)
{
  int dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

  if (dim == 2)
  {
    return qiter->q.qiter_2d->is_mirror;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return qiter->q.qiter_3d->is_mirror;
  }
}

void
parflow_p4est_ghost_prepare_exchange (parflow_p4est_grid_t * pfgrid,
                                      parflow_p4est_qiter_t * qiter,
                                      int * g_children_info)
{
  parflow_p4est_grid_2d_t * pfg2d;
  parflow_p4est_grid_3d_t * pfg3d;
  parflow_p4est_ghost_data_t   *gd;
  int cmidx;
  int dim = PARFLOW_P4EST_GET_QITER_DIM(pfgrid);

  if (parflow_p4est_qiter_is_mirror(qiter)){
      switch (dim) {
      case 2:
          cmidx = qiter->q.qiter_2d->current_mirror_idx;
          pfg2d = pfgrid->p.p4;
          gd = &pfg2d->mirror_data[cmidx];
          pfg2d->mpointer[cmidx] = gd;
          if (g_children_info != NULL)
              memcpy (gd->ghost_children, g_children_info,
                      P4EST_CHILDREN * sizeof (int));
          break;
      case 3:
          cmidx = qiter->q.qiter_3d->current_mirror_idx;
          pfg3d = pfgrid->p.p8;
          gd = &pfg3d->mirror_data[cmidx];
          pfg3d->mpointer[cmidx] = gd;
          if (g_children_info != NULL)
              memcpy (gd->ghost_children, g_children_info,
                      P8EST_CHILDREN * sizeof (int));
          break;
      default:
          SC_ABORT_NOT_REACHED ();
          break;
      }
  }
}


int
parflow_p4est_qiter_mirror_idx (parflow_p4est_qiter_t * qiter)
{
  int dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

  if (dim == 2)
  {
    return qiter->q.qiter_2d->current_mirror_idx;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return qiter->q.qiter_3d->current_mirror_idx;
  }
}

int
parflow_p4est_qiter_num_mirrors (parflow_p4est_qiter_t * qiter)
{
  int dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

  if (dim == 2)
  {
    return qiter->q.qiter_2d->num_mirrors;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return qiter->q.qiter_3d->num_mirrors;
  }
}

int
parflow_p4est_qiter_get_level(parflow_p4est_qiter_t * qiter)
{
  int dim = PARFLOW_P4EST_GET_QITER_DIM(qiter);

  if (dim == 2)
  {
    return qiter->q.qiter_2d->quad->level;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return qiter->q.qiter_3d->quad->level;
  }
}

parflow_p4est_ghost_data_t *
parflow_p4est_get_ghost_data(parflow_p4est_grid_t *  pfgrid,
                             parflow_p4est_qiter_t * qiter)
{
  int dim = PARFLOW_P4EST_GET_QITER_DIM(pfgrid);

  if (dim == 2)
  {
    return parflow_p4est_get_ghost_data_2d(pfgrid->p.p4,
                                           qiter->q.qiter_2d);
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return parflow_p4est_get_ghost_data_3d(pfgrid->p.p8,
                                           qiter->q.qiter_3d);
  }
}

void parflow_p4est_ghost_exchange (parflow_p4est_grid_t * pfgrid)
{
    int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

    if (dim == 2)
    {
        parflow_p4est_ghost_exchange_2d (pfgrid->p.p4);
    }
    else
    {
        P4EST_ASSERT(dim == 3);
        parflow_p4est_ghost_exchange_3d (pfgrid->p.p8);
    }
}

int
parflow_p4est_local_num_quads(parflow_p4est_grid_t * pfgrid)
{
  int K;
  int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

  if (dim == 2)
  {
    K = pfgrid->p.p4->forest->local_num_quadrants;
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    K = pfgrid->p.p8->forest->local_num_quadrants;
  }

  return K;
}

int
parflow_p4est_rank_is_empty(parflow_p4est_grid_t * pfgrid)
{
  int K = parflow_p4est_local_num_quads(pfgrid);

  return K > 0 ? 0 : 1;
}

void
parflow_p4est_get_projection_info(Subgrid *subgrid, int z_level,
                                  parflow_p4est_grid_t *pfgrid, int info[2])
{
  int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

  if (dim == 2)
  {
    parflow_p4est_get_projection_info_2d(subgrid, info);
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    parflow_p4est_get_projection_info_3d(subgrid, z_level,
                                         pfgrid->p.p8, info);
  }
}

void
parflow_p4est_nquads_per_rank(parflow_p4est_grid_t *pfgrid,
                              int *                 quads_per_rank)
{
  int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

  if (dim == 2)
  {
    parflow_p4est_nquads_per_rank_2d(pfgrid->p.p4,
                                     quads_per_rank);
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    parflow_p4est_nquads_per_rank_3d(pfgrid->p.p8,
                                     quads_per_rank);
  }
}

void
parflow_p4est_get_brick_coord(Subgrid *subgrid, parflow_p4est_grid_t *pfgrid,
                              p4est_gloidx_t bcoord[3])
{
  int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

  if (dim == 2)
  {
    parflow_p4est_get_brick_coord_2d(subgrid, pfgrid->p.p4,
                                     bcoord);
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    parflow_p4est_get_brick_coord_3d(subgrid, pfgrid->p.p8,
                                     bcoord);
  }
}

int parflow_p4est_check_neigh(Subgrid *sfine, Subgrid *scoarse,
                              parflow_p4est_grid_t * pfgrid)
{
  int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

  if (dim == 2)
  {
    return parflow_p4est_check_neigh_2d(sfine, scoarse,
                                        pfgrid->p.p4);
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    return parflow_p4est_check_neigh_3d(sfine, scoarse,
                                        pfgrid->p.p8);
  }
}

void            parflow_p4est_inner_ghost_create(SubgridArray * innerGhostsubgrids,
                                                 Subgrid * subgrid,
                                     parflow_p4est_qiter_t * qiter,
                                     parflow_p4est_grid_t * pfgrid
                                     )
{
  int dim = PARFLOW_P4EST_GET_GRID_DIM(pfgrid);

  if (dim == 2)
  {
    parflow_p4est_inner_ghost_create_2d(innerGhostsubgrids,
                                        subgrid, qiter->q.qiter_2d,
                                        pfgrid->p.p4);
  }
  else
  {
    P4EST_ASSERT(dim == 3);
    parflow_p4est_inner_ghost_create_3d(innerGhostsubgrids,
                                        subgrid, qiter->q.qiter_3d,
                                        pfgrid->p.p8);
  }
}


Subgrid *parflow_p4est_fetch_subgrid(SubgridArray *subgrids,
                                     SubgridArray *all_subgrids,
                                     int local_idx, int ghost_idx)
{
  int num_local = SubgridArraySize(subgrids);
  int num_ghost = SubgridArraySize(all_subgrids) - num_local;
  Subgrid *s = NULL;

  if (ghost_idx < 0)
  {
    //P4EST_ASSERT(local_idx >= 0 && local_idx < num_local);
    s = SubgridArraySubgrid(subgrids, local_idx);
  }
  else
  {
    //P4EST_ASSERT(ghost_idx >= 0 && ghost_idx < num_ghost);
    s = SubgridArraySubgrid(all_subgrids, num_local + ghost_idx);
  }

  return s;
}
