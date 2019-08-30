/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
 *  LLC. Produced at the Lawrence Livermore National Laboratory. Written
 *  by the Parflow Team (see the CONTRIBUTORS file)
 *  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
 *
 *  This file is part of Parflow. For details, see
 *  http://www.llnl.gov/casc/parflow
 *
 *  Please read the COPYRIGHT file or Our Notice and the LICENSE file
 *  for the GNU Lesser General Public License.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License (as published
 *  by the Free Software Foundation) version 2.1 dated February 1999.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
 *  and conditions of the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 *  USA
 **********************************************************************EHEADER*/

/*****************************************************************************
*
* Create the distributed grid from the user-grid input
*
*****************************************************************************/

#include "parflow.h"
#ifdef HAVE_P4EST
#include "parflow_p4est_dependences.h"
#endif

/*--------------------------------------------------------------------------
 * GetGridSubgrids:
 *   Returns a SubgridArray containing subgrids on my process.
 *
 *   Notes:
 *     The ordering of `subgrids' is inferred by `all_subgrids'.
 *     The `subgrids' array points to subgrids in `all_subgrids'.
 *--------------------------------------------------------------------------*/

SubgridArray  *GetGridSubgrids(
                               SubgridArray *all_subgrids)
{
  SubgridArray  *subgrids;

  Subgrid       *s;

  int i, my_proc;


  my_proc = amps_Rank(amps_CommWorld);

  subgrids = NewSubgridArray();

  ForSubgridI(i, all_subgrids)
  {
    s = SubgridArraySubgrid(all_subgrids, i);
    if (SubgridProcess(s) == my_proc)
      AppendSubgrid(s, subgrids);
  }

  return subgrids;
}


/*--------------------------------------------------------------------------
 * CreateGrid:
 *   We currently assume that the user's grid consists of 1 subgrid only.
 *--------------------------------------------------------------------------*/

Grid           *CreateGrid(
                           Grid *user_grid)
{
  Grid        *grid;
  SubgridArray  *subgrids;
  SubgridArray  *all_subgrids;

#ifdef HAVE_P4EST
  Subgrid       *s0, *gs0;
  SubgridArray  *inner_ghost_subgrids;
  parflow_p4est_grid_t       *pfgrid;
  parflow_p4est_sg_param_t subgparam, *sp = &subgparam;
  parflow_p4est_qiter_t      *qiter;
  parflow_p4est_quad_data_t  *quad_data = NULL;
  parflow_p4est_ghost_data_t *ghost_data = NULL;
  int initial_level;
  int num_local_quads, num_nonlocal_quads;
  int num_ghost_children;
  int                        *z_levels;
  int                        *g_exchange_info;
  int i, lz, pz, offset;
  int nchildren, child_id;
#endif

  if (!USE_P4EST)
  {
    /*-----------------------------------------------------------------------
     * Create all_subgrids
     *-----------------------------------------------------------------------*/

    if (!(all_subgrids = DistributeUserGrid(user_grid)))
    {
      if (!amps_Rank(amps_CommWorld))
        amps_Printf("Incorrect process allocation input\n");
      exit(1);
    }

    /*-----------------------------------------------------------------------
     * Create subgrids
     *-----------------------------------------------------------------------*/

    subgrids = GetGridSubgrids(all_subgrids);
  }
  else
  {
#ifdef HAVE_P4EST
    BeginTiming(P4ESTSetupTimingIndex);

    all_subgrids = NewSubgridArray();
    subgrids = NewSubgridArray();
    inner_ghost_subgrids = NewSubgridArray();

    /* Initialize information to compute number of subgrids
     * and their corresponding dimensions */
    parflow_p4est_sg_param_init(sp);

    /* Allocate and populate array to store z_levels in the grid */
    z_levels = P4EST_ALLOC(int, GlobalsNumProcsZ);

    for (lz = 0; lz < GlobalsNumProcsZ; ++lz)
    {
      pz = lz < sp->l[2]  ? sp->m[2] + 1 : sp->m[2];
      offset = (lz >= sp->l[2]) ? sp->l[2] : 0;
      z_levels[lz] = lz * pz + offset;
    }

    /* Create the pfgrid. */
    pfgrid = parflow_p4est_grid_new(GlobalsNumProcsX, GlobalsNumProcsY,
                                    GlobalsNumProcsZ);

    initial_level = parflow_p4est_get_initial_level(pfgrid);
    num_local_quads = parflow_p4est_local_num_quads(pfgrid);
    num_nonlocal_quads = parflow_p4est_ghost_num_quads(pfgrid);
    nchildren =  1 << parflow_p4est_dim (pfgrid);

    /* Allocate p4est mesh structure if required*/
    parflow_p4est_grid_mesh_init(pfgrid);

    /* Loop over the quadrants (leaves) of this forest
     * and attach a subgrid on each */
    for (qiter = parflow_p4est_qiter_init(pfgrid, PARFLOW_P4EST_QUAD);
         qiter != NULL;
         qiter = parflow_p4est_qiter_next(qiter))
    {
      /* Update paramenters to decide dimensions for the new subgrid */
      parflow_p4est_sg_param_update(qiter, sp);

      /* Allocate new subgrid and attach it to this quadrant */
      quad_data = parflow_p4est_get_quad_data(qiter);
      quad_data->pf_subgrid =
        NewSubgrid(sp->icorner[0], sp->icorner[1], sp->icorner[2],
                   sp->p[0], sp->p[1], sp->p[2], 0, 0, 0,
                   parflow_p4est_qiter_get_owner_rank(qiter));

      SubgridLevel(quad_data->pf_subgrid) =
        parflow_p4est_qiter_get_level(qiter) - initial_level;

      SubgridLocIdx(quad_data->pf_subgrid) =
        parflow_p4est_qiter_get_local_idx(qiter);

      SubgridGhostIdx(quad_data->pf_subgrid) =
        parflow_p4est_qiter_get_ghost_idx(qiter);

      /* Remember tree that owns this subgrid */
      SubgridOwnerTree(quad_data->pf_subgrid) =
        (int32_t)parflow_p4est_qiter_get_tree(qiter);

      /*Retrieve -z and +z neighborhood information*/
      parflow_p4est_get_zneigh(quad_data->pf_subgrid, qiter, pfgrid);

      /* Remember parent's coorner */
      for (i = 0; i < 3; i++)
        quad_data->pf_subgrid->pcorner[i] = sp->pcorner[i];

      AppendSubgrid(quad_data->pf_subgrid, subgrids);
      AppendSubgrid(quad_data->pf_subgrid, all_subgrids);

      parflow_p4est_inner_ghost_create(inner_ghost_subgrids,
                                       quad_data->pf_subgrid, qiter, pfgrid);

      g_exchange_info = quad_data->pf_subgrid->ghostChildren;

      parflow_p4est_ghost_prepare_exchange (pfgrid, qiter, g_exchange_info);
    }

    /* Assert that all local quadrants were visited */
    P4EST_ASSERT(num_local_quads == SubgridArraySize(subgrids));

    /*Destroy p4est mesh structure */
    parflow_p4est_grid_mesh_destroy(pfgrid);

    /* If any, append local 'ghost children' to the local subgrids array */
    num_ghost_children = SubgridArraySize(inner_ghost_subgrids);
    for(i=0; i < num_ghost_children; i++)
    {
        /* fetch a 'ghost child' subgrid */
        gs0 = SubgridArraySubgrid(inner_ghost_subgrids, i);

        /* After construction, the local index of a 'ghost child'
         * is the the local index of its parent; fetch parent */
        s0 =  SubgridArraySubgrid(subgrids, SubgridLocIdx(gs0));

        /* Update local index of this 'ghost child' subgrid to the
         * position it will occupy in the local subgrids array */
        SubgridLocIdx(gs0) = num_local_quads  + i;

        /* Parent subgrid gets access to this child */
        child_id = (-2-SubgridGhostIdx(gs0)) % nchildren;
        s0->ghostChildren[child_id] = SubgridLocIdx(gs0);

        AppendSubgrid(gs0, subgrids);
    }

    /* Share ghost information */
    parflow_p4est_ghost_exchange (pfgrid);

    /* Loop over the ghost layer */
    for (qiter = parflow_p4est_qiter_init(pfgrid, PARFLOW_P4EST_GHOST);
         qiter != NULL;
         qiter = parflow_p4est_qiter_next(qiter))
    {
      /* Update paramenters to decide dimensions for the new subgrid */
      parflow_p4est_sg_param_update(qiter, sp);

      /* Allocate new subgrid and attach it to the corresponding
       * ghost_data structure */
      ghost_data = parflow_p4est_get_ghost_data(pfgrid, qiter);
      ghost_data->pf_subgrid =
        NewSubgrid(sp->icorner[0], sp->icorner[1], sp->icorner[2],
                   sp->p[0], sp->p[1], sp->p[2], 0, 0, 0,
                   parflow_p4est_qiter_get_owner_rank(qiter));

      SubgridLevel(ghost_data->pf_subgrid) =
        parflow_p4est_qiter_get_level(qiter) - initial_level;

      SubgridLocIdx(ghost_data->pf_subgrid) =
        parflow_p4est_qiter_get_local_idx(qiter);

      SubgridGhostIdx(ghost_data->pf_subgrid) =
        parflow_p4est_qiter_get_ghost_idx(qiter);

      /* Remember tree that owns this subgrid */
      SubgridOwnerTree(ghost_data->pf_subgrid) =
        (int32_t)parflow_p4est_qiter_get_tree(qiter);

      /* Remember parent's coorner */
      for (i = 0; i < 3; i++)
        ghost_data->pf_subgrid->pcorner[i] = sp->pcorner[i];

      AppendSubgrid(ghost_data->pf_subgrid, all_subgrids);

      parflow_p4est_inner_ghost_create(inner_ghost_subgrids,
                                       ghost_data->pf_subgrid, qiter, pfgrid);
    }

    /* If any, append 'ghost children' from subgrids in the ghost layer
     * subgrids array */
    for(i = num_ghost_children; i < SubgridArraySize(inner_ghost_subgrids); i++)
    {
        /* fetch a 'ghost child' subgrid */
        gs0 = SubgridArraySubgrid(inner_ghost_subgrids, i);

        /* After construction, the local index of a 'ghost child'
         * is the the index of its parent in the ghost layer; fetch parent */
        s0 =  SubgridArraySubgrid(all_subgrids, num_local_quads + SubgridLocIdx(gs0));

        /* Parent subgrid gets access to this child */
        child_id = (-2-SubgridGhostIdx(gs0)) % nchildren;
        s0->ghostChildren[child_id] = num_local_quads + i;

        AppendSubgrid(gs0, subgrids);
    }

    /* ParFlow should not see or loop over the inner ghost subgrids*/
    SubgridArraySize(subgrids) = num_local_quads;

    /*There is no PxQxR processor arrange with p4est, set invalid values*/
    GlobalsP = GlobalsQ = GlobalsR = -1;

    EndTiming(P4ESTSetupTimingIndex);
#else
    PARFLOW_ERROR("ParFlow compiled without p4est");
#endif
  }

  /*-----------------------------------------------------------------------
   * Create the grid.
   *-----------------------------------------------------------------------*/

  grid = NewGrid(subgrids, all_subgrids);

  if (USE_P4EST)
  {
#ifdef HAVE_P4EST
    grid->pfgrid = pfgrid;
    grid->z_levels = z_levels;
    grid->proj_flag = 0;
    grid->owns_pfgrid = 1;
    grid->innerGhostSubgrids  = inner_ghost_subgrids;
#else
    PARFLOW_ERROR("ParFlow compiled without p4est");
#endif
  }

  /*-----------------------------------------------------------------------
   * Create communication packages.
   *-----------------------------------------------------------------------*/

  CreateComputePkgs(grid);

  // SGS Debug
  globals->grid3d = grid;

  parflow_p4est_vector_test(grid);
  parflow_p4est_matrix_test(grid);

  return grid;
}

