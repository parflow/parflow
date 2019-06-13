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
* Routines for determining send and receive regions from stencil patterns.
*
*****************************************************************************/

#include "parflow.h"

#ifdef HAVE_P4EST
#include "parflow_p4est_dependences.h"
#endif

/*--------------------------------------------------------------------------
 * ComputeRegFromStencil:
 *   RDF hack for now.  Should "grow" recv_reg by stencil, intersect with
 *   compute_reg, and union with send_reg to get dep_region, then subtract
 *   this from compute_reg to get ind_region.
 *--------------------------------------------------------------------------*/

void             ComputeRegFromStencil(
                                       Region **       dep_reg_ptr,
                                       Region **       ind_reg_ptr,
                                       SubregionArray *cr_array, /* compute region SubregionArray */
                                       Region *        send_reg,
                                       Region *        recv_reg,
                                       Stencil *       stencil)
{
  Region          *dep_reg;
  Region          *ind_reg;

  SubregionArray  *ir_array;
  SubregionArray  *dr_array;

  SubregionArray  *a0, *a1;

  int i, j, k;

  (void)recv_reg;
  (void)stencil;

  /*-----------------------------------------------------------------------
   * Set up the dependent region
   *-----------------------------------------------------------------------*/

  dep_reg = NewRegion(SubregionArraySize(cr_array));

#ifndef NO_OVERLAP_COMM_COMP
  ForSubregionArrayI(i, dep_reg)
  {
    FreeSubregionArray(RegionSubregionArray(dep_reg, i));
    RegionSubregionArray(dep_reg, i) =
      UnionSubgridArray((SubgridArray*)RegionSubregionArray(send_reg, i));
  }
#else
  ForSubregionArrayI(i, dep_reg)
  {
    AppendSubregion(RegionSubregionArray(dep_reg, i),
                    SubregionArraySubregion(cr_array, i));
  }
#endif

  /*-----------------------------------------------------------------------
   * Set up the independent region
   *-----------------------------------------------------------------------*/

  ind_reg = NewRegion(SubregionArraySize(cr_array));

#ifndef OVERLAP_COMM_COMP
  ForSubregionArrayI(k, ind_reg)
  {
    ir_array = RegionSubregionArray(ind_reg, k);
    dr_array = RegionSubregionArray(dep_reg, k);

    AppendSubregion(DuplicateSubregion(SubregionArraySubregion(cr_array, k)),
                    ir_array);

    ForSubregionI(i, dr_array)
    {
      a0 = NewSubregionArray();

      ForSubregionI(j, ir_array)
      {
        a1 = SubtractSubgrids(SubregionArraySubregion(ir_array, j),
                              SubregionArraySubregion(dr_array, i));

        AppendSubregionArray(a1, a0);
        SubregionArraySize(a1) = 0;
        FreeSubregionArray(a1);
      }

      FreeSubregionArray(ir_array);
      ir_array = a0;
    }

    RegionSubregionArray(ind_reg, k) = ir_array;
  }
#endif

  *dep_reg_ptr = dep_reg;
  *ind_reg_ptr = ind_reg;
}




/*--------------------------------------------------------------------------
 * GetGridNeighbors
 *   Returns a SubgridArray containing neighbors of `subgrids'.
 *   The neighbors are determined by the stencil passed in.
 *
 * Note: The returned neighbors point to subgrids in the all_subgrids array.
 *--------------------------------------------------------------------------*/

SubgridArray  *GetGridNeighbors(
                                SubgridArray *subgrids,
                                SubgridArray *all_subgrids,
                                Stencil *     stencil)
{
  SubgridArray  *neighbors;

  SubgridArray  *neighbor_subgrids;
  SubgridArray  *tmp_array;

  Subgrid       *subgrid;
  Subgrid       *tmp_subgrid;

  int i, j, k;

  StencilElt    *stencil_shape = StencilShape(stencil);


  /*-----------------------------------------------------------------------
   * Determine neighbor_subgrids: array of neighboring subgrids defined
   *   by stencil
   *-----------------------------------------------------------------------*/

  neighbor_subgrids = NewSubgridArray();

  ForSubgridI(i, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, i);

    tmp_subgrid = DuplicateSubgrid(subgrid);

    for (j = 0; j < StencilSize(stencil); j++)
    {
      SubgridIX(tmp_subgrid) = SubgridIX(subgrid) + stencil_shape[j][0];
      SubgridIY(tmp_subgrid) = SubgridIY(subgrid) + stencil_shape[j][1];
      SubgridIZ(tmp_subgrid) = SubgridIZ(subgrid) + stencil_shape[j][2];

      tmp_array = SubtractSubgrids(tmp_subgrid, subgrid);
      ForSubgridI(k, tmp_array)
      AppendSubgrid(SubgridArraySubgrid(tmp_array, k),
                    neighbor_subgrids);
      SubgridArraySize(tmp_array) = 0;
      FreeSubgridArray(tmp_array);
    }

    FreeSubgrid(tmp_subgrid);
  }

  /*-----------------------------------------------------------------------
   * Determine neighbors
   *-----------------------------------------------------------------------*/

  neighbors = NewSubgridArray();

  ForSubgridI(i, all_subgrids)
  {
    subgrid = SubgridArraySubgrid(all_subgrids, i);

    ForSubgridI(j, neighbor_subgrids)
    {
      if ((tmp_subgrid = IntersectSubgrids(subgrid,
                                           SubgridArraySubgrid(neighbor_subgrids, j))))
      {
        AppendSubgrid(subgrid, neighbors);
        FreeSubgrid(tmp_subgrid);
        break;
      }
    }
  }

  FreeSubgridArray(neighbor_subgrids);

  return neighbors;
}


/*--------------------------------------------------------------------------
 * CommRegFromStencil: RDF todo
 *   Compute the send and recv regions that correspond to a given
 *   computational stencil pattern.
 *--------------------------------------------------------------------------*/

void  CommRegFromStencil(
                         Region **send_region_ptr,
                         Region **recv_region_ptr,
                         Grid *   grid,
                         Stencil *stencil)
{
  Region        *send_region = NULL;
  Region        *recv_region = NULL;

  SubgridArray  *subgrids = GridSubgrids(grid);
  SubgridArray  *neighbors;

  Region        *subgrid_region = NULL;
  Region        *neighbor_region = NULL;

  Region        *region0;
  Region        *region1;

  SubgridArray  *sa0 = NULL;
  SubgridArray  *sa1;
  SubgridArray  *sa2;
  SubgridArray  *sa3;

  Subgrid       *subgrid0;
  Subgrid       *subgrid1;
  Subgrid       *subgrid2;

  int           *proc_array = NULL;
  int num_procs;

  int r, p, i, j, k;

#ifdef HAVE_P4EST
  Subgrid       *s;
  SubgridArray  *sa2_loc;
  int which_child, tt, num_trees;
  int num_loc_idxs;
  int ix, iy, iz;
  int tweak, lev1, lev0;
  int           *tree_array = NULL;
  int           *loc_idx_array = NULL;
  int           *ghost_idx_array = NULL;
#endif

  /*------------------------------------------------------
   * Determine neighbors: When p4est is enabled the array
   * all_subgrids contain the local subgrids plus its neighbors.
   *------------------------------------------------------*/
  if (!USE_P4EST)
  {
    neighbors = GetGridNeighbors(subgrids, GridAllSubgrids(grid), stencil);
  }
  else
  {
#ifdef HAVE_P4EST
    neighbors = GridAllSubgrids(grid);
#else
    PARFLOW_ERROR("ParFlow compiled without p4est");
#endif
  }

  /*------------------------------------------------------
   * Determine subgrid_region and neighbor_region
   *------------------------------------------------------*/

  for (r = 0; r < 2; r++)
  {
    switch (r)
    {
      case 0:
        sa0 = subgrids;
        break;

      case 1:
        sa0 = neighbors;
        break;
    }

    region0 = NewRegion(SubgridArraySize(sa0));

    ForSubgridI(i, sa0)
    {
      subgrid0 = SubgridArraySubgrid(sa0, i);

      subgrid1 = DuplicateSubgrid(subgrid0);

      for (j = 0; j < StencilSize(stencil); j++)
      {
        SubgridIX(subgrid1) =
          SubgridIX(subgrid0) + StencilShape(stencil)[j][0];
        SubgridIY(subgrid1) =
          SubgridIY(subgrid0) + StencilShape(stencil)[j][1];
        SubgridIZ(subgrid1) =
          SubgridIZ(subgrid0) + StencilShape(stencil)[j][2];

        sa2 = SubtractSubgrids(subgrid1, subgrid0);
        ForSubgridI(k, sa2)
        AppendSubgrid(SubgridArraySubgrid(sa2, k),
                      RegionSubregionArray(region0, i));
        SubgridArraySize(sa2) = 0;
        FreeSubgridArray(sa2);
      }

      FreeSubgrid(subgrid1);
    }

    switch (r)
    {
      case 0:
        subgrid_region = region0;
        break;

      case 1:
        neighbor_region = region0;
        break;
    }
  }

  /*------------------------------------------------------
   * Determine send_region and recv_region
   *------------------------------------------------------*/

  for (r = 0; r < 2; r++)
  {
    switch (r)
    {
      case 0:
        region0 = neighbor_region;
        sa0 = subgrids;
        break;

      case 1:
        region0 = subgrid_region;
        sa0 = neighbors;
        break;
    }

    region1 = NewRegion(SubgridArraySize(subgrids));

    ForSubgridI(i, sa0)
    {
      subgrid0 = SubgridArraySubgrid(sa0, i);
#ifdef HAVE_P4EST
      lev0 = SubgridLevel(subgrid0);
#endif
      ForSubregionArrayI(j, region0)
      {
        sa2 = RegionSubregionArray(region0, j);

        ForSubgridI(k, sa2)
        {
          subgrid1 = SubgridArraySubgrid(sa2, k);
          if (USE_P4EST)
          {
#ifdef HAVE_P4EST
            lev1 = SubgridLevel(subgrid1);
            tweak = (lev0 == lev1) ? 0 : lev0 < lev1 ? -1 : +1;
            switch (tweak)
            {
              case -1:

                which_child = parflow_p4est_check_neigh(subgrid1, subgrid0, grid->pfgrid);

                /* We try to intersect a coarse subgrid 'sc' with a fine shifted
                 * one 'sf'. To get the right values we replace sf by its
                 * shifted parent and perform the intersection. */
                if (which_child > 0 && r==0)
                {
                  ix = SubgridIX(subgrid1);
                  iy = SubgridIY(subgrid1);
                  iz = SubgridIZ(subgrid1);

                  s = parflow_p4est_fetch_subgrid(grid->subgrids,
                                                  grid->all_subgrids,
                                                  SubgridLocIdx(subgrid1),
                                                  SubgridGhostIdx(subgrid1));
                  SubgridIX(subgrid1) =
                    ix - SubgridIX(s) + SubgridParentIX(s);
                  SubgridIY(subgrid1) =
                    iy - SubgridIY(s) + SubgridParentIY(s);
                  SubgridIZ(subgrid1) =
                    iz - SubgridIZ(s) + SubgridParentIZ(s);

                  subgrid2 = IntersectSubgrids(subgrid0, subgrid1);

                  SubgridIX(subgrid1) = ix;
                  SubgridIY(subgrid1) = iy;
                  SubgridIZ(subgrid1) = iz;
                }

                /* Replace sc by a suitable 'ghost children' */
                if (which_child > 0 && r==1)
                {
                  s = SubgridArraySubgrid(subgrids, subgrid0->ghostChildren[which_child]);
                  subgrid2 = IntersectSubgrids(s, subgrid1);
                }
                break;

              case +1:

                which_child = parflow_p4est_check_neigh(subgrid0, subgrid1, grid->pfgrid);

                /* We try to intersect a fine subgrid 'sf' with a coarse shifted
                 * one 'sc'. To get the right values we replace sf by its parent
                 * and then map the result of the intersection to sf. */
                if (which_child > 0)
                {
                  ix = SubgridIX(subgrid0);
                  iy = SubgridIY(subgrid0);
                  iz = SubgridIZ(subgrid0);

                  SubgridIX(subgrid0) = SubgridParentIX(subgrid0);
                  SubgridIY(subgrid0) = SubgridParentIY(subgrid0);
                  SubgridIZ(subgrid0) = SubgridParentIZ(subgrid0);

                  subgrid2 = IntersectSubgrids(subgrid0, subgrid1);

                  SubgridIX(subgrid0) = ix;
                  SubgridIY(subgrid0) = iy;
                  SubgridIZ(subgrid0) = iz;

                  if (subgrid2)
                  {
                    SubgridIX(subgrid2) =
                      SubgridIX(subgrid2) - SubgridParentIX(subgrid0) + ix;
                    SubgridIY(subgrid2) =
                      SubgridIY(subgrid2) - SubgridParentIY(subgrid0) + iy;
                    SubgridIZ(subgrid2) =
                      SubgridIZ(subgrid2) - SubgridParentIZ(subgrid0) + iz;

                    /* Remember 'ghost children' for later use if we are  computing
                     * the send region */
                    switch (r) {
                    case 0:
                        s = SubgridArraySubgrid(subgrids, subgrid1->ghostChildren[which_child]);
                        break;
                    case 1:
                        s = subgrid0;
                        break;
                    }
                  }
                }
                break;

              default:
                /* Same size neighbors, proceed as normal*/
                which_child = -1;
                P4EST_ASSERT(tweak == 0);
                subgrid2 = IntersectSubgrids(subgrid0, subgrid1);
                break;
            }
#endif
          }
          else
          {
            subgrid2 = IntersectSubgrids(subgrid0, subgrid1);
          }
          if ((subgrid2))
          {
            if (USE_P4EST)
            {
#ifdef HAVE_P4EST
              if (GridIsProjected(grid))
                SubregionIZ(subgrid2) = 0;
#endif
            }
            switch (r)
            {
              case 0:
                SubgridProcess(subgrid2) = SubgridProcess(subgrid1);
                if (USE_P4EST)
                {
#ifdef HAVE_P4EST
                  SubgridLocIdx(subgrid2) = (which_child > 0) ?
                              SubgridLocIdx(s) : SubgridLocIdx(subgrid1);
                  SubgridGhostIdx(subgrid2) = SubgridGhostIdx(subgrid1);
                  SubgridOwnerTree(subgrid2) = SubgridOwnerTree(subgrid1);
#endif
                }
                AppendSubgrid(subgrid2,
                              RegionSubregionArray(region1, i));
                break;

              case 1:
                SubgridProcess(subgrid2) = SubgridProcess(subgrid0);
                if (USE_P4EST)
                {
#ifdef HAVE_P4EST
                  SubgridLocIdx(subgrid2) = (which_child > 0) ?
                              SubgridLocIdx(s) : SubgridLocIdx(subgrid0);
                  SubgridGhostIdx(subgrid2) = SubgridGhostIdx(subgrid0);
                  SubgridOwnerTree(subgrid2) = SubgridOwnerTree(subgrid0);
#endif
                }
                AppendSubgrid(subgrid2,
                              RegionSubregionArray(region1, j));
                break;
            }
          }
        }
      }
    }

    switch (r)
    {
      case 0:
        send_region = region1;
        break;

      case 1:
        recv_region = region1;
        break;
    }
  }

  FreeRegion(subgrid_region);
  FreeRegion(neighbor_region);

  /*------------------------------------------------------
   * Union the send_region and recv_region by process
   *------------------------------------------------------*/

  proc_array = talloc(int, SubgridArraySize(neighbors));

  for (r = 0; r < 2; r++)
  {
    switch (r)
    {
      case 0:
        region0 = send_region;
        break;

      case 1:
        region0 = recv_region;
        break;
    }

    region1 = NewRegion(RegionSize(region0));

    ForSubregionArrayI(i, region0)
    {
      sa0 = RegionSubregionArray(region0, i);
      sa1 = RegionSubregionArray(region1, i);

      /* determine proc_array and num_procs */

      num_procs = 0;
      ForSubgridI(j, sa0)
      {
        subgrid0 = SubgridArraySubgrid(sa0, j);

        for (p = 0; p < num_procs; p++)
          if (SubgridProcess(subgrid0) == proc_array[p])
            break;
        if (p == num_procs)
        {
          proc_array[p] = SubgridProcess(subgrid0);
          num_procs++;
        }
      }

      /* union by process */

      for (p = 0; p < num_procs; p++)
      {
        /* put subgrids on proc_array[p] into sa2 */
        sa2 = NewSubgridArray();

        ForSubgridI(j, sa0)
        {
          subgrid0 = SubgridArraySubgrid(sa0, j);

          if (SubgridProcess(subgrid0) == proc_array[p])
            AppendSubgrid(subgrid0, sa2);
        }

        /* If p4est is activated we need to remember the local index and the
         * tree owning each subgrid. */
        if (USE_P4EST)
        {
#ifdef HAVE_P4EST
          BeginTiming(P4ESTSetupTimingIndex);

          tree_array = talloc(int, SubgridArraySize(neighbors));
          loc_idx_array = talloc(int, SubgridArraySize(neighbors));
          ghost_idx_array = talloc(int, SubgridArraySize(neighbors));

          /* Determine loc_idx_array, tree_array and num of
           * different local indexes and trees in sa2 */
          num_loc_idxs = num_trees = 0;
          ForSubgridI(j, sa2)
          {
            subgrid0 = SubgridArraySubgrid(sa2, j);

            for (k = 0; k < num_loc_idxs; k++)
              if (SubgridLocIdx(subgrid0) == loc_idx_array[k])
              {
                P4EST_ASSERT(SubgridGhostIdx(subgrid0) == ghost_idx_array[k]);
                break;
              }
            if (k == num_loc_idxs)
            {
              loc_idx_array[k] = SubgridLocIdx(subgrid0);
              ghost_idx_array[k] = SubgridGhostIdx(subgrid0);
              num_loc_idxs++;
            }

            for (tt = 0; tt < num_trees; tt++)
              if (SubgridOwnerTree(subgrid0) == tree_array[tt])
                break;
            if (tt == num_trees)
            {
              tree_array[tt] = SubgridOwnerTree(subgrid0);
              num_trees++;
            }
          }

          /* put subgrids with loc_idx_array[k] and array_tree[tt]
           * into sa2_loc */
          for (tt = 0; tt < num_trees; tt++)
          {
            for (k = 0; k < num_loc_idxs; k++)
            {
              sa2_loc = NewSubgridArray();

              ForSubgridI(j, sa2)
              {
                subgrid0 = SubgridArraySubgrid(sa2, j);
                if (SubgridLocIdx(subgrid0) == loc_idx_array[k] &&
                    SubgridOwnerTree(subgrid0) == tree_array[tt])
                  AppendSubgrid(subgrid0, sa2_loc);
              }

              /*Union of subgrids in sa2_loc*/
              sa3 = UnionSubgridArray(sa2_loc);

              /*Rembember processor, tree ghost and local indx*/
              ForSubgridI(j, sa3)
              {
                subgrid0 = SubgridArraySubgrid(sa3, j);
                SubgridProcess(subgrid0) = proc_array[p];
                SubgridOwnerTree(subgrid0) = tree_array[tt];
                SubgridLocIdx(subgrid0) = loc_idx_array[k];
                SubgridGhostIdx(subgrid0) = ghost_idx_array[k];

                /* Remember level and parent corner */
                subgrid1 = parflow_p4est_fetch_subgrid(subgrids, neighbors,
                                                       loc_idx_array[k], ghost_idx_array[k]);
                SubgridLevel(subgrid0) = SubgridLevel(subgrid1);
                SubgridParentIX(subgrid0) = SubgridParentIX(subgrid1);
                SubgridParentIY(subgrid0) = SubgridParentIY(subgrid1);
                SubgridParentIZ(subgrid0) = SubgridParentIZ(subgrid1);
              }
              AppendSubgridArray(sa3, sa1);

              SubregionArraySize(sa2_loc) = 0;
              FreeSubgridArray(sa2_loc);
              SubregionArraySize(sa3) = 0;
              FreeSubgridArray(sa3);
            }
          }

          SubregionArraySize(sa2) = 0;
          FreeSubgridArray(sa2);

          tfree(tree_array);
          tfree(loc_idx_array);
          tfree(ghost_idx_array);

          EndTiming(P4ESTSetupTimingIndex);
#endif
        }
        else
        {
          sa3 = UnionSubgridArray(sa2);
          ForSubgridI(j, sa3)
          SubgridProcess(SubgridArraySubgrid(sa3, j)) = proc_array[p];
          AppendSubgridArray(sa3, sa1);

          SubregionArraySize(sa2) = 0;
          FreeSubgridArray(sa2);
          SubregionArraySize(sa3) = 0;
          FreeSubgridArray(sa3);
        }
      }
    }

    FreeRegion(region0);

    switch (r)
    {
      case 0:
        send_region = region1;
        break;

      case 1:
        recv_region = region1;
        break;
    }
  }

  tfree(proc_array);

  /*------------------------------------------------------
   * Return
   *------------------------------------------------------*/
  if (!USE_P4EST)
  {
    SubregionArraySize(neighbors) = 0;
    FreeSubgridArray(neighbors);
  }
  *send_region_ptr = send_region;
  *recv_region_ptr = recv_region;
}



