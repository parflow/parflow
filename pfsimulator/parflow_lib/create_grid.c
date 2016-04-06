/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER*/

/******************************************************************************
 *
 * Create the distributed grid from the user-grid input
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * GetGridSubgrids:
 *   Returns a SubgridArray containing subgrids on my process.
 *
 *   Notes:
 *     The ordering of `subgrids' is inferred by `all_subgrids'.
 *     The `subgrids' array points to subgrids in `all_subgrids'.
 *--------------------------------------------------------------------------*/

SubgridArray  *GetGridSubgrids(
   SubgridArray  *all_subgrids)
{
   SubgridArray  *subgrids;

   Subgrid       *s;

   int            i, my_proc;


   my_proc   = amps_Rank(amps_CommWorld);

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
   Grid           *user_grid)
{
   Grid        *grid;

   SubgridArray  *subgrids;
   SubgridArray  *all_subgrids;

#ifdef HAVE_P4EST
   int                q, k, Q;
   int                nx, ny, nz;
   int                num_procs, owner_rank;
   int                Px, Py, Pz;
   double             v[3];
   Subgrid            *user_subgrid;
   sc_array_t         *tquadrants;
   p4est_t            *forest;
   p4est_topidx_t      tt;
   p4est_tree_t       *tree;
   p4est_quadrant_t   *quad;
#endif

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

   /*-----------------------------------------------------------------------
    * Create the grid.
    *-----------------------------------------------------------------------*/

   grid = NewGrid(subgrids, all_subgrids);

   /*-----------------------------------------------------------------------
    * Create communication packages.
    *-----------------------------------------------------------------------*/

   CreateComputePkgs(grid);

   // SGS Debug
   globals -> grid3d = grid;

#ifdef HAVE_P4EST

   user_subgrid = GridSubgrid(user_grid, 0);

   nx = SubgridNX(user_subgrid);
   ny = SubgridNY(user_subgrid);
   nz = SubgridNZ(user_subgrid);

   Px = pfmax (nx - 1, 1);
   Py = pfmax (ny - 1, 1);
   Pz = pfmax (nz - 1, 1);

   num_procs = GlobalsNumProcs;

   /*  Make sure there will be as much
      processors as quadrants. Only for step1 */
   P4EST_ASSERT( num_procs == Px*Py*Pz );

   /* Create the p{4,8}est object. */
   grid->pfgrid = parflow_p4est_grid_new (nx, ny, nz);

   forest =  grid->pfgrid->forest;

   //printf ("\n=S= Local num quadrants %i \n", (int) forest->local_num_quadrants);

   /* loop over al quadrants to attach a subgrid on it */
   for (tt = forest->first_local_tree, k = 0;
        tt <= forest->last_local_tree; ++tt) {

       tree = p4est_tree_array_index (forest->trees, tt);
       tquadrants = &tree->quadrants;
       Q = (int) tquadrants->elem_count;
       P4EST_ASSERT( Q > 0 );

       for (q = 0; q < Q; ++q, ++k) {
           quad = p4est_quadrant_array_index (tquadrants, q);
           parflow_p4est_qcoord_to_vertex (grid->pfgrid, tt, quad, v);
           owner_rank = parflow_p4est_quad_owner_rank(quad);
           quad->p.user_data =
               (void *) NewSubgrid(v[0], v[1], v[2],
                                   8, 8, 8,
                                   0, 0, 0,
                                   owner_rank);

       }
     }

     P4EST_ASSERT( k == (int) forest->local_num_quadrants );
#endif

   return grid;
}

