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
   int                Nx, Ny, Nz;
   int                mx, my, mz;
   int                owner_rank;
   int                Px, Py, Pz;
   int                lx, ly, lz;
   int                px, py, pz;
   double             x, y, z;
   double             level_factor, v[3];
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

   Nx = SubgridNX(user_subgrid);
   Ny = SubgridNY(user_subgrid);
   Nz = SubgridNZ(user_subgrid);

   mx = GlobalsSubrgridPointsX;
   my = GlobalsSubrgridPointsY;
   mz = GlobalsSubrgridPointsZ;

  /* Compute number of subgrids per coordinate direction. */
   Px = Nx / mx;
   Py = Ny / my;
   Pz = (Nz == 1) ? 1 : Nz / mz;

   lx = Nx % mx;
   ly = Ny % my;
   lz = Nz % mz;

   /* Create the p{4,8}est object. */
   grid->pfgrid = parflow_p4est_grid_new (Px, Py, Pz);

   forest =  grid->pfgrid->forest;

   /* Loop over the trees un the forest */
   for (tt = forest->first_local_tree, k = 0;
        tt <= forest->last_local_tree; ++tt) {

       tree = p4est_tree_array_index (forest->trees, tt);
       tquadrants = &tree->quadrants;
       Q = (int) tquadrants->elem_count;
       P4EST_ASSERT( Q > 0 );

       /* Loop on the quadrants (leafs) of this forest
          and attach a subgrid on each */
       for (q = 0; q < Q; ++q, ++k) {
           quad = p4est_quadrant_array_index (tquadrants, q);
           parflow_p4est_qcoord_to_vertex (grid->pfgrid, tt, quad, v);
           level_factor = pow (2., quad->level);
           owner_rank = parflow_p4est_quad_owner_rank(quad);

           /* Get bottom left corner (anchor node) for the
              new subgrid */
           x = level_factor * v[0];
           y = level_factor * v[1];
           z = level_factor * v[2];

           /* Decide the dimensions for the new subgrid */
           px = (int) x < lx ? mx + 1 : mx;
           py = (int) y < ly ? my + 1 : my;
           if (Nz > 1){
              pz = (int) z < lz ? mz + 1 : mz;
           }else{
              pz = 1;
           }

           /* Allocate new subgrid and attach it to this quadrant */
           quad->p.user_data =
               (void *) NewSubgrid( x,  y,  z, px, py, pz,
                                    0,  0,  0, owner_rank);
       }
     }

     /* Assert that every quadrant was visited */
     P4EST_ASSERT( k == (int) forest->local_num_quadrants );
#endif

   return grid;
}

