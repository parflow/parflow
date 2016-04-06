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
#ifdef HAVE_P4EST
#include "../p4est_test/parflow_p4est.h"
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
   int                Nx, Ny, Nz;
   int                mx, my, mz;
   int                Px, Py, Pz;
   int                lx, ly, lz;
   int                px, py, pz;
   int                ix, iy, iz;
   int                ghost_idx;
   double             v[3];
   Subgrid            *user_subgrid, **ss;
   parflow_p4est_qiter_t *qiter;
   parflow_p4est_quad_data_t *quad_data;
   parflow_p4est_ghost_data_t *ghost_data;
#endif

#ifndef HAVE_P4EST
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

#else
   grid = talloc(Grid, 1);
   user_subgrid = GridSubgrid(user_grid, 0);

   Nx = SubgridNX(user_subgrid);
   Ny = SubgridNY(user_subgrid);
   Nz = SubgridNZ(user_subgrid);

   mx = GlobalsSubgridPointsX;
   my = GlobalsSubgridPointsY;
   mz = GlobalsSubgridPointsZ;

   /* Compute number of subgrids per coordinate direction. */
   Px = Nx / mx;
   Py = Ny / my;
   Pz = (Nz == 1) ? 1 : Nz / mz;

   lx = Nx % mx;
   ly = Ny % my;
   lz = Nz % mz;

   /* Create the pfgrid. */
   grid->pfgrid = parflow_p4est_grid_new (Px, Py, Pz);

   /* Loop on the quadrants (leafs) of this forest
      and attach a subgrid on each */
   for (qiter = parflow_p4est_qiter_init(grid->pfgrid, PARFLOW_P4EST_QUAD);
        qiter != NULL;
        qiter = parflow_p4est_qiter_next(qiter)) {

       /* Get bottom left corner (anchor node)  in
            * index space for the new subgrid */
       parflow_p4est_qiter_qcorner(qiter, v);

       /* Decide the dimensions for the new subgrid */
       ix = (int) v[0];
       iy = (int) v[1];
       iz = (int) v[2];

       px =  ix < lx ? mx + 1 : mx;
       py =  iy < ly ? my + 1 : my;
       if (Nz > 1){
           pz = iz < lz ? mz + 1 : mz;
       }
       else{
           pz = 1;
       }

       /* Allocate new subgrid and attach it to this quadrant */
       quad_data = parflow_p4est_qiter_get_data(qiter);
       quad_data->pf_subgrid = (Subgrid_t *) NewSubgrid(ix, iy, iz, px, py, pz, 0,  0,  0,
                                             parflow_p4est_qiter_get_owner_rank(qiter));
    }

   ghost_data = parflow_p4est_get_ghost_data(grid->pfgrid);
   ss = (Subgrid **) ghost_data->ghost_subgrids->array;
   /* Loop over the ghost layer */
    for (qiter = parflow_p4est_qiter_init(grid->pfgrid, PARFLOW_P4EST_GHOST);
         qiter != NULL;
         qiter = parflow_p4est_qiter_next(qiter)) {

        /* Get bottom left corner (anchor node)  in
             * index space for the new subgrid */
        parflow_p4est_qiter_qcorner(qiter, v);

        /* Decide the dimensions for the new subgrid */
        ix = (int) v[0];
        iy = (int) v[1];
        iz = (int) v[2];

        px =  ix < lx ? mx + 1 : mx;
        py =  iy < ly ? my + 1 : my;
        if (Nz > 1){
            pz = iz < lz ? mz + 1 : mz;
          }
        else{
            pz = 1;
          }

       /* Allocate new subgrid and attach it to the corresponding
        * ghost quadrant position */
        ghost_idx = parflow_p4est_qiter_get_ghost_idx(qiter);
        ss[ghost_idx] = NewSubgrid(ix, iy, iz, px, py, pz, 0,  0,  0,
                                     parflow_p4est_qiter_get_owner_rank(qiter));
   }
#endif

   return grid;
}

