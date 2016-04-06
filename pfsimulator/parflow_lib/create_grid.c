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
   parflow_p4est_sg_param_t subgparam, *sp = &subgparam;
   Subgrid                  *user_subgrid;
   parflow_p4est_qiter_t    *qiter;
   parflow_p4est_quad_data_t  *quad_data;
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

   /* Initialize information to compute number of subgrids
    * and their corresponding dimensions */
   parflow_p4est_sg_param_init(sp);

   /* Create the pfgrid. */
   grid->pfgrid = parflow_p4est_grid_new (sp->P[0], sp->P[1], sp->P[2]);

   /* Loop on the quadrants (leafs) of this forest
      and attach a subgrid on each */
   for (qiter = parflow_p4est_qiter_init(grid->pfgrid, PARFLOW_P4EST_QUAD);
        qiter != NULL;
        qiter = parflow_p4est_qiter_next(qiter)) {

       /* Update paramenters to decide dimensions for the new subgrid */
       parflow_p4est_sg_param_update(qiter, sp);

       /* Allocate new subgrid and attach it to this quadrant */
       quad_data = parflow_p4est_get_quad_data(qiter);
       quad_data->pf_subgrid =
           NewSubgrid(sp->icorner[0], sp->icorner[1], sp->icorner[2],
                      sp->p[0], sp->p[1], sp->p[2], 0,  0,  0,
                      parflow_p4est_qiter_get_owner_rank(qiter));
    }

   /* Loop over the ghost layer */
    for (qiter = parflow_p4est_qiter_init(grid->pfgrid, PARFLOW_P4EST_GHOST);
         qiter != NULL;
         qiter = parflow_p4est_qiter_next(qiter)) {

        /* Update paramenters to decide dimensions for the new subgrid */
        parflow_p4est_sg_param_update(qiter, sp);

       /* Allocate new subgrid and attach it to the corresponding
        * ghost_data structure */
        ghost_data = parflow_p4est_get_ghost_data(grid->pfgrid, qiter);
        ghost_data->pf_subgrid =
            NewSubgrid(sp->icorner[0], sp->icorner[1], sp->icorner[2],
                       sp->p[0], sp->p[1], sp->p[2], 0,  0,  0,
                       parflow_p4est_qiter_get_owner_rank(qiter));
   }
#endif

   return grid;
}

