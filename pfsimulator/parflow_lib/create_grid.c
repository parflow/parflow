/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
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
  globals->grid3d = grid;

  return grid;
}

