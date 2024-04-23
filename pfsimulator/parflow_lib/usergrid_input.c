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
* ReadUserSubgrid, ReadUserGrid,
* FreeUserGrid
*
* Routines for reading user_grid input.
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * ReadUserSubgrid
 *--------------------------------------------------------------------------*/

Subgrid    *ReadUserSubgrid()
{
  Subgrid  *new_subgrid;

  int ix, iy, iz;
  int nx, ny, nz;
  int rx, ry, rz;

  ix = GetIntDefault("UserGrid.IX", 0);
  iy = GetIntDefault("UserGrid.IY", 0);
  iz = GetIntDefault("UserGrid.IZ", 0);

  rx = GetIntDefault("UserGrid.RX", 0);
  ry = GetIntDefault("UserGrid.RY", 0);
  rz = GetIntDefault("UserGrid.RZ", 0);

  nx = GetInt("ComputationalGrid.NX");
  ny = GetInt("ComputationalGrid.NY");
  nz = GetInt("ComputationalGrid.NZ");

  new_subgrid = NewSubgrid(ix, iy, iz, nx, ny, nz, rx, ry, rz, -1);

  return new_subgrid;
}

/*--------------------------------------------------------------------------
 * ReadUserGrid
 *--------------------------------------------------------------------------*/

Grid      *ReadUserGrid()
{
  Grid          *user_grid;

  SubgridArray  *user_all_subgrids;
  SubgridArray  *user_subgrids;

  int num_user_subgrids;

  int i;


  num_user_subgrids = GetIntDefault("UserGrid.NumSubgrids", 1);

  /* read user_subgrids */
  user_all_subgrids = NewSubgridArray();
  user_subgrids = NewSubgridArray();
  for (i = 0; i < num_user_subgrids; i++)
  {
    AppendSubgrid(ReadUserSubgrid(), user_all_subgrids);
    AppendSubgrid(SubgridArraySubgrid(user_all_subgrids, i), user_subgrids);
  }

  /* create user_grid */
  user_grid = NewGrid(user_subgrids, user_all_subgrids);

  return user_grid;
}


/*--------------------------------------------------------------------------
 * FreeUserGrid
 *--------------------------------------------------------------------------*/

void  FreeUserGrid(
                   Grid *user_grid)
{
  FreeGrid(user_grid);
}

