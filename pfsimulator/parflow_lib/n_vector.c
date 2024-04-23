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

#include "parflow.h"

static struct {
  Grid *grid;
  int num_ghost;
} pf2kinsol_data;


void SetPf2KinsolData(
                      Grid *grid,
                      int   num_ghost)
{
  pf2kinsol_data.grid = grid;
  pf2kinsol_data.num_ghost = num_ghost;
}

N_Vector N_VNew(
                int   N,
                void *machEnv)
{
  Grid    *grid;
  int num_ghost;

  (void)N;
  (void)machEnv;

  grid = pf2kinsol_data.grid;
  num_ghost = pf2kinsol_data.num_ghost;
  return(NewVectorType(grid, 1, num_ghost, vector_cell_centered));
}

void N_VPrint(
              N_Vector x)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;

  double     *xp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;

  int sg, i, j, k, i_x;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    x_sub = VectorSubvector(x, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);

    i_x = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
    {
      printf("%g\n", xp[i_x]);
      fflush(NULL);
    });
  }
  printf("\n");
  fflush(NULL);
}
