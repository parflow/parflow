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

/*
 * Setup array for storing the top of the domain.
 *
 * Computes and array that is NX * NY that contains the
 * k-index into the supplied vector that is at the top
 * of the geometry.
 *
 * Only works with 1 subgrid per task.
 *
 * This assumes number of processors is 1 in Z; assumes
 * that the entire Z column is on a single task.
 *
 */

#include "parflow.h"

void ComputeTopAndBottom(Problem *    problem,      /* General problem information */
                         ProblemData *problem_data  /* Contains geometry information for the problem */
                         )
{
  GrGeomSolid   *gr_solid = ProblemDataGrDomain(problem_data);
  Vector        *top = ProblemDataIndexOfDomainTop(problem_data);
  Vector        *bottom = ProblemDataIndexOfDomainBottom(problem_data);
  Vector        *perm_x = ProblemDataPermeabilityX(problem_data);

  Grid          *grid2d = VectorGrid(top);
  SubgridArray  *grid2d_subgrids = GridSubgrids(grid2d);

  /* use perm grid as top is 2D and want to loop over Z */
  Grid          *grid3d = VectorGrid(perm_x);
  SubgridArray  *grid3d_subgrids = GridSubgrids(grid3d);


  double *top_data, *bottom_data;
  int index;

  VectorUpdateCommHandle   *handle;

  (void)problem;

  InitVectorAll(top, -1);
  InitVectorAll(bottom, -1);

  int is;
  ForSubgridI(is, grid3d_subgrids)
  {
    Subgrid       *grid2d_subgrid = SubgridArraySubgrid(grid2d_subgrids, is);
    Subgrid       *grid3d_subgrid = SubgridArraySubgrid(grid3d_subgrids, is);

    Subvector     *top_subvector = VectorSubvector(top, is);
    Subvector     *bottom_subvector = VectorSubvector(bottom, is);

    int grid3d_ix = SubgridIX(grid3d_subgrid);
    int grid3d_iy = SubgridIY(grid3d_subgrid);
    int grid3d_iz = SubgridIZ(grid3d_subgrid);

    int grid2d_iz = SubgridIZ(grid2d_subgrid);

    int grid3d_nx = SubgridNX(grid3d_subgrid);
    int grid3d_ny = SubgridNY(grid3d_subgrid);
    int grid3d_nz = SubgridNZ(grid3d_subgrid);

    int grid3d_r = SubgridRX(grid3d_subgrid);

    top_data = SubvectorData(top_subvector);
    bottom_data = SubvectorData(bottom_subvector);

    int i, j, k;
    GrGeomInLoop(i, j, k,
                 gr_solid, grid3d_r,
                 grid3d_ix, grid3d_iy, grid3d_iz,
                 grid3d_nx, grid3d_ny, grid3d_nz,
    {
      index = SubvectorEltIndex(top_subvector, i, j, grid2d_iz);

      if (top_data[index] < k)
      {
        top_data[index] = k;
      }

      if (bottom_data[index] > k || bottom_data[index] < 0)
      {
        bottom_data[index] = k;
      }
    });
  }      /* End of subgrid loop */

  /* Pass top values to neighbors.  */
  handle = InitVectorUpdate(top, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  /* Pass bottom values to neighbors.  */
  handle = InitVectorUpdate(bottom, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
}
