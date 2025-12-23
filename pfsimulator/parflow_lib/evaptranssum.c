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

void EvapTransSum(ProblemData *problem_data, double dt, Vector *evap_trans_sum, Vector *evap_trans)
{
  GrGeomSolid *gr_domain = ProblemDataGrDomain(problem_data);

  double dx, dy, dz;
  int i, j, k, r, is;
  int ix, iy, iz;
  int nx, ny, nz;

  Subgrid     *subgrid;
  Grid        *grid = VectorGrid(evap_trans_sum);

  Subvector   *evap_trans_sum_subvector;
  Subvector   *evap_trans_subvector;

  double vol_time;

  int index_evap_trans_sum;
  int index_evap_trans;

  double *evap_trans_sum_ptr;
  double *evap_trans_ptr;

  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    evap_trans_sum_subvector = VectorSubvector(evap_trans_sum, is);
    evap_trans_subvector = VectorSubvector(evap_trans, is);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    r = SubgridRX(subgrid);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    vol_time = dx * dy * dz * dt;

    evap_trans_sum_ptr = SubvectorData(evap_trans_sum_subvector);
    evap_trans_ptr = SubvectorData(evap_trans_subvector);

    GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
    {
      index_evap_trans_sum = SubvectorEltIndex(evap_trans_sum_subvector, i, j, k);
      index_evap_trans = SubvectorEltIndex(evap_trans_subvector, i, j, k);

      evap_trans_sum_ptr[index_evap_trans_sum] += evap_trans_ptr[index_evap_trans] * vol_time;
    });
  }
}
