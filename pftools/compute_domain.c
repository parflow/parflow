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
#include <stdio.h>
#include <math.h>

#include "compute_domain.h"

/* Locally defined integer min/max functions */
static inline int domain_max(int a, int b)
{
  return a > b ? a : b;
};
static inline int domain_min(int a, int b)
{
  return a < b ? a : b;
};

void ComputeDomain(
                   SubgridArray *all_subgrids,
                   Databox *     top,
                   Databox *     bottom,
                   int           P,
                   int           Q,
                   int           R)
{
  int num_procs = P * Q * R;
  int p;

  // For each subgrid find the min/max k values
  // in the active region (using top/bottom).
  // Reset the subgrid to reflect this vertical extent.
  for (p = 0; p < num_procs; p++)
  {
    int s_i;
    ForSubgridI(s_i, all_subgrids)
    {
      Subgrid* subgrid = SubgridArraySubgrid(all_subgrids, s_i);

      int process = SubgridProcess(subgrid);

      if (process == p)
      {
        int i;
        int j;

        int ix = SubgridIX(subgrid);
        int iy = SubgridIY(subgrid);
        int iz = SubgridIZ(subgrid);

        int nx = SubgridNX(subgrid);
        int ny = SubgridNY(subgrid);
        int nz = SubgridNZ(subgrid);

        int patch_top = iz;
        int patch_bottom = iz + nz;

        for (j = iy; j < iy + ny; ++j)
        {
          for (i = ix; i < ix + nx; ++i)
          {
            int k_top = *(DataboxCoeff(top, i, j, 0));
            if (k_top >= 0)
            {
              patch_top = domain_max(patch_top, k_top);
            }

            int k_bottom = *(DataboxCoeff(bottom, i, j, 0));
            if (k_bottom >= 0)
            {
              patch_bottom = domain_min(patch_bottom, k_bottom);
            }
          }
        }

        // adjust grid to include 2 pad cells
        patch_top = domain_min(patch_top + 2, iz + nz - 1);
        patch_bottom = domain_max(patch_bottom - 2, iz);

        // adjust for ghost cells, need to have patches
        // that extend in height to the neighbor patches.
        //
        // There is a more efficient way to compute all this but
        // since these are 2d arrays it should be reasonably quick.
        // Not a single loop since we don't need to pad these values.
        ix = domain_max(0, ix - 1);
        nx = domain_min(DataboxNx(top) - ix, nx + 2 - ix);

        iy = domain_max(0, iy - 1);
        ny = domain_min(DataboxNy(top) - iy, ny + 2 - iy);

        for (j = iy; j < iy + ny; ++j)
        {
          for (i = ix; i < ix + nx; ++i)
          {
            int k_top = *(DataboxCoeff(top, i, j, 0));
            if (k_top >= 0)
            {
              patch_top = domain_max(patch_top, k_top);
            }

            int k_bottom = *(DataboxCoeff(bottom, i, j, 0));
            if (k_bottom >= 0)
            {
              patch_bottom = domain_min(patch_bottom, k_bottom);
            }
          }
        }

        SubgridIZ(subgrid) = patch_bottom;
        SubgridNZ(subgrid) = (patch_top - SubgridIZ(subgrid)) + 1;
      }
    }
  }
}

SubgridArray  *Extract2DDomain(
                               SubgridArray *all_subgrids)
{
  SubgridArray  *new_subgrids;
  int s_i;

  new_subgrids = CopyGrid(all_subgrids);

  ForSubgridI(s_i, all_subgrids)
  {
    Subgrid* subgrid = SubgridArraySubgrid(new_subgrids, s_i);

    SubgridIZ(subgrid) = 0;
    SubgridNZ(subgrid) = 1;
  }

  return new_subgrids;
}


