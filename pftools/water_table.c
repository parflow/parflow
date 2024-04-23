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
#include "water_table.h"

/*-----------------------------------------------------------------------
 * ComputeWaterTableDepth:
 *
 * Computes the water table depth as the first cell with a saturation=1 starting
 * from top.   Depth is depth below the top surface.
 *
 * Negative values indicate the water table was not found, either below domain or
 * the column at (i,j) is outside of domain
 *
 * Returns a Databox water_table_depth with depth values at
 * each (i,j) location.
 *
 *-----------------------------------------------------------------------*/

#include <math.h>

void ComputeWaterTableDepth(
                            Databox *top,
                            Databox *saturation,
                            Databox *water_table_depth)
{
  int i, j;
  int nx, ny, nz;
  double dz;

  nx = DataboxNx(saturation);
  ny = DataboxNy(saturation);
  nz = DataboxNz(saturation);

  dz = DataboxDz(saturation);

  for (j = 0; j < ny; j++)
  {
    for (i = 0; i < nx; i++)
    {
      int top_k = *(DataboxCoeff(top, i, j, 0));
      if (top_k < 0)
      {
        /* inactive column so set to bogus value */
        *(DataboxCoeff(water_table_depth, i, j, 0)) = -9999999.0;
      }
      else if (top_k < nz)
      {
        /* loop down until we find saturation => 1 */
        int k = top_k;
        while ((k >= 0) && (*(DataboxCoeff(saturation, i, j, k)) < 1))
        {
          k--;
        }

        /*
         * Make sure water table was found in the column, set to bogus value
         * if it was not.
         */
        if (k >= 0)
        {
          *(DataboxCoeff(water_table_depth, i, j, 0)) = (top_k - k) * dz;
        }
        else
        {
          *(DataboxCoeff(water_table_depth, i, j, 0)) = -9999999.0;
        }
      }
      else
      {
        printf("Error: Index in top (k=%d) is outside of domain (nz=%d)\n", top_k, nz);
      }
    }
  }
}


