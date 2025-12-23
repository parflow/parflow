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
#include "top.h"

#include <stdio.h>
#include <math.h>


/*-----------------------------------------------------------------------
 * ComputeTop:
 *
 * Computes the top indices of the computation domain as defined by
 * the mask values.  * Mask has values 0 outside of domain so first
 * non-zero entry is the top.
 *
 * Returns a top Databox with (z) indices of the top surface for each
 * i,j location.
 *
 *-----------------------------------------------------------------------*/

void ComputeTop(Databox *mask, Databox  *top)
{
  int i, j, k;
  int nx, ny, nz;

  nx = DataboxNx(mask);
  ny = DataboxNy(mask);
  nz = DataboxNz(mask);

  for (j = 0; j < ny; j++)
  {
    for (i = 0; i < nx; i++)
    {
      for (k = nz - 1; k >= 0; --k)
      {
        if (*(DataboxCoeff(mask, i, j, k)) > 0.0)
        {
          break;
        }
      }

      if (k >= 0)
      {
        *(DataboxCoeff(top, i, j, 0)) = k;
      }
      else
      {
        *(DataboxCoeff(top, i, j, 0)) = -1;
      }
    }
  }
}

void ComputeBottom(Databox *mask, Databox  *bottom)
{
  int i, j, k;
  int nx, ny, nz;

  nx = DataboxNx(mask);
  ny = DataboxNy(mask);
  nz = DataboxNz(mask);

  for (j = 0; j < ny; j++)
  {
    for (i = 0; i < nx; i++)
    {
      for (k = 0; k < nz; ++k)
      {
        if (*(DataboxCoeff(mask, i, j, k)) > 0.0)
        {
          break;
        }
      }

      if (k >= 0)
      {
        *(DataboxCoeff(bottom, i, j, 0)) = k;
      }
      else
      {
        *(DataboxCoeff(bottom, i, j, 0)) = -1;
      }
    }
  }
}


/*-----------------------------------------------------------------------
 * ExtractTop:
 *
 * Extracts the top values of a dataset based on a top dataset (which contains the
 * z indices that define the top of the domain).
 *
 * Returns a Databox with top values extracted for each i,j location.
 *
 *-----------------------------------------------------------------------*/

void ExtractTop(Databox *top, Databox  *data, Databox *top_values_of_data)
{
  int i, j;
  int nx, ny, nz;

  nx = DataboxNx(data);
  ny = DataboxNy(data);
  nz = DataboxNz(data);

  for (j = 0; j < ny; j++)
  {
    for (i = 0; i < nx; i++)
    {
      int k = *(DataboxCoeff(top, i, j, 0));
      if (k < 0)
      {
        /* outside domain what value? */
        *(DataboxCoeff(top_values_of_data, i, j, 0)) = 0.0;
      }
      else if (k < nz)
      {
        *(DataboxCoeff(top_values_of_data, i, j, 0)) = *(DataboxCoeff(data, i, j, k));
      }
      else
      {
        printf("Error: Index in top (k=%d) is outside of data (nz=%d)\n", k, nz);
      }
    }
  }
}
