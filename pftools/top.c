/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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
#include <stdbool.h>

void ComputeTop(Databox *mask, Databox  *top)
{
  int nx = DataboxNx(mask);
  int ny = DataboxNy(mask);
  int nz = DataboxNz(mask);

  for (int j = 0; j < ny; j++)
  {
    for (int i = 0; i < nx; i++)
    {
      int k;
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
  int nx = DataboxNx(mask);
  int ny = DataboxNy(mask);
  int nz = DataboxNz(mask);

  for (int j = 0; j < ny; j++)
  {
    for (int i = 0; i < nx; i++)
    {
      int k;
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

void ExtractTop(Databox *top, Databox  *data, Databox *top_data)
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
        *(DataboxCoeff(top_data, i, j, 0)) = 0.0;
      }
      else if (k < nz)
      {
        *(DataboxCoeff(top_data, i, j, 0)) = *(DataboxCoeff(data, i, j, k));
      }
      else
      {
        printf("Error: Index in top (k=%d) is outside of data (nz=%d)\n", k, nz);
      }
    }
  }
}

void ExtractTopBoundary(Databox *top, Databox  *data, Databox *boundary_data)
{
  int nx = DataboxNx(data);
  int ny = DataboxNy(data);

  for (int j = 0; j < ny; j++)
  {
    bool inside = false;

    for (int i = 0; i < nx; i++)
    {
      int k = *(DataboxCoeff(top, i, j, 0));

      if (inside)
      {
	if (k < 0)
	{
	  /* inside transition to outside */
	  *(DataboxCoeff(boundary_data, i-1, j, 0)) = *(DataboxCoeff(data, i, j, 0));
	  inside = false;
	}
      }
      else
      {
	if ( k >= 0)
	{
	  /* outside transition to inside */
	  *(DataboxCoeff(boundary_data, i, j, 0)) = *(DataboxCoeff(data, i, j, 0));
	  inside = true;
	}
      }
    } // for i

    /* Take care of domain that ends along index space boundary, will be no final 
       transition inside to outside in this case */
    if (inside)
    {
      *(DataboxCoeff(boundary_data, nx-1, j, 0)) = *(DataboxCoeff(data, nx-1, j, 0));
    }
  }
}
