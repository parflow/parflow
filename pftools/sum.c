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

#include "databox.h"


/*****************************************************************************
* Sum
*
*****************************************************************************/


/*-----------------------------------------------------------------------
 * Compute x = Sum of all elements of X
 *-----------------------------------------------------------------------*/

void       Sum(Databox *X, double *sum)
{
  int nx, ny, nz;
  double         *xp;
  int m;

  nx = DataboxNx(X);
  ny = DataboxNy(X);
  nz = DataboxNz(X);

  xp = DataboxCoeffs(X);

  m = 0;

  *sum = 0;

  for (m = 0; m < (nx * ny * nz); m++)
  {
    *sum += xp[m];
  }
}


/*-----------------------------------------------------------------------
 * Compute x = cell-wise sum of X and Y
 *-----------------------------------------------------------------------*/
void       CellSum(Databox *X, Databox *Y, Databox *mask, Databox *sum)
{
  int m, nx, ny, nz;
  double         *xp, *yp;
  double         *mask_val;
  double         *sum_val;

  nx = DataboxNx(X);
  ny = DataboxNy(X);
  nz = DataboxNz(X);

  xp = DataboxCoeffs(X);
  yp = DataboxCoeffs(Y);
  mask_val = DataboxCoeffs(mask);
  sum_val = DataboxCoeffs(sum);

  for (m = 0; m < (nx * ny * nz); m++)
  {
    if (mask_val[m] > 0)
    {
      sum_val[m] = xp[m] + yp[m];
    }
  }
}


/*-----------------------------------------------------------------------
 * Compute x = cell-wise difference of X and Y
 *-----------------------------------------------------------------------*/
void       CellDiff(Databox *X, Databox *Y, Databox *mask, Databox *diff)
{
  int m, nx, ny, nz;
  double         *xp, *yp;
  double         *mask_val;
  double         *diff_val;

  nx = DataboxNx(X);
  ny = DataboxNy(X);
  nz = DataboxNz(X);

  xp = DataboxCoeffs(X);
  yp = DataboxCoeffs(Y);
  mask_val = DataboxCoeffs(mask);
  diff_val = DataboxCoeffs(diff);

  for (m = 0; m < (nx * ny * nz); m++)
  {
    if (mask_val[m] > 0)
    {
      diff_val[m] = xp[m] - yp[m];
    }
  }
}


/*-----------------------------------------------------------------------
 * Compute x = cell-wise product of X and Y
 *-----------------------------------------------------------------------*/
void       CellMult(Databox *X, Databox *Y, Databox *mask, Databox *mult)
{
  int m, nx, ny, nz;
  double         *xp, *yp;
  double         *mask_val;
  double         *mult_val;

  nx = DataboxNx(X);
  ny = DataboxNy(X);
  nz = DataboxNz(X);

  xp = DataboxCoeffs(X);
  yp = DataboxCoeffs(Y);
  mask_val = DataboxCoeffs(mask);
  mult_val = DataboxCoeffs(mult);

  for (m = 0; m < (nx * ny * nz); m++)
  {
    if (mask_val[m] > 0)
    {
      mult_val[m] = xp[m] * yp[m];
    }
  }
}


/*-----------------------------------------------------------------------
 * Compute x = cell-wise quotient of X and Y (X/Y)
 *-----------------------------------------------------------------------*/
void       CellDiv(Databox *X, Databox *Y, Databox *mask, Databox *div)
{
  int m, nx, ny, nz;
  double         *xp, *yp;
  double         *mask_val;
  double         *div_val;

  nx = DataboxNx(X);
  ny = DataboxNy(X);
  nz = DataboxNz(X);

  xp = DataboxCoeffs(X);
  yp = DataboxCoeffs(Y);
  mask_val = DataboxCoeffs(mask);
  div_val = DataboxCoeffs(div);

  for (m = 0; m < (nx * ny * nz); m++)
  {
    if (mask_val[m] > 0)
    {
      div_val[m] = xp[m] / yp[m];
    }
  }
}


/*-----------------------------------------------------------------------
 * Compute x = cell-wise sum of X and <constant>
 *-----------------------------------------------------------------------*/
void       CellSumConst(Databox *X, double val, Databox *mask, Databox *sum)
{
  int m, nx, ny, nz;
  double         *xp;
  double         *mask_val;
  double         *sum_val;

  nx = DataboxNx(X);
  ny = DataboxNy(X);
  nz = DataboxNz(X);

  xp = DataboxCoeffs(X);
  mask_val = DataboxCoeffs(mask);
  sum_val = DataboxCoeffs(sum);

  for (m = 0; m < (nx * ny * nz); m++)
  {
    if (mask_val[m] > 0)
    {
      sum_val[m] = xp[m] + val;
    }
  }
}


/*-----------------------------------------------------------------------
 * Compute x = cell-wise difference of X and <constant>
 *-----------------------------------------------------------------------*/
void       CellDiffConst(Databox *X, double val, Databox *mask, Databox *diff)
{
  int m, nx, ny, nz;
  double         *xp;
  double         *mask_val;
  double         *diff_val;

  nx = DataboxNx(X);
  ny = DataboxNy(X);
  nz = DataboxNz(X);

  xp = DataboxCoeffs(X);
  mask_val = DataboxCoeffs(mask);
  diff_val = DataboxCoeffs(diff);

  for (m = 0; m < (nx * ny * nz); m++)
  {
    if (mask_val[m] > 0)
    {
      diff_val[m] = xp[m] - val;
    }
  }
}


/*-----------------------------------------------------------------------
 * Compute x = cell-wise product of X and <constant>
 *-----------------------------------------------------------------------*/
void       CellMultConst(Databox *X, double val, Databox *mask, Databox *mult)
{
  int m, nx, ny, nz;
  double         *xp;
  double         *mask_val;
  double         *mult_val;

  nx = DataboxNx(X);
  ny = DataboxNy(X);
  nz = DataboxNz(X);

  xp = DataboxCoeffs(X);
  mask_val = DataboxCoeffs(mask);
  mult_val = DataboxCoeffs(mult);

  for (m = 0; m < (nx * ny * nz); m++)
  {
    if (mask_val[m] > 0)
    {
      mult_val[m] = xp[m] * val;
    }
  }
}


/*-----------------------------------------------------------------------
 * Compute x = cell-wise quotient of X and <constant>
 *-----------------------------------------------------------------------*/
void       CellDivConst(Databox *X, double val, Databox *mask, Databox *div)
{
  int m, nx, ny, nz;
  double         *xp;
  double         *mask_val;
  double         *div_val;

  nx = DataboxNx(X);
  ny = DataboxNy(X);
  nz = DataboxNz(X);

  xp = DataboxCoeffs(X);
  mask_val = DataboxCoeffs(mask);
  div_val = DataboxCoeffs(div);

  for (m = 0; m < (nx * ny * nz); m++)
  {
    if (mask_val[m] > 0)
    {
      div_val[m] = xp[m] / val;
    }
  }
}

