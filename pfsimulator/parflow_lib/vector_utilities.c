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
/* This file contains utility routines for ParFlow's Vector class.
 *
 * This file was modified from a coresponding PVODE package file to account
 * for the ParFlow Vector class and AMPS message-passing system.
 *
 * Routines included:
 *
 * PFVLinearSum(a, x, b, y, z)       z = a * x + b * y
 * PFVConstInit(c, z)                z = c
 * PFVProd(x, y, z)                  z_i = x_i * y_i
 * PFVDiv(x, y, z)                   z_i = x_i / y_i
 * PFVScale(c, x, z)                 z = c * x
 * PFVAbs(x, z)                      z_i = |x_i|
 * PFVInv(x, z)                      z_i = 1 / x_i
 * PFVAddConst(x, b, z)              z_i = x_i + b
 * PFVDotProd(x, y)                  Returns x dot y
 * PFVMaxNorm(x)                     Returns ||x||_{max}
 * PFVWrmsNorm(x, w)                 Returns sqrt((sum_i (x_i + w_i)^2)/length)
 * PFVWL2Norm(x, w)                  Returns sqrt(sum_i (x_i * w_i)^2)
 * PFVL1Norm(x)                      Returns sum_i |x_i|
 * PFVMin(x)                         Returns min_i x_i
 * PFVMax(x)                         Returns max_i x_i
 * PFVConstrProdPos(c, x)            Returns FALSE if some c_i = 0 &
 *                                      c_i*x_i <= 0.0
 * PFVCompare(c, x, z)               z_i = (x_i > c)
 * PFVInvTest(x, z)                  Returns (x_i != 0 forall i), z_i = 1 / x_i
 * PFVCopy(x, y)                     y = x
 * PFVSum(x, y, z)                   z = x + y
 * PFVDiff(x, y, z)                  z = x - y
 * PFVNeg(x, z)                      z = - x
 * PFVScaleSum(c, x, y, z)           z = c * (x + y)
 * PFVScaleDiff(c, x, y, z)          z = c * (x - y)
 * PFVLin1(a, x, y, z)               z = a * x + y
 * PFVLin2(a, x, y, z)               z = a * x - y
 * PFVAxpy(a, x, y)                  y = y + a * x
 * PFVScaleBy(a, x)                  x = x * a
 *
 * PFVLayerCopy (a, b, x, y)         NBE: Extracts layer b from vector y, inserts into layer a of vector x
 ****************************************************************************/

#include "parflow.h"

#include <string.h>

#define ZERO 0.0
#define ONE  1.0

/* Kinsol API is in C */
#ifdef __cplusplus
extern "C"
#endif

void PFVLinearSum(
/* LinearSum : z = a * x + b * y              */
                  double  a,
                  Vector *x,
                  double  b,
                  Vector *y,
                  Vector *z)

{
  double c;
  Vector *v1, *v2;
  int test;

  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *y_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  const double * __restrict__ yp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_y, ny_y, nz_y;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_y, i_z;

  if ((b == ONE) && (z == y))      /* BLAS usage: axpy y <- ax+y */
  {
    PFVAxpy(a, x, y);
    return;
  }

  if ((a == ONE) && (z == x))      /* BLAS usage: axpy x <- by+x */
  {
    PFVAxpy(b, y, x);
    return;
  }

  /* Case: a == b == 1.0 */

  if ((a == ONE) && (b == ONE))
  {
    PFVSum(x, y, z);
    return;
  }

  /* Cases: (1) a == 1.0, b = -1.0, (2) a == -1.0, b == 1.0 */

  if ((test = ((a == ONE) && (b == -ONE))) || ((a == -ONE) && (b == ONE)))
  {
    v1 = test ? y : x;
    v2 = test ? x : y;
    PFVDiff(v2, v1, z);
    return;
  }

  /* Cases: (1) a == 1.0, b == other or 0.0, (2) a == other or 0.0, b == 1.0 */
  /* if a or b is 0.0, then user should have called N_VScale */

  if ((test = (a == ONE)) || (b == ONE))
  {
    c = test ? b : a;
    v1 = test ? y : x;
    v2 = test ? x : y;
    PFVLin1(c, v1, v2, z);
    return;
  }

  /* Cases: (1) a == -1.0, b != 1.0, (2) a != 1.0, b == -1.0 */

  if ((test = (a == -ONE)) || (b == -ONE))
  {
    c = test ? b : a;
    v1 = test ? y : x;
    v2 = test ? x : y;
    PFVLin2(c, v1, v2, z);
    return;
  }

  /* Case: a == b */
  /* catches case both a and b are 0.0 - user should have called N_VConst */

  if (a == b)
  {
    PFVScaleSum(a, x, y, z);
    return;
  }

  /* Case: a == -b */

  if (a == -b)
  {
    PFVScaleDiff(a, x, y, z);
    return;
  }

  /* Do all cases not handled above:
   * (1) a == other, b == 0.0 - user should have called N_VScale
   * (2) a == 0.0, b == other - user should have called N_VScale
   * (3) a,b == other, a !=b, a != -b */

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    z_sub = VectorSubvector(z, sg);
    x_sub = VectorSubvector(x, sg);
    y_sub = VectorSubvector(y, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_y = SubvectorNX(y_sub);
    ny_y = SubvectorNY(y_sub);
    nz_y = SubvectorNZ(y_sub);

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    zp = SubvectorElt(z_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);
    yp = SubvectorElt(y_sub, ix, iy, iz);

    i_x = 0;
    i_y = 0;
    i_z = 0;
    BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_y, nx_y, ny_y, nz_y, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = a * xp[i_x] + b * yp[i_y];
    });
  }
  IncFLOPCount(3 * VectorSize(z));
}

void PFVConstInit(
/* ConstInit : z = c   */
                  double  c,
                  Vector *z)
{
  Grid       *grid = VectorGrid(z);
  Subgrid    *subgrid;

  Subvector  *z_sub;

  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_z;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    z_sub = VectorSubvector(z, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    zp = SubvectorElt(z_sub, ix, iy, iz);

    i_z = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = c;
    });
  }
}

void PFVProd(
/* Prod : z_i = x_i * y_i   */
             Vector *x,
             Vector *y,
             Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *y_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  const double * __restrict__ yp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_y, ny_y, nz_y;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_y, i_z;

  grid = VectorGrid(x);
  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    z_sub = VectorSubvector(z, sg);
    x_sub = VectorSubvector(x, sg);
    y_sub = VectorSubvector(y, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_y = SubvectorNX(y_sub);
    ny_y = SubvectorNY(y_sub);
    nz_y = SubvectorNZ(y_sub);

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    zp = SubvectorElt(z_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);
    yp = SubvectorElt(y_sub, ix, iy, iz);

    i_x = 0;
    i_y = 0;
    i_z = 0;
    BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_y, nx_y, ny_y, nz_y, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = xp[i_x] * yp[i_y];
    });
  }
  IncFLOPCount(VectorSize(x));
}

void PFVDiv(
/* Div : z_i = x_i / y_i   */
            Vector *x,
            Vector *y,
            Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *y_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  const double * __restrict__ yp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_y, ny_y, nz_y;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_y, i_z;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    z_sub = VectorSubvector(z, sg);
    x_sub = VectorSubvector(x, sg);
    y_sub = VectorSubvector(y, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_y = SubvectorNX(y_sub);
    ny_y = SubvectorNY(y_sub);
    nz_y = SubvectorNZ(y_sub);

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    zp = SubvectorElt(z_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);
    yp = SubvectorElt(y_sub, ix, iy, iz);

    i_x = 0;
    i_y = 0;
    i_z = 0;
    BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_y, nx_y, ny_y, nz_y, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = xp[i_x] / yp[i_y];
    });
  }
  IncFLOPCount(VectorSize(x));
}

void PFVScale(
/* Scale : z = c * x   */
              double  c,
              Vector *x,
              Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_z;

  if (z == x)
  {       /* BLAS usage: scale x <- cx */
    PFVScaleBy(c, x);
    return;
  }

  if (c == ONE)
  {
    PFVCopy(x, z);
  }
  else if (c == -ONE)
  {
    PFVNeg(x, z);
  }
  else
  {
    ForSubgridI(sg, GridSubgrids(grid))
    {
      subgrid = GridSubgrid(grid, sg);

      z_sub = VectorSubvector(z, sg);
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

      nx_z = SubvectorNX(z_sub);
      ny_z = SubvectorNY(z_sub);
      nz_z = SubvectorNZ(z_sub);

      zp = SubvectorElt(z_sub, ix, iy, iz);
      xp = SubvectorElt(x_sub, ix, iy, iz);

      i_x = 0;
      i_z = 0;
      BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                i_x, nx_x, ny_x, nz_x, 1, 1, 1,
                i_z, nx_z, ny_z, nz_z, 1, 1, 1,
      {
        zp[i_z] = c * xp[i_x];
      });
    }
  }
  IncFLOPCount(VectorSize(x));
}

void PFVAbs(
/* Abs : z_i = |x_i|   */
            Vector *x,
            Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_z;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    z_sub = VectorSubvector(z, sg);
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

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    zp = SubvectorElt(z_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);

    i_x = 0;
    i_z = 0;
    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = fabs(xp[i_x]);
    });
  }
}

void PFVInv(
/* Inv : z_i = 1 / x_i    */
            Vector *x,
            Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_z;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    z_sub = VectorSubvector(z, sg);
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

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    zp = SubvectorElt(z_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);

    i_x = 0;
    i_z = 0;
    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = ONE / xp[i_x];
    });
  }
  IncFLOPCount(VectorSize(x));
}

void PFVAddConst(
/* AddConst : z_i = x_i + b  */
                 Vector *x,
                 double  b,
                 Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_z;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    z_sub = VectorSubvector(z, sg);
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

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    zp = SubvectorElt(z_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);

    i_x = 0;
    i_z = 0;
    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = xp[i_x] + b;
    });
  }
  IncFLOPCount(VectorSize(x));
}

double PFVDotProd(
/* DotProd = x dot y   */
                  Vector *x,
                  Vector *y)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *y_sub;

  const double * __restrict__ yp;
  const double * __restrict__ xp;
  double sum = ZERO;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_y, ny_y, nz_y;

  int sg, i, j, k, i_x, i_y;

  amps_Invoice result_invoice;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    x_sub = VectorSubvector(x, sg);
    y_sub = VectorSubvector(y, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_y = SubvectorNX(y_sub);
    ny_y = SubvectorNY(y_sub);
    nz_y = SubvectorNZ(y_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    yp = SubvectorElt(y_sub, ix, iy, iz);

    i_x = 0;
    i_y = 0;

    BoxLoopReduceI2(sum,
                    i, j, k, ix, iy, iz, nx, ny, nz,
                    i_x, nx_x, ny_x, nz_x, 1, 1, 1,
                    i_y, nx_y, ny_y, nz_y, 1, 1, 1,
    {
      ReduceSum(sum, xp[i_x] * yp[i_y]);
    });
  }

  result_invoice = amps_NewInvoice("%d", &sum);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
  amps_FreeInvoice(result_invoice);

  IncFLOPCount(2 * VectorSize(x));

  return(sum);
}

double PFVMaxNorm(
/* MaxNorm = || x ||_{max}   */
                  Vector *x)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;

  const double * __restrict__ xp;
  double max_val = ZERO;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;

  int sg, i, j, k, i_x;

  amps_Invoice result_invoice;

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
    BoxLoopReduceI1(max_val,
                    i, j, k, ix, iy, iz, nx, ny, nz,
                    i_x, nx_x, ny_x, nz_x, 1, 1, 1,
    {
      double xp_abs = fabs(xp[i_x]);
      ReduceMax(max_val, xp_abs);
    });
  }

  result_invoice = amps_NewInvoice("%d", &max_val);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Max);
  amps_FreeInvoice(result_invoice);

  return(max_val);
}

double PFVWrmsNorm(
/* WrmsNorm = sqrt((sum_i (x_i * w_i)^2)/length)  */
                   Vector *x,
                   Vector *w)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *w_sub;

  const double * __restrict__ wp;
  const double * __restrict__ xp;
  double sum = ZERO;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_w, ny_w, nz_w;

  int sg, i, j, k, i_x, i_w;

  amps_Invoice result_invoice;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    x_sub = VectorSubvector(x, sg);
    w_sub = VectorSubvector(w, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_w = SubvectorNX(w_sub);
    ny_w = SubvectorNY(w_sub);
    nz_w = SubvectorNZ(w_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    wp = SubvectorElt(w_sub, ix, iy, iz);

    i_x = 0;
    i_w = 0;

    BoxLoopReduceI2(sum,
                    i, j, k, ix, iy, iz, nx, ny, nz,
                    i_x, nx_x, ny_x, nz_x, 1, 1, 1,
                    i_w, nx_w, ny_w, nz_w, 1, 1, 1,
    {
      double prod = xp[i_x] * wp[i_w];
      ReduceSum(sum, prod * prod);
    });
  }

  result_invoice = amps_NewInvoice("%d", &sum);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
  amps_FreeInvoice(result_invoice);

  IncFLOPCount(3 * VectorSize(x));

  return(sqrt(sum / (x->size)));
}

double PFVWL2Norm(
/* WL2Norm = sqrt(sum_i (x_i * w_i)^2)  */
                  Vector *x,
                  Vector *w)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *w_sub;

  const double * __restrict__ wp;
  const double * __restrict__ xp;
  double sum = ZERO;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_w, ny_w, nz_w;

  int sg, i, j, k, i_x, i_w;

  amps_Invoice result_invoice;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    x_sub = VectorSubvector(x, sg);
    w_sub = VectorSubvector(w, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_w = SubvectorNX(w_sub);
    ny_w = SubvectorNY(w_sub);
    nz_w = SubvectorNZ(w_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    wp = SubvectorElt(w_sub, ix, iy, iz);

    i_x = 0;
    i_w = 0;

    BoxLoopReduceI2(sum,
                    i, j, k, ix, iy, iz, nx, ny, nz,
                    i_x, nx_x, ny_x, nz_x, 1, 1, 1,
                    i_w, nx_w, ny_w, nz_w, 1, 1, 1,
    {
      const double prod = xp[i_x] * wp[i_w];
      ReduceSum(sum, prod * prod);
    });
  }

  result_invoice = amps_NewInvoice("%d", &sum);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
  amps_FreeInvoice(result_invoice);

  IncFLOPCount(3 * VectorSize(x));

  return(sqrt(sum));
}

double PFVL1Norm(
/* L1Norm = sum_i |x_i|  */
                 Vector *x)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;

  const double * __restrict__ xp;
  double sum = ZERO;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;

  int sg, i, j, k, i_x;

  amps_Invoice result_invoice;

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
    BoxLoopReduceI1(sum,
                    i, j, k, ix, iy, iz, nx, ny, nz,
                    i_x, nx_x, ny_x, nz_x, 1, 1, 1,
    {
      ReduceSum(sum, fabs(xp[i_x]));
    });
  }

  result_invoice = amps_NewInvoice("%d", &sum);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
  amps_FreeInvoice(result_invoice);

  return(sum);
}

double PFVMin(
/* Min = min_i(x_i)   */
              Vector *x)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;

  const double * __restrict__ xp;
  double min_val = ZERO;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;

  int sg, i, j, k, i_x;

  amps_Invoice result_invoice;

  result_invoice = amps_NewInvoice("%d", &min_val);

  grid = VectorGrid(x);

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

    /* Get initial guess for min_val */
    if (sg == 0)
    {
      i_x = 0;
      BoxLoopReduceI1(min_val,
                      i, j, k, ix, iy, iz, 1, 1, 1,
                      i_x, nx_x, ny_x, nz_x, 1, 1, 1,
      {
        ReduceSum(min_val, xp[i_x]);
      });
    }

    i_x = 0;
    BoxLoopReduceI1(min_val,
                    i, j, k, ix, iy, iz, nx, ny, nz,
                    i_x, nx_x, ny_x, nz_x, 1, 1, 1,
    {
      ReduceMin(min_val, xp[i_x]);
    });
  }

  amps_AllReduce(amps_CommWorld, result_invoice, amps_Min);
  amps_FreeInvoice(result_invoice);

  return(min_val);
}

double PFVMax(
/* Max = max_i(x_i)   */
              Vector *x)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;

  const double * __restrict__ xp;
  double max_val = ZERO;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;

  int sg, i, j, k, i_x;

  amps_Invoice result_invoice;

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

    /* Get initial guess for max_val */
    if (sg == 0)
    {
      i_x = 0;
      BoxLoopReduceI1(max_val,
                      i, j, k, ix, iy, iz, 1, 1, 1,
                      i_x, nx_x, ny_x, nz_x, 1, 1, 1,
      {
        ReduceSum(max_val, xp[i_x]);
      });
    }

    i_x = 0;

    BoxLoopReduceI1(max_val,
                    i, j, k, ix, iy, iz, nx, ny, nz,
                    i_x, nx_x, ny_x, nz_x, 1, 1, 1,
    {
      ReduceMax(max_val, xp[i_x]);
    });
  }

  result_invoice = amps_NewInvoice("%d", &max_val);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Max);
  amps_FreeInvoice(result_invoice);

  return(max_val);
}

int PFVConstrProdPos(
/* ConstrProdPos: Returns a boolean FALSE if some c[i]!=0.0  */
/*                and x[i]*c[i]<=0.0 */
                     Vector *c,
                     Vector *x)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *c_sub;
  Subvector  *x_sub;

  const double * __restrict__ cp;
  const double * __restrict__ xp;

  int *val = talloc(int, 1);

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_c, ny_c, nz_c;

  int sg, i, j, k, i_x, i_c;

  amps_Invoice result_invoice;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    x_sub = VectorSubvector(x, sg);
    c_sub = VectorSubvector(c, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_c = SubvectorNX(c_sub);
    ny_c = SubvectorNY(c_sub);
    nz_c = SubvectorNZ(c_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    cp = SubvectorElt(c_sub, ix, iy, iz);

    *val = 1;
    i_c = 0;
    i_x = 0;
    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_c, nx_c, ny_c, nz_c, 1, 1, 1,
    {
      if (cp[i_c] != ZERO)
      {
        if ((xp[i_x] * cp[i_c]) <= ZERO)
          *val = 0;
      }
    });
  }

  result_invoice = amps_NewInvoice("%i", val);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Min);
  amps_FreeInvoice(result_invoice);

  if (*val == 0)
  {
    tfree(val);
    return(FALSE);
  }
  else
  {
    tfree(val);
    return(TRUE);
  }
}

void PFVCompare(
/* Compare : z_i = (x_i > c)  */
                double  c,
                Vector *x,
                Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_z;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    z_sub = VectorSubvector(z, sg);
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

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    zp = SubvectorElt(z_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);

    i_x = 0;
    i_z = 0;
    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = (fabs(xp[i_x]) >= c) ? ONE : ZERO;
    });
  }
}


int PFVInvTest(
/* InvTest = (x_i != 0 forall i), z_i = 1 / x_i  */
               Vector *x,
               Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  double * __restrict__ zp;

  int *val = talloc(int, 1);

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_z;

  amps_Invoice result_invoice;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    z_sub = VectorSubvector(z, sg);
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

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    zp = SubvectorElt(z_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);

    i_x = 0;
    i_z = 0;
    *val = 1;
    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      if (xp[i_x] == ZERO)
        *val = 0;
      else
        zp[i_z] = ONE / (xp[i_x]);
    });
  }

  result_invoice = amps_NewInvoice("%i", val);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Min);
  amps_FreeInvoice(result_invoice);

  if (*val == 0)
  {
    tfree(val);
    return(FALSE);
  }
  else
  {
    tfree(val);
    return(TRUE);
  }
}


/***************** Private Helper Functions **********************/

/**
 * Copies elements from vector X to vector Y.
 *
 * Includes ghost layer.
 *
 * Assumes X is same size as Y.
 *
 * @param x source vector
 * @param Y destination vector
 */
void PFVCopy(Vector *x,
             Vector *y)
{
  Grid *grid = VectorGrid(x);
  int sg;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    Subvector  *x_sub = VectorSubvector(x, sg);
    Subvector  *y_sub = VectorSubvector(y, sg);

    tmemcpy(SubvectorData(y_sub), SubvectorData(x_sub), SubvectorDataSize(y_sub) * sizeof(double));
  }
}

void PFVSum(
/* Sum : z = x + y   */
            Vector *x,
            Vector *y,
            Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *y_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  const double * __restrict__ yp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_y, ny_y, nz_y;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_y, i_z;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    x_sub = VectorSubvector(x, sg);
    y_sub = VectorSubvector(y, sg);
    z_sub = VectorSubvector(z, sg);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_y = SubvectorNX(y_sub);
    ny_y = SubvectorNY(y_sub);
    nz_y = SubvectorNZ(y_sub);

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    yp = SubvectorElt(y_sub, ix, iy, iz);
    zp = SubvectorElt(z_sub, ix, iy, iz);

    i_x = 0;
    i_y = 0;
    i_z = 0;
    BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_y, nx_y, ny_y, nz_y, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = xp[i_x] + yp[i_y];
    });
  }
  IncFLOPCount(VectorSize(x));
}

void PFVDiff(
/* Diff : z = x - y  */
             Vector *x,
             Vector *y,
             Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *y_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  const double * __restrict__ yp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_y, ny_y, nz_y;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_y, i_z;


  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    x_sub = VectorSubvector(x, sg);
    y_sub = VectorSubvector(y, sg);
    z_sub = VectorSubvector(z, sg);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_y = SubvectorNX(y_sub);
    ny_y = SubvectorNY(y_sub);
    nz_y = SubvectorNZ(y_sub);

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    yp = SubvectorElt(y_sub, ix, iy, iz);
    zp = SubvectorElt(z_sub, ix, iy, iz);

    i_x = 0;
    i_y = 0;
    i_z = 0;
    BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_y, nx_y, ny_y, nz_y, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = xp[i_x] - yp[i_y];
    });
  }
  IncFLOPCount(VectorSize(x));
}

void PFVNeg(
/* Neg : z = - x   */
            Vector *x,
            Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_z;


  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    x_sub = VectorSubvector(x, sg);
    z_sub = VectorSubvector(z, sg);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    zp = SubvectorElt(z_sub, ix, iy, iz);

    i_x = 0;
    i_z = 0;
    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = -xp[i_x];
    });
  }
}

void PFVScaleSum(
/* ScaleSum : z = c * x + y   */
                 double  c,
                 Vector *x,
                 Vector *y,
                 Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *y_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  const double * __restrict__ yp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_y, ny_y, nz_y;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_y, i_z;


  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    x_sub = VectorSubvector(x, sg);
    y_sub = VectorSubvector(y, sg);
    z_sub = VectorSubvector(z, sg);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_y = SubvectorNX(y_sub);
    ny_y = SubvectorNY(y_sub);
    nz_y = SubvectorNZ(y_sub);

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    yp = SubvectorElt(y_sub, ix, iy, iz);
    zp = SubvectorElt(z_sub, ix, iy, iz);

    i_x = 0;
    i_y = 0;
    i_z = 0;
    BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_y, nx_y, ny_y, nz_y, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = c * (xp[i_x] + yp[i_y]);
    });
  }
  IncFLOPCount(2 * VectorSize(x));
}

void PFVScaleDiff(
/* ScaleDiff : z = c * x - y   */
                  double  c,
                  Vector *x,
                  Vector *y,
                  Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *y_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  const double * __restrict__ yp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_y, ny_y, nz_y;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_y, i_z;


  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    x_sub = VectorSubvector(x, sg);
    y_sub = VectorSubvector(y, sg);
    z_sub = VectorSubvector(z, sg);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_y = SubvectorNX(y_sub);
    ny_y = SubvectorNY(y_sub);
    nz_y = SubvectorNZ(y_sub);

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    yp = SubvectorElt(y_sub, ix, iy, iz);
    zp = SubvectorElt(z_sub, ix, iy, iz);

    i_x = 0;
    i_y = 0;
    i_z = 0;
    BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_y, nx_y, ny_y, nz_y, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = c * (xp[i_x] - yp[i_y]);
    });
  }
  IncFLOPCount(2 * VectorSize(x));
}

void PFVLin1(
/* Lin1 : z = a * x + y   */
             double  a,
             Vector *x,
             Vector *y,
             Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *y_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  const double * __restrict__ yp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_y, ny_y, nz_y;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_y, i_z;


  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    x_sub = VectorSubvector(x, sg);
    y_sub = VectorSubvector(y, sg);
    z_sub = VectorSubvector(z, sg);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_y = SubvectorNX(y_sub);
    ny_y = SubvectorNY(y_sub);
    nz_y = SubvectorNZ(y_sub);

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    yp = SubvectorElt(y_sub, ix, iy, iz);
    zp = SubvectorElt(z_sub, ix, iy, iz);

    i_x = 0;
    i_y = 0;
    i_z = 0;
    BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_y, nx_y, ny_y, nz_y, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = a * (xp[i_x]) + yp[i_y];
    });
  }
  IncFLOPCount(2 * VectorSize(x));
}

void PFVLin2(
/* Lin2 : z = a * x - y   */
             double  a,
             Vector *x,
             Vector *y,
             Vector *z)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *y_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  const double * __restrict__ yp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_y, ny_y, nz_y;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_y, i_z;


  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    x_sub = VectorSubvector(x, sg);
    y_sub = VectorSubvector(y, sg);
    z_sub = VectorSubvector(z, sg);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_y = SubvectorNX(y_sub);
    ny_y = SubvectorNY(y_sub);
    nz_y = SubvectorNZ(y_sub);

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    yp = SubvectorElt(y_sub, ix, iy, iz);
    zp = SubvectorElt(z_sub, ix, iy, iz);

    i_x = 0;
    i_y = 0;
    i_z = 0;
    BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_y, nx_y, ny_y, nz_y, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = a * (xp[i_x]) - yp[i_y];
    });
  }
  IncFLOPCount(2 * VectorSize(x));
}

void PFVAxpy(
/* axpy : y = y + a * x   */
             double  a,
             Vector *x,
             Vector *y)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *y_sub;

  const double * __restrict__ xp;
  double * __restrict__ yp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_y, ny_y, nz_y;

  int sg, i, j, k, i_x, i_y;


  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    x_sub = VectorSubvector(x, sg);
    y_sub = VectorSubvector(y, sg);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_y = SubvectorNX(y_sub);
    ny_y = SubvectorNY(y_sub);
    nz_y = SubvectorNZ(y_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    yp = SubvectorElt(y_sub, ix, iy, iz);

    i_x = 0;
    i_y = 0;
    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_y, nx_y, ny_y, nz_y, 1, 1, 1,
    {
      yp[i_y] += a * (xp[i_x]);
    });
  }
  IncFLOPCount(2 * VectorSize(x));
}

void PFVScaleBy(
/* ScaleBy : x = x * a   */
                double  a,
                Vector *x)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;

  double * __restrict__ xp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;

  int sg, i, j, k, i_x;


  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    x_sub = VectorSubvector(x, sg);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);

    i_x = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
    {
      xp[i_x] = xp[i_x] * a;
    });
  }
  IncFLOPCount(VectorSize(x));
}

void PFVLayerCopy(
/* NBE: Extract layer b from y and insert into layer a of x   */
                  int     a,
                  int     b,
                  Vector *x,
                  Vector *y)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *y_sub;

  const double * __restrict__ yp;
  double * __restrict__ xp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;

  int sg, i, j, k, i_x, i_y;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    x_sub = VectorSubvector(x, sg);
    y_sub = VectorSubvector(y, sg);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    yp = SubvectorElt(y_sub, ix, iy, b);

    i_x = 0;
    i_y = 0;

    DeclareInc(jinc, kinc, nx, ny, nz, nx_x, ny_x, nz_x, 1, 1, 1);

    i_x = 0;
    i_y = 0;

    for (k = iz; k < iz + nz; k++)
    {
      if (k == a)
      {
        for (j = iy; j < iy + ny; j++)
        {
          for (i = ix; i < ix + nx; i++)
          {
            xp[i_x] = yp[i_y];
            i_x += 1;
            i_y += 1;
          }
          i_x += jinc;
          i_y += jinc;
        }
      }
      else
      {
        for (j = iy; j < iy + ny; j++)
        {
          for (i = ix; i < ix + nx; i++)
          {
            i_x += 1;
          }
          i_x += jinc;
        }
      }
      i_x += kinc;
    }
  }
  IncFLOPCount(2 * VectorSize(x));
}

#ifdef __cpluspus
}
#endif
