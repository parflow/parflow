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
*
* FILE: diag_scale.c
*
* FUNCTIONS IN THIS FILE:
* DiagScale
*
* DESCRIPTION:
* Routine for diagonally scaling and unscaling a matrix A.
* The diagonal scaling matrix D is passed into the routine
* as a vector. The following transformations are then applied
* to the matrix A and the vectors x and b:
*
*               A~ = D*A*D
*               x~ = D(-1)*x
*               b~ = D*b
*
* are computed.
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * DiagScale
 *--------------------------------------------------------------------------*/

void     DiagScale(
                   Vector *x,
                   Matrix *A,
                   Vector *b,
                   Vector *d)
{
  Grid           *grid = MatrixGrid(A);
  Subgrid        *subgrid;

  VectorUpdateCommHandle *handle = NULL;

  ComputePkg     *compute_pkg;
  Region         *compute_reg = NULL;
  SubregionArray *subregion_array;
  Subregion      *subregion;

  Submatrix      *A_sub;
  Subvector      *x_sub;
  Subvector      *b_sub;
  Subvector      *d_sub;

  double         *cp, *ep, *np, *up;
  double         *xp, *bp, *dp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_m, ny_m, nz_m;
  int nx_v, ny_v, nz_v;

  int s_y, s_z;

  int compute_i, i_sa, i_s, i, j, k, iv = 0, im;


  /*-----------------------------------------------------------------------
   * Scale x and b
   *-----------------------------------------------------------------------*/

  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    x_sub = VectorSubvector(x, i_s);
    b_sub = VectorSubvector(b, i_s);
    d_sub = VectorSubvector(d, i_s);

    nx_v = SubvectorNX(x_sub);
    ny_v = SubvectorNY(x_sub);
    nz_v = SubvectorNZ(x_sub);

    xp = SubvectorElt(x_sub, ix, iy, iz);
    bp = SubvectorElt(b_sub, ix, iy, iz);
    dp = SubvectorElt(d_sub, ix, iy, iz);

    iv = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      xp[iv] = xp[iv] / dp[iv];
      bp[iv] = bp[iv] * dp[iv];
    });
  }

  /*-----------------------------------------------------------------------
   * Scale A
   *-----------------------------------------------------------------------*/

  compute_pkg = GridComputePkg(VectorGrid(x), VectorUpdateAll);

  for (compute_i = 0; compute_i < 2; compute_i++)
  {
    switch (compute_i)
    {
      case 0:
        handle = InitVectorUpdate(d, VectorUpdateAll);
        compute_reg = ComputePkgIndRegion(compute_pkg);
        break;

      case 1:
        FinalizeVectorUpdate(handle);
        compute_reg = ComputePkgDepRegion(compute_pkg);
        break;
    }

    ForSubregionArrayI(i_sa, compute_reg)
    {
      subregion_array = RegionSubregionArray(compute_reg, i_sa);

      if (SubregionArraySize(subregion_array))
      {
        A_sub = MatrixSubmatrix(A, i_sa);
        d_sub = VectorSubvector(d, i_sa);

        nx_m = SubmatrixNX(A_sub);
        ny_m = SubmatrixNY(A_sub);
        nz_m = SubmatrixNZ(A_sub);

        nx_v = SubvectorNX(d_sub);
        ny_v = SubvectorNY(d_sub);
        nz_v = SubvectorNZ(d_sub);
      }

      ForSubregionI(i_s, subregion_array)
      {
        subregion = SubregionArraySubregion(subregion_array, i_s);

        ix = SubregionIX(subregion);
        iy = SubregionIY(subregion);
        iz = SubregionIZ(subregion);

        nx = SubregionNX(subregion);
        ny = SubregionNY(subregion);
        nz = SubregionNZ(subregion);

        s_y = nx_v;
        s_z = nx_v * ny_v;

        switch (StencilSize(MatrixStencil(A)))
        {
          case 1:
            cp = SubmatrixElt(A_sub, 0, ix, iy, iz);

            dp = SubvectorElt(d_sub, ix, iy, iz);

            im = 0;
            BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                      im, nx_m, ny_m, nz_m, 1, 1, 1,
          {
            cp[im] *= dp[iv] * dp[iv];
          });
            break;

          case 7:
            cp = SubmatrixElt(A_sub, 0, ix, iy, iz);
            ep = SubmatrixElt(A_sub, 2, ix, iy, iz);
            np = SubmatrixElt(A_sub, 4, ix, iy, iz);
            up = SubmatrixElt(A_sub, 6, ix, iy, iz);

            dp = SubvectorElt(d_sub, ix, iy, iz);

            im = 0;
            iv = 0;
            BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                      im, nx_m, ny_m, nz_m, 1, 1, 1,
                      iv, nx_v, ny_v, nz_v, 1, 1, 1,
          {
            cp[im] *= dp[iv] * dp[iv];
            ep[im] *= dp[iv + 1] * dp[iv];
            np[im] *= dp[iv + s_y] * dp[iv];
            up[im] *= dp[iv + s_z] * dp[iv];
          });
            break;

          default:
            break;
        }
      }
    }
  }

  /*-----------------------------------------------------------------------
   * Update matrix ghost points
   *-----------------------------------------------------------------------*/

  if (MatrixCommPkg(A))
  {
    CommHandle *matrix_handle = InitMatrixUpdate(A);
    FinalizeMatrixUpdate(matrix_handle);
  }
}

