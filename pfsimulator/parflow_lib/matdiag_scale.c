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
* FILE: matdiag_scale.c
*
* WRITTEN BY:   Bill Bosl
*               phone: (510) 423-2873
*               e-mail: wjbosl@llnl.gov
*
* DESCRIPTION:
* Routine for computing the diagonal scaling vector D by taking
* the inverse of the diagonal elements of A. The subroutine
* DiagScale is then called to scale or unscale the matrix A.
* The scaling operation that is carried out is:
*
*               A~ = D^(-1/2) A D^(-1/2)
*
*               x~ = D^(1/2) x
*
*               b~ = D^(-1/2) b
*
* Unscaling is accomplished by inverting the vector
* D that was previously computed, then calling DiagScale.
*
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void PublicXtra;

typedef struct {
  /* InitInstanceXtra arguments */
  Grid      *grid;

  /* instance data */
  Vector    *d;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * MatDiagScale
 *--------------------------------------------------------------------------*/

void MatDiagScale(Vector *x, Matrix *A, Vector *b, int flag)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  Vector         *d = (instance_xtra->d);

  int si;

  Grid           *grid = MatrixGrid(A);
  Subgrid        *subgrid;

  Subvector      *d_sub;
  Submatrix      *A_sub;

  double         *cp;
  double         *dp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;
  int nx_m, ny_m, nz_m;

  int iv, im;
  int i, j, k;


  /*-----------------------------------------------------------------------
   * Compute the diagonal scaling vector d if flag = 1 (DO)
   *-----------------------------------------------------------------------*/

  if (flag)
  {
    for (si = 0; si < GridNumSubgrids(grid); si++)
    {
      subgrid = GridSubgrid(grid, si);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      d_sub = VectorSubvector(d, si);
      A_sub = MatrixSubmatrix(A, si);

      nx_v = SubvectorNX(d_sub);
      ny_v = SubvectorNY(d_sub);
      nz_v = SubvectorNZ(d_sub);

      nx_m = SubmatrixNX(A_sub);
      ny_m = SubmatrixNY(A_sub);
      nz_m = SubmatrixNZ(A_sub);

      dp = SubvectorElt(d_sub, ix, iy, iz);
      cp = SubmatrixElt(A_sub, 0, ix, iy, iz);

      iv = 0;
      im = 0;
      BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                iv, nx_v, ny_v, nz_v, 1, 1, 1,
                im, nx_m, ny_m, nz_m, 1, 1, 1,
      {
        dp[iv] = 1.0 / sqrt(cp[im]);
      });
    }
  }

  /*-----------------------------------------------------------------------
   * In this case, we're going to unscale, so we need to invert the
   * d matrix that was originally computed.
   *-----------------------------------------------------------------------*/

  else
  {
    for (si = 0; si < GridNumSubgrids(grid); si++)
    {
      subgrid = GridSubgrid(grid, si);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      d_sub = VectorSubvector(d, si);

      nx_v = SubvectorNX(d_sub);
      ny_v = SubvectorNY(d_sub);
      nz_v = SubvectorNZ(d_sub);

      dp = SubvectorElt(d_sub, ix, iy, iz);

      iv = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                iv, nx_v, ny_v, nz_v, 1, 1, 1,
      {
        dp[iv] = 1.0 / (dp[iv]);
      });
    }
  }

  /* Call the diagonal scaling function to do the work */
  DiagScale(x, A, b, d);
}

/*--------------------------------------------------------------------------
 * MatDiagScaleInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *MatDiagScaleInitInstanceXtra(
                                        Grid *grid)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;


  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
  {
    /* free old data */
    if ((instance_xtra->grid) != NULL)
    {
      FreeVector(instance_xtra->d);
    }

    /* set new data */
    (instance_xtra->grid) = grid;

    (instance_xtra->d) = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * MatDiagScaleFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  MatDiagScaleFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    FreeVector(instance_xtra->d);
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * MatDiagScaleNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *MatDiagScaleNewPublicXtra(char *name)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  (void)name;

  public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * MatDiagScaleFreePublicXtra
 *--------------------------------------------------------------------------*/

void  MatDiagScaleFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * MatDiagScaleSizeOfTempData
 *--------------------------------------------------------------------------*/

int  MatDiagScaleSizeOfTempData()
{
  return 0;
}
