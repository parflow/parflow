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
* Pointwise red/black Gauss-Seidel
*
*****************************************************************************/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int max_iter;
  int symmetric;
} PublicXtra;

typedef struct {
  /* InitInstanceXtra arguments */
  Grid     *grid;
  Matrix   *A;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * RedBlackGSPoint:
 *--------------------------------------------------------------------------*/

void     RedBlackGSPoint(
                         Vector *x,
                         Vector *b,
                         double  tol,
                         int     zero)
{
  PUSH_NVTX("RedBlackGSPoint",5)

  PFModule       *this_module = ThisPFModule;
  PublicXtra     *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra   *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int max_iter = (public_xtra->max_iter);
  int symmetric = (public_xtra->symmetric);

  Matrix    *A = (instance_xtra->A);

  SubregionArray *subregion_array;
  Subregion      *subregion;

  ComputePkg     *compute_pkg;
  Region         *compute_reg = NULL;

  Submatrix      *A_sub = NULL;
  Subvector      *x_sub = NULL;
  Subvector      *b_sub = NULL;

  StencilElt     *s;

  double         *a0, *a1, *a2, *a3, *a4, *a5, *a6;
  double         *x0, *x1, *x2, *x3, *x4, *x5, *x6;
  double         *bp;

  int ix, iy, iz;
  int nx, ny, nz;

  int nx_m = 0, ny_m = 0, nz_m = 0;
  int nx_v = 0, ny_v = 0, nz_v = 0;

  int sx, sy, sz;

  int compute_i, i_sa, i_s, i, j, k;
  int im, iv;

  int count;
  int rb = 0;
  int iter = 0;

  VectorUpdateCommHandle     *handle = NULL;

  int vector_update_mode;

  (void)tol;

  /*-----------------------------------------------------------------------
   * Begin timing
   *-----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   * Start red/black Gauss-Seidel
   *-----------------------------------------------------------------------*/

  if (symmetric)
    count = 1;
  else
    count = 2;

  /*-----------------------------------------------------------------------
   * If (zero) optimize iteration
   *   RDF don't need dependent and independent regions here
   *-----------------------------------------------------------------------*/

  if (zero)
  {
    count++;

    /* if max_iter = 0, set x to zero, and return if not symmetric */
    if ((count / 2) > max_iter)
    {
      InitVector(x, 0.0);
      if (!symmetric)
        return;
    }

    switch (rb)
    {
      case 0:
        /* red sweep */
        vector_update_mode = VectorUpdateRPoint;
        break;

      case 1:
        /* black sweep */
        vector_update_mode = VectorUpdateBPoint;
        break;
    }

    compute_pkg = GridComputePkg(VectorGrid(x), vector_update_mode);

    for (compute_i = 0; compute_i < 2; compute_i++)
    {
      switch (compute_i)
      {
        case 0:
          compute_reg = ComputePkgIndRegion(compute_pkg);
          break;

        case 1:
          compute_reg = ComputePkgDepRegion(compute_pkg);
          break;
      }

      ForSubregionArrayI(i_sa, compute_reg)
      {
        subregion_array = RegionSubregionArray(compute_reg, i_sa);

        if (SubregionArraySize(subregion_array))
        {
          x_sub = VectorSubvector(x, i_sa);
          b_sub = VectorSubvector(b, i_sa);

          A_sub = MatrixSubmatrix(A, i_sa);

          nx_m = SubmatrixNX(A_sub);
          ny_m = SubmatrixNY(A_sub);
          nz_m = SubmatrixNZ(A_sub);

          nx_v = SubvectorNX(x_sub);
          ny_v = SubvectorNY(x_sub);
          nz_v = SubvectorNZ(x_sub);
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

          sx = SubregionSX(subregion);
          sy = SubregionSY(subregion);
          sz = SubregionSZ(subregion);

          a0 = SubmatrixElt(A_sub, 0, ix, iy, iz);
          x0 = SubvectorElt(x_sub, ix, iy, iz);
          bp = SubvectorElt(b_sub, ix, iy, iz);

          iv = im = 0;

          BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                    iv, nx_v, ny_v, nz_v, sx, sy, sz,
                    im, nx_m, ny_m, nz_m, sx, sy, sz,
          {
            x0[iv] = bp[iv] / a0[im];

            SKIP_PARALLEL_SYNC;
          });
        }
      }
    }

    rb = !rb;

    IncFLOPCount(-12 * (VectorSize(x) / 2));
  }

  /*-----------------------------------------------------------------------
   * Do regular iterations
   *-----------------------------------------------------------------------*/

  for (; (count / 2) <= max_iter; count++)
  {
    iter = count / 2;

    switch (rb)
    {
      case 0:
        /* red sweep */
        vector_update_mode = VectorUpdateRPoint;
        break;

      case 1:
        /* black sweep */
        vector_update_mode = VectorUpdateBPoint;
        break;
    }

    compute_pkg = GridComputePkg(VectorGrid(x), vector_update_mode);

    for (compute_i = 0; compute_i < 2; compute_i++)
    {
      switch (compute_i)
      {
        case 0:
          PARALLEL_SYNC;
          handle = InitVectorUpdate(x, vector_update_mode);
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
          x_sub = VectorSubvector(x, i_sa);
          b_sub = VectorSubvector(b, i_sa);

          A_sub = MatrixSubmatrix(A, i_sa);

          nx_m = SubmatrixNX(A_sub);
          ny_m = SubmatrixNY(A_sub);
          nz_m = SubmatrixNZ(A_sub);

          nx_v = SubvectorNX(x_sub);
          ny_v = SubvectorNY(x_sub);
          nz_v = SubvectorNZ(x_sub);
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

          sx = SubregionSX(subregion);
          sy = SubregionSY(subregion);
          sz = SubregionSZ(subregion);

          s = StencilShape(MatrixStencil(A));

          a0 = SubmatrixElt(A_sub, 0, ix, iy, iz);
          a1 = SubmatrixElt(A_sub, 1, ix, iy, iz);
          a2 = SubmatrixElt(A_sub, 2, ix, iy, iz);
          a3 = SubmatrixElt(A_sub, 3, ix, iy, iz);
          a4 = SubmatrixElt(A_sub, 4, ix, iy, iz);
          a5 = SubmatrixElt(A_sub, 5, ix, iy, iz);
          a6 = SubmatrixElt(A_sub, 6, ix, iy, iz);

          x0 = SubvectorElt(x_sub, ix, iy, iz);
          x1 = SubvectorElt(x_sub,
                            (ix + s[1][0]),
                            (iy + s[1][1]),
                            (iz + s[1][2]));
          x2 = SubvectorElt(x_sub,
                            (ix + s[2][0]),
                            (iy + s[2][1]),
                            (iz + s[2][2]));
          x3 = SubvectorElt(x_sub,
                            (ix + s[3][0]),
                            (iy + s[3][1]),
                            (iz + s[3][2]));
          x4 = SubvectorElt(x_sub,
                            (ix + s[4][0]),
                            (iy + s[4][1]),
                            (iz + s[4][2]));
          x5 = SubvectorElt(x_sub,
                            (ix + s[5][0]),
                            (iy + s[5][1]),
                            (iz + s[5][2]));
          x6 = SubvectorElt(x_sub,
                            (ix + s[6][0]),
                            (iy + s[6][1]),
                            (iz + s[6][2]));

          bp = SubvectorElt(b_sub, ix, iy, iz);

          iv = im = 0;
          
          BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                    iv, nx_v, ny_v, nz_v, sx, sy, sz,
                    im, nx_m, ny_m, nz_m, sx, sy, sz,
          {
            x0[iv] = (bp[iv] - (a1[im] * x1[iv] +
                                a2[im] * x2[iv] +
                                a3[im] * x3[iv] +
                                a4[im] * x4[iv] +
                                a5[im] * x5[iv] +
                                a6[im] * x6[iv])) / a0[im];
                                
            SKIP_PARALLEL_SYNC;                                
          });
        }
      }
    }

    rb = !rb;
  }

  /*-----------------------------------------------------------------------
   * end timing
   *-----------------------------------------------------------------------*/

  if (symmetric)
    IncFLOPCount(13 * (iter * VectorSize(x) + (VectorSize(x) / 2)));
  else
    IncFLOPCount(13 * (iter * VectorSize(x)));

    PARALLEL_SYNC;

    POP_NVTX
}


/*--------------------------------------------------------------------------
 * RedBlackGSPointInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule     *RedBlackGSPointInitInstanceXtra(
                                              Problem *    problem,
                                              Grid *       grid,
                                              ProblemData *problem_data,
                                              Matrix *     A,
                                              double *     temp_data)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  (void)problem;
  (void)grid;
  (void)problem_data;
  (void)temp_data;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `A'
   *-----------------------------------------------------------------------*/

  if (A != NULL)
    (instance_xtra->A) = A;

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * RedBlackGSPointFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void   RedBlackGSPointFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * RedBlackGSPointNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *RedBlackGSPointNewPublicXtra(char *name)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  int symmetric;

  char          *switch_name;
  char key[IDB_MAX_KEY_LEN];
  NameArray switch_na;

  switch_na = NA_NewNameArray("True False");

  public_xtra = ctalloc(PublicXtra, 1);

  sprintf(key, "%s.MaxIter", name);
  public_xtra->max_iter = GetIntDefault(key, 1);

  sprintf(key, "%s.Symmetric", name);
  switch_name = GetStringDefault(key, "True");

  symmetric = NA_NameToIndexExitOnError(switch_na, switch_name, key);

  switch (symmetric)
  {
    /* True */
    case 0:
    {
      public_xtra->symmetric = 1;
      break;
    }

    /* False */
    case 1:
    {
      public_xtra->symmetric = 0;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }

  NA_FreeNameArray(switch_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * RedBlackGSPointFreePublicXtra
 *--------------------------------------------------------------------------*/

void   RedBlackGSPointFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (public_xtra)
  {
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * RedBlackGSPointSizeOfTempData
 *--------------------------------------------------------------------------*/

int  RedBlackGSPointSizeOfTempData()
{
  return 0;
}
