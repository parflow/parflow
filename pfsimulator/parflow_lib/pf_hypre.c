/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2026, Lawrence Livermore National Security,
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

#ifdef HAVE_HYPRE
#include "hypre_dependences.h"

/*
 * Versions of Hypre > 2.10.x require a dimension argument for BoxCreate.
 */
#if PARFLOW_HYPRE_VERSION_MAJOR > 2 || \
  (PARFLOW_HYPRE_VERSION_MAJOR >= 2 && PARFLOW_HYPRE_VERSION_MINOR >= 10)
#define PARFLOW_HYPRE_DIM 3
#else
#define PARFLOW_HYPRE_DIM
#endif

void CopyParFlowVectorToHypreVector(Vector *            rhs,
                                    HYPRE_StructVector* hypre_b)
{
  Grid* grid = VectorGrid(rhs);
  int sg;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    Subgrid* subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);
    Subvector* rhs_sub = VectorSubvector(rhs, sg);

    double* rhs_ptr = SubvectorData(rhs_sub);

    int ix = SubgridIX(subgrid);
    int iy = SubgridIY(subgrid);
    int iz = SubgridIZ(subgrid);

    int nx = SubgridNX(subgrid);
    int ny = SubgridNY(subgrid);
    int nz = SubgridNZ(subgrid);

    int nx_v = SubvectorNX(rhs_sub);
    int ny_v = SubvectorNY(rhs_sub);
    int nz_v = SubvectorNZ(rhs_sub);

    int ilo[3], ihi[3];
    int vlo[3], vhi[3];

    hypre_Box *set_box, *value_box;

    /* Active subgrid extent. */
    ilo[0] = ix;
    ilo[1] = iy;
    ilo[2] = iz;
    ihi[0] = ix + nx - 1;
    ihi[1] = iy + ny - 1;
    ihi[2] = iz + nz - 1;

    /* Full subvector extent, including ghosts. */
    vlo[0] = SubvectorIX(rhs_sub);
    vlo[1] = SubvectorIY(rhs_sub);
    vlo[2] = SubvectorIZ(rhs_sub);
    vhi[0] = vlo[0] + nx_v - 1;
    vhi[1] = vlo[1] + ny_v - 1;
    vhi[2] = vlo[2] + nz_v - 1;

    set_box = hypre_BoxCreate(PARFLOW_HYPRE_DIM);
    value_box = hypre_BoxCreate(PARFLOW_HYPRE_DIM);

    hypre_BoxSetExtents(set_box, ilo, ihi);
    hypre_BoxSetExtents(value_box, vlo, vhi);

    /* action = 0 : set values (one bulk call for the whole subgrid) */
    hypre_StructVectorSetBoxValues(*hypre_b, set_box, value_box, rhs_ptr,
                                   0, -1, 0);

    hypre_BoxDestroy(set_box);
    hypre_BoxDestroy(value_box);
  }
  HYPRE_StructVectorAssemble(*hypre_b);
}


void CopyHypreVectorToParflowVector(HYPRE_StructVector* hypre_x,
                                    Vector *            soln)
{
  Grid* grid = VectorGrid(soln);
  int sg;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    Subgrid* subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);
    Subvector* soln_sub = VectorSubvector(soln, sg);

    double* soln_ptr = SubvectorData(soln_sub);

    int ix = SubgridIX(subgrid);
    int iy = SubgridIY(subgrid);
    int iz = SubgridIZ(subgrid);

    int nx = SubgridNX(subgrid);
    int ny = SubgridNY(subgrid);
    int nz = SubgridNZ(subgrid);

    int nx_v = SubvectorNX(soln_sub);
    int ny_v = SubvectorNY(soln_sub);
    int nz_v = SubvectorNZ(soln_sub);

    int ilo[3], ihi[3];
    int vlo[3], vhi[3];

    hypre_Box *set_box, *value_box;

    ilo[0] = ix;
    ilo[1] = iy;
    ilo[2] = iz;
    ihi[0] = ix + nx - 1;
    ihi[1] = iy + ny - 1;
    ihi[2] = iz + nz - 1;

    vlo[0] = SubvectorIX(soln_sub);
    vlo[1] = SubvectorIY(soln_sub);
    vlo[2] = SubvectorIZ(soln_sub);
    vhi[0] = vlo[0] + nx_v - 1;
    vhi[1] = vlo[1] + ny_v - 1;
    vhi[2] = vlo[2] + nz_v - 1;

    set_box = hypre_BoxCreate(PARFLOW_HYPRE_DIM);
    value_box = hypre_BoxCreate(PARFLOW_HYPRE_DIM);

    hypre_BoxSetExtents(set_box, ilo, ihi);
    hypre_BoxSetExtents(value_box, vlo, vhi);

    /* action = -1 : get values (Bulk transfer for the subgrid) */
    hypre_StructVectorSetBoxValues(*hypre_x, set_box, value_box, soln_ptr,
                                   -1, -1, 0);

    hypre_BoxDestroy(set_box);
    hypre_BoxDestroy(value_box);
  }
}


void HypreAssembleGrid(
                       Grid*             pf_grid,
                       HYPRE_StructGrid* hypre_grid,
                       double*           dxyz
                       )
{
  int sg;

  int ilo[3];
  int ihi[3];

  if (pf_grid != NULL)
  {
    /* Free the HYPRE grid */
    if (*hypre_grid)
    {
      HYPRE_StructGridDestroy(*hypre_grid);
      hypre_grid = NULL;
    }

    /* Set the HYPRE grid */
    HYPRE_StructGridCreate(amps_CommWorld, 3, hypre_grid);

    /* Set local grid extents as global grid values */
    ForSubgridI(sg, GridSubgrids(pf_grid))
    {
      Subgrid* subgrid = GridSubgrid(pf_grid, sg);

      ilo[0] = SubgridIX(subgrid);
      ilo[1] = SubgridIY(subgrid);
      ilo[2] = SubgridIZ(subgrid);
      ihi[0] = ilo[0] + SubgridNX(subgrid) - 1;
      ihi[1] = ilo[1] + SubgridNY(subgrid) - 1;
      ihi[2] = ilo[2] + SubgridNZ(subgrid) - 1;

      dxyz[0] = SubgridDX(subgrid);
      dxyz[1] = SubgridDY(subgrid);
      dxyz[2] = SubgridDZ(subgrid);
    }
    HYPRE_StructGridSetExtents(*hypre_grid, ilo, ihi);
    HYPRE_StructGridAssemble(*hypre_grid);
  }
}

void HypreInitialize(Matrix*              pf_Bmat,
                     HYPRE_StructGrid*    hypre_grid,
                     HYPRE_StructStencil* hypre_stencil,
                     HYPRE_StructMatrix*  hypre_mat,
                     HYPRE_StructVector*  hypre_b,
                     HYPRE_StructVector*  hypre_x
                     )
{
  int full_ghosts[6] = { 1, 1, 1, 1, 1, 1 };
  int no_ghosts[6] = { 0, 0, 0, 0, 0, 0 };

  /* Matrix structure is fixed for the nonlinear solve. */
  int stencil_size = MatrixDataStencilSize(pf_Bmat);

  /* Set stencil parameters */
  if (!(*hypre_stencil))
  {
    HYPRE_StructStencilCreate(3, stencil_size, hypre_stencil);

    for (int i = 0; i < stencil_size; i++)
    {
      HYPRE_StructStencilSetElement(*hypre_stencil, i,
                                    &(MatrixDataStencil(pf_Bmat))[i * 3]);
    }
  }

  /* Set up new matrix */
  int symmetric = MatrixSymmetric(pf_Bmat);
  if (!(*hypre_mat))
  {
    HYPRE_StructMatrixCreate(amps_CommWorld, *hypre_grid,
                             *hypre_stencil,
                             hypre_mat);
    HYPRE_StructMatrixSetNumGhost(*hypre_mat, full_ghosts);
    HYPRE_StructMatrixSetSymmetric(*hypre_mat, symmetric);
    HYPRE_StructMatrixInitialize(*hypre_mat);
  }

  /* Set up new right-hand-side vector */
  if (!(*hypre_b))
  {
    HYPRE_StructVectorCreate(amps_CommWorld,
                             *hypre_grid,
                             hypre_b);
    HYPRE_StructVectorSetNumGhost(*hypre_b, no_ghosts);
    HYPRE_StructVectorInitialize(*hypre_b);
  }

  /* Set up new solution vector */
  if (!(*hypre_x))
  {
    HYPRE_StructVectorCreate(amps_CommWorld,
                             *hypre_grid,
                             hypre_x);
    HYPRE_StructVectorSetNumGhost(*hypre_x, full_ghosts);
    HYPRE_StructVectorInitialize(*hypre_x);
  }
  HYPRE_StructVectorSetConstantValues(*hypre_x, 0.0e0);
  HYPRE_StructVectorAssemble(*hypre_x);
}

/*
 * NOTE ON NAME: this function is still called HypreAssembleMatrixAsElements
 * for API compatibility with pf_pfmg.c (no caller changes needed). However,
 * Matrix values are inserted by box and stencil entry.
 */
void HypreAssembleMatrixAsElements(
                                   Matrix *            pf_Bmat,
                                   Matrix *            pf_Cmat,
                                   HYPRE_StructMatrix* hypre_mat,
                                   ProblemData *       problem_data
                                   )
{
  Grid *mat_grid = MatrixGrid(pf_Bmat);
  int sg;

  int stencil_size = MatrixDataStencilSize(pf_Bmat);
  int symmetric = MatrixSymmetric(pf_Bmat);

  int stencil_indices[7] = { 0, 1, 2, 3, 4, 5, 6 };
  int stencil_indices_symm[4] = { 0, 1, 2, 3 };

  Vector* top = ProblemDataIndexOfDomainTop(problem_data);

  ForSubgridI(sg, GridSubgrids(mat_grid))
  {
    Subgrid* subgrid = GridSubgrid(mat_grid, sg);
    Submatrix* pfB_sub = MatrixSubmatrix(pf_Bmat, sg);

    int ix = SubgridIX(subgrid);
    int iy = SubgridIY(subgrid);
    int iz = SubgridIZ(subgrid);

    int nx = SubgridNX(subgrid);
    int ny = SubgridNY(subgrid);
    int nz = SubgridNZ(subgrid);

    int nx_m = SubmatrixNX(pfB_sub);
    int ny_m = SubmatrixNY(pfB_sub);
    int nz_m = SubmatrixNZ(pfB_sub);

    int ilo[3], ihi[3], vlo[3], vhi[3];
    hypre_Box *set_box, *value_box;

    ilo[0] = ix;
    ilo[1] = iy;
    ilo[2] = iz;
    ihi[0] = ix + nx - 1;
    ihi[1] = iy + ny - 1;
    ihi[2] = iz + nz - 1;

    vlo[0] = SubmatrixIX(pfB_sub);
    vlo[1] = SubmatrixIY(pfB_sub);
    vlo[2] = SubmatrixIZ(pfB_sub);
    vhi[0] = vlo[0] + nx_m - 1;
    vhi[1] = vlo[1] + ny_m - 1;
    vhi[2] = vlo[2] + nz_m - 1;

    set_box = hypre_BoxCreate(PARFLOW_HYPRE_DIM);
    value_box = hypre_BoxCreate(PARFLOW_HYPRE_DIM);
    hypre_BoxSetExtents(set_box, ilo, ihi);
    hypre_BoxSetExtents(value_box, vlo, vhi);

    /* Bulk copy by stencil entry. */
    if (symmetric)
    {
      for (int stencil = 0; stencil < stencil_size; ++stencil)
      {
        /* symmetric stencil values are stored at 0, 2, 4, 6 */
        double *values = SubmatrixStencilData(pfB_sub, stencil * 2);
        hypre_StructMatrixSetBoxValues(*hypre_mat, set_box, value_box, 1,
                                       &stencil_indices_symm[stencil], values,
                                       0, -1, 0);
      }
    }
    else
    {
      for (int stencil = 0; stencil < stencil_size; ++stencil)
      {
        double *values = SubmatrixStencilData(pfB_sub, stencil);
        hypre_StructMatrixSetBoxValues(*hypre_mat, set_box, value_box, 1,
                                       &stencil_indices[stencil], values,
                                       0, -1, 0);
      }
    }

    hypre_BoxDestroy(set_box);
    hypre_BoxDestroy(value_box);

    /* Update the top surface from the C matrix. */
    if (pf_Cmat != NULL)
    {
      Submatrix* pfC_sub = MatrixSubmatrix(pf_Cmat, sg);
      Subvector* top_sub = VectorSubvector(top, sg);
      double* top_dat = SubvectorData(top_sub);
      int sy_v = SubvectorNX(top_sub);

      int i, j, k;
      int im = SubmatrixEltIndex(pfB_sub, ix, iy, iz);

      if (symmetric)
      {
        double *ep = SubmatrixStencilData(pfB_sub, 2);
        double *np = SubmatrixStencilData(pfB_sub, 4);
        double *up = SubmatrixStencilData(pfB_sub, 6);
        double *cp_c = SubmatrixStencilData(pfC_sub, 0);
        double coeffs_symm[4];
        int index[3];

        BoxLoopI1(i, j, k, ix, iy, 0, nx, ny, 1,
                  im, nx_m, ny_m, nz_m, 1, 1, 1,
        {
          int itop = SubvectorEltIndex(top_sub, i, j, 0);
          int ktop = (int)top_dat[itop];

          if (ktop >= 0)
          {
            int io = SubmatrixEltIndex(pfC_sub, i, j, 0);
            int ioB = SubmatrixEltIndex(pfB_sub, i, j, ktop);

            /* update diagonal coeff */
            coeffs_symm[0] = cp_c[io];                 //cp[ioB] is zero
            /* update east coeff */
            coeffs_symm[1] = ep[ioB];
            /* update north coeff */
            coeffs_symm[2] = np[ioB];
            /* update upper coeff */
            coeffs_symm[3] = up[ioB];                 // JB keeps upper term on surface. This should be zero

            index[0] = i;
            index[1] = j;
            index[2] = ktop;
            HYPRE_StructMatrixSetValues(*hypre_mat,
                                        index,
                                        stencil_size,
                                        stencil_indices_symm,
                                        coeffs_symm);
          }
        });
      }
      else
      {
        double *wp = SubmatrixStencilData(pfB_sub, 1);
        double *ep = SubmatrixStencilData(pfB_sub, 2);
        double *sop = SubmatrixStencilData(pfB_sub, 3);
        double *np = SubmatrixStencilData(pfB_sub, 4);
        double *lp = SubmatrixStencilData(pfB_sub, 5);
        double *up = SubmatrixStencilData(pfB_sub, 6);
        double *cp_c = SubmatrixStencilData(pfC_sub, 0);
        double *wp_c = SubmatrixStencilData(pfC_sub, 1);
        double *ep_c = SubmatrixStencilData(pfC_sub, 2);
        double *sop_c = SubmatrixStencilData(pfC_sub, 3);
        double *np_c = SubmatrixStencilData(pfC_sub, 4);
        double coeffs[7];
        int index[3];

        BoxLoopI1(i, j, k, ix, iy, 0, nx, ny, 1,
                  im, nx_m, ny_m, nz_m, 1, 1, 1,
        {
          int itop = SubvectorEltIndex(top_sub, i, j, 0);
          int ktop = (int)top_dat[itop];

          if (ktop >= 0)
          {
            int io = SubmatrixEltIndex(pfC_sub, i, j, 0);
            int ioB = SubmatrixEltIndex(pfB_sub, i, j, ktop);
            int k1;

            /* update diagonal coeff */
            coeffs[0] = cp_c[io];                 //cp[ioB] is zero
            /* update west coeff */
            k1 = (int)top_dat[itop - 1];
            if (k1 == ktop)
              coeffs[1] = wp_c[io];                    //wp[ioB] is zero
            else
              coeffs[1] = wp[ioB];
            /* update east coeff */
            k1 = (int)top_dat[itop + 1];
            if (k1 == ktop)
              coeffs[2] = ep_c[io];                    //ep[ioB] is zero
            else
              coeffs[2] = ep[ioB];
            /* update south coeff */
            k1 = (int)top_dat[itop - sy_v];
            if (k1 == ktop)
              coeffs[3] = sop_c[io];                    //sop[ioB] is zero
            else
              coeffs[3] = sop[ioB];
            /* update north coeff */
            k1 = (int)top_dat[itop + sy_v];
            if (k1 == ktop)
              coeffs[4] = np_c[io];                    //np[ioB] is zero
            else
              coeffs[4] = np[ioB];
            /* update lower coeff */
            coeffs[5] = lp[ioB];                 // JB keeps lower term on surface.
            /* update upper coeff */
            coeffs[6] = up[ioB];                 // JB keeps upper term on surface. This should be zero

            index[0] = i;
            index[1] = j;
            index[2] = ktop;
            HYPRE_StructMatrixSetValues(*hypre_mat,
                                        index,
                                        stencil_size,
                                        stencil_indices, coeffs);
          }
        });
      }
    }
  }   /* End subgrid loop */

  HYPRE_StructMatrixAssemble(*hypre_mat);
}

#endif // HAVE_HYPRE