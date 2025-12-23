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

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Common functions for HYPRE
 *--------------------------------------------------------------------------*/

#ifdef HAVE_HYPRE
#include "hypre_dependences.h"

void CopyParFlowVectorToHypreVector(Vector *            rhs,
                                    HYPRE_StructVector* hypre_b)
{
  Grid* grid = VectorGrid(rhs);
  int sg;
  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;
  int i, j, k;
  int index[3];

  ForSubgridI(sg, GridSubgrids(grid))
  {
    Subgrid* subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);
    Subvector* rhs_sub = VectorSubvector(rhs, sg);

    double* rhs_ptr = SubvectorData(rhs_sub);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_v = SubvectorNX(rhs_sub);
    ny_v = SubvectorNY(rhs_sub);
    nz_v = SubvectorNZ(rhs_sub);

    int iv = SubvectorEltIndex(rhs_sub, ix, iy, iz);


    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      index[0] = i;
      index[1] = j;
      index[2] = k;

#if HYPRE_RELEASE_NUMBER >= 30000
      HYPRE_StructVectorSetValues(*hypre_b, index, &rhs_ptr[iv]);
#else
      HYPRE_StructVectorSetValues(*hypre_b, index, rhs_ptr[iv]);
#endif
    });
  }
  HYPRE_StructVectorAssemble(*hypre_b);
}


void CopyHypreVectorToParflowVector(HYPRE_StructVector* hypre_x,
                                    Vector *            soln)
{
  Grid* grid = VectorGrid(soln);
  int sg;
  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;
  int i, j, k;
  int index[3];

  ForSubgridI(sg, GridSubgrids(grid))
  {
    Subgrid* subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);
    Subvector* soln_sub = VectorSubvector(soln, sg);

    double* soln_ptr = SubvectorData(soln_sub);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_v = SubvectorNX(soln_sub);
    ny_v = SubvectorNY(soln_sub);
    nz_v = SubvectorNZ(soln_sub);

    int iv = SubvectorEltIndex(soln_sub, ix, iy, iz);

    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      index[0] = i;
      index[1] = j;
      index[2] = k;

      double value;
      HYPRE_StructVectorGetValues(*hypre_x, index, &value);
      soln_ptr[iv] = value;
    });
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

  /* For remainder of routine, assume matrix is structured the same for
   * entire nonlinear solve process */
  /* Set stencil parameters */
  int stencil_size = MatrixDataStencilSize(pf_Bmat);

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

void HypreAssembleMatrixAsElements(
                                   Matrix *            pf_Bmat,
                                   Matrix *            pf_Cmat,
                                   HYPRE_StructMatrix* hypre_mat,
                                   ProblemData *       problem_data
                                   )
{
  Grid *mat_grid = MatrixGrid(pf_Bmat);
  double *cp, *wp = NULL, *ep, *sop = NULL, *np, *lp = NULL, *up = NULL;
  double *cp_c, *wp_c = NULL, *ep_c = NULL, *sop_c = NULL, *np_c = NULL, *top_dat;
  int sg;
  int ix, iy, iz;
  int nx, ny, nz;
  int nx_m, ny_m, nz_m, sy_v;
  int i, j, k, itop, k1, ktop;
  int im, io;

  int stencil_indices[7] = { 0, 1, 2, 3, 4, 5, 6 };
  int stencil_indices_symm[4] = { 0, 1, 2, 3 };
  int index[3];

  double coeffs[7];
  double coeffs_symm[4];

  int stencil_size = MatrixDataStencilSize(pf_Bmat);
  int symmetric = MatrixSymmetric(pf_Bmat);

  Vector* top = ProblemDataIndexOfDomainTop(problem_data);

  if (pf_Cmat == NULL) /* No overland flow */
  {
    ForSubgridI(sg, GridSubgrids(mat_grid))
    {
      Subgrid* subgrid = GridSubgrid(mat_grid, sg);

      Submatrix* pfB_sub = MatrixSubmatrix(pf_Bmat, sg);


      if (symmetric)
      {
        /* Pull off upper diagonal coeffs here for symmetric part */
        cp = SubmatrixStencilData(pfB_sub, 0);
        ep = SubmatrixStencilData(pfB_sub, 2);
        np = SubmatrixStencilData(pfB_sub, 4);
        up = SubmatrixStencilData(pfB_sub, 6);
      }
      else
      {
        cp = SubmatrixStencilData(pfB_sub, 0);
        wp = SubmatrixStencilData(pfB_sub, 1);
        ep = SubmatrixStencilData(pfB_sub, 2);
        sop = SubmatrixStencilData(pfB_sub, 3);
        np = SubmatrixStencilData(pfB_sub, 4);
        lp = SubmatrixStencilData(pfB_sub, 5);
        up = SubmatrixStencilData(pfB_sub, 6);
      }

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      nx_m = SubmatrixNX(pfB_sub);
      ny_m = SubmatrixNY(pfB_sub);
      nz_m = SubmatrixNZ(pfB_sub);

      im = SubmatrixEltIndex(pfB_sub, ix, iy, iz);

      if (symmetric)
      {
        BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                  im, nx_m, ny_m, nz_m, 1, 1, 1,
        {
          coeffs_symm[0] = cp[im];
          coeffs_symm[1] = ep[im];
          coeffs_symm[2] = np[im];
          coeffs_symm[3] = up[im];
          index[0] = i;
          index[1] = j;
          index[2] = k;
          HYPRE_StructMatrixSetValues(*hypre_mat,
                                      index,
                                      stencil_size,
                                      stencil_indices_symm,
                                      coeffs_symm);
        });
      }
      else
      {
        BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                  im, nx_m, ny_m, nz_m, 1, 1, 1,
        {
          coeffs[0] = cp[im];
          coeffs[1] = wp[im];
          coeffs[2] = ep[im];
          coeffs[3] = sop[im];
          coeffs[4] = np[im];
          coeffs[5] = lp[im];
          coeffs[6] = up[im];
          index[0] = i;
          index[1] = j;
          index[2] = k;
          HYPRE_StructMatrixSetValues(*hypre_mat,
                                      index,
                                      stencil_size,
                                      stencil_indices, coeffs);
        });
      }
    }   /* End subgrid loop */
  }
  else  /* Overland flow is activated. Update preconditioning matrix */
  {
    ForSubgridI(sg, GridSubgrids(mat_grid))
    {
      Subgrid* subgrid = GridSubgrid(mat_grid, sg);

      Submatrix* pfB_sub = MatrixSubmatrix(pf_Bmat, sg);
      Submatrix* pfC_sub = MatrixSubmatrix(pf_Cmat, sg);

      Subvector* top_sub = VectorSubvector(top, sg);

      if (symmetric)
      {
        /* Pull off upper diagonal coeffs here for symmetric part */
        cp = SubmatrixStencilData(pfB_sub, 0);
        ep = SubmatrixStencilData(pfB_sub, 2);
        np = SubmatrixStencilData(pfB_sub, 4);
        up = SubmatrixStencilData(pfB_sub, 6);

        //          cp_c    = SubmatrixStencilData(pfC_sub, 0);
        //          ep_c    = SubmatrixStencilData(pfC_sub, 2);
        //          np_c    = SubmatrixStencilData(pfC_sub, 4);
        cp_c = SubmatrixStencilData(pfC_sub, 0);
        wp_c = SubmatrixStencilData(pfC_sub, 1);
        ep_c = SubmatrixStencilData(pfC_sub, 2);
        sop_c = SubmatrixStencilData(pfC_sub, 3);
        np_c = SubmatrixStencilData(pfC_sub, 4);
        top_dat = SubvectorData(top_sub);
      }
      else
      {
        cp = SubmatrixStencilData(pfB_sub, 0);
        wp = SubmatrixStencilData(pfB_sub, 1);
        ep = SubmatrixStencilData(pfB_sub, 2);
        sop = SubmatrixStencilData(pfB_sub, 3);
        np = SubmatrixStencilData(pfB_sub, 4);
        lp = SubmatrixStencilData(pfB_sub, 5);
        up = SubmatrixStencilData(pfB_sub, 6);

        cp_c = SubmatrixStencilData(pfC_sub, 0);
        wp_c = SubmatrixStencilData(pfC_sub, 1);
        ep_c = SubmatrixStencilData(pfC_sub, 2);
        sop_c = SubmatrixStencilData(pfC_sub, 3);
        np_c = SubmatrixStencilData(pfC_sub, 4);
        top_dat = SubvectorData(top_sub);
      }

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      nx_m = SubmatrixNX(pfB_sub);
      ny_m = SubmatrixNY(pfB_sub);
      nz_m = SubmatrixNZ(pfB_sub);

      sy_v = SubvectorNX(top_sub);

      im = SubmatrixEltIndex(pfB_sub, ix, iy, iz);

      if (symmetric)
      {
        BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                  im, nx_m, ny_m, nz_m, 1, 1, 1,
        {
          itop = SubvectorEltIndex(top_sub, i, j, 0);
          ktop = (int)top_dat[itop];
          io = SubmatrixEltIndex(pfC_sub, i, j, iz);
          /* Since we are using a boxloop, we need to check for top index
           * to update with the surface contributions */
          if (ktop == k)
          {
            /* update diagonal coeff */
            coeffs_symm[0] = cp_c[io];               //cp[im] is zero
            /* update east coeff */
            coeffs_symm[1] = ep[im];
            /* update north coeff */
            coeffs_symm[2] = np[im];
            /* update upper coeff */
            coeffs_symm[3] = up[im];               // JB keeps upper term on surface. This should be zero
          }
          else
          {
            coeffs_symm[0] = cp[im];
            coeffs_symm[1] = ep[im];
            coeffs_symm[2] = np[im];
            coeffs_symm[3] = up[im];
          }

          index[0] = i;
          index[1] = j;
          index[2] = k;
          HYPRE_StructMatrixSetValues(*hypre_mat,
                                      index,
                                      stencil_size,
                                      stencil_indices_symm,
                                      coeffs_symm);
        });
      }
      else
      {
        BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                  im, nx_m, ny_m, nz_m, 1, 1, 1,
        {
          itop = SubvectorEltIndex(top_sub, i, j, 0);
          ktop = (int)top_dat[itop];
          io = SubmatrixEltIndex(pfC_sub, i, j, iz);
          /* Since we are using a boxloop, we need to check for top index
           * to update with the surface contributions */
          if (ktop == k)
          {
            /* update diagonal coeff */
            coeffs[0] = cp_c[io];               //cp[im] is zero
            /* update west coeff */
            k1 = (int)top_dat[itop - 1];
            if (k1 == ktop)
              coeffs[1] = wp_c[io];                  //wp[im] is zero
            else
              coeffs[1] = wp[im];
            /* update east coeff */
            k1 = (int)top_dat[itop + 1];
            if (k1 == ktop)
              coeffs[2] = ep_c[io];                  //ep[im] is zero
            else
              coeffs[2] = ep[im];
            /* update south coeff */
            k1 = (int)top_dat[itop - sy_v];
            if (k1 == ktop)
              coeffs[3] = sop_c[io];                  //sop[im] is zero
            else
              coeffs[3] = sop[im];
            /* update north coeff */
            k1 = (int)top_dat[itop + sy_v];
            if (k1 == ktop)
              coeffs[4] = np_c[io];                  //np[im] is zero
            else
              coeffs[4] = np[im];
            /* update upper coeff */
            coeffs[5] = lp[im];               // JB keeps lower term on surface.
            /* update upper coeff */
            coeffs[6] = up[im];               // JB keeps upper term on surface. This should be zero
          }
          else
          {
            coeffs[0] = cp[im];
            coeffs[1] = wp[im];
            coeffs[2] = ep[im];
            coeffs[3] = sop[im];
            coeffs[4] = np[im];
            coeffs[5] = lp[im];
            coeffs[6] = up[im];
          }

          index[0] = i;
          index[1] = j;
          index[2] = k;
          HYPRE_StructMatrixSetValues(*hypre_mat,
                                      index,
                                      stencil_size,
                                      stencil_indices, coeffs);
        });
      }
    }   /* End subgrid loop */
  }  /* end if pf_Cmat==NULL */

  HYPRE_StructMatrixAssemble(*hypre_mat);
}

#endif // HAVE_HYPRE
