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

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

#ifdef HAVE_HYPRE
#include "hypre_dependences.h"

typedef struct {
  int max_iter;
  int num_pre_relax;
  int num_post_relax;
  int smoother;
  int raptype;

  int time_index_pfmg;
  int time_index_copy_hypre;
} PublicXtra;

typedef struct {
  double dxyz[3];

  HYPRE_StructGrid hypre_grid;
  HYPRE_StructMatrix hypre_mat;
  HYPRE_StructVector hypre_b, hypre_x;
  HYPRE_StructStencil hypre_stencil;

  HYPRE_StructSolver hypre_pfmg_data;
} InstanceXtra;

#endif


/*--------------------------------------------------------------------------
 * PFMG
 *--------------------------------------------------------------------------*/

void         PFMG(
                  Vector *soln,
                  Vector *rhs,
                  double  tol,
                  int     zero)
{
  (void)zero;

#ifdef HAVE_HYPRE
  PFModule           *this_module = ThisPFModule;
  InstanceXtra       *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra         *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  HYPRE_StructMatrix hypre_mat = instance_xtra->hypre_mat;
  HYPRE_StructVector hypre_b = instance_xtra->hypre_b;
  HYPRE_StructVector hypre_x = instance_xtra->hypre_x;

  HYPRE_StructSolver hypre_pfmg_data = instance_xtra->hypre_pfmg_data;

  Grid               *grid = VectorGrid(rhs);
  Subgrid            *subgrid;
  int sg;

  Subvector          *rhs_sub;
  Subvector          *soln_sub;

  double             *rhs_ptr;
  double             *soln_ptr;
  double value;

  int index[3];

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;
  int i, j, k;
  int iv;

  int num_iterations;
  double rel_norm;

  /* Copy rhs to hypre_b vector. */
  BeginTiming(public_xtra->time_index_copy_hypre);

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);
    rhs_sub = VectorSubvector(rhs, sg);

    rhs_ptr = SubvectorData(rhs_sub);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_v = SubvectorNX(rhs_sub);
    ny_v = SubvectorNY(rhs_sub);
    nz_v = SubvectorNZ(rhs_sub);

    iv = SubvectorEltIndex(rhs_sub, ix, iy, iz);

    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      index[0] = i;
      index[1] = j;
      index[2] = k;

      HYPRE_StructVectorSetValues(hypre_b, index, rhs_ptr[iv]);
    });
  }
  HYPRE_StructVectorAssemble(hypre_b);

  EndTiming(public_xtra->time_index_copy_hypre);

  if (tol > 0.0)
  {
    IfLogging(1)
    {
      HYPRE_StructPFMGSetLogging(instance_xtra->hypre_pfmg_data, 1);
    }
  }

  /* Invoke the preconditioner using a zero initial guess */
  HYPRE_StructPFMGSetZeroGuess(hypre_pfmg_data);

  BeginTiming(public_xtra->time_index_pfmg);

  HYPRE_StructPFMGSolve(hypre_pfmg_data, hypre_mat, hypre_b, hypre_x);

  EndTiming(public_xtra->time_index_pfmg);

  if (tol > 0.0)
  {
    IfLogging(1)
    {
      FILE  *log_file;

      HYPRE_StructPFMGGetNumIterations(hypre_pfmg_data, &num_iterations);
      HYPRE_StructPFMGGetFinalRelativeResidualNorm(hypre_pfmg_data,
                                                   &rel_norm);

      log_file = OpenLogFile("PFMG");
      fprintf(log_file, "PFMG num. its: %i  PFMG Final norm: %12.4e\n",
              num_iterations, rel_norm);
      CloseLogFile(log_file);
    }
  }

  /* Copy solution from hypre_x vector to the soln vector. */
  BeginTiming(public_xtra->time_index_copy_hypre);

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);
    soln_sub = VectorSubvector(soln, sg);

    soln_ptr = SubvectorData(soln_sub);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_v = SubvectorNX(soln_sub);
    ny_v = SubvectorNY(soln_sub);
    nz_v = SubvectorNZ(soln_sub);

    iv = SubvectorEltIndex(soln_sub, ix, iy, iz);

    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      index[0] = i;
      index[1] = j;
      index[2] = k;

      HYPRE_StructVectorGetValues(hypre_x, index, &value);
      soln_ptr[iv] = value;
    });
  }
  EndTiming(public_xtra->time_index_copy_hypre);
#else
  amps_Printf("Error: Parflow not compiled with hypre, can't use pfmg\n");
#endif
}

/*--------------------------------------------------------------------------
 * PFMGInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *PFMGInitInstanceXtra(
                                Problem *    problem,
                                Grid *       grid,
                                ProblemData *problem_data,
                                Matrix *     pf_Bmat,
                                Matrix *     pf_Cmat,
                                double *     temp_data)
{
#ifdef HAVE_HYPRE
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;

  int max_iter = public_xtra->max_iter;
  int num_pre_relax = public_xtra->num_pre_relax;
  int num_post_relax = public_xtra->num_post_relax;
  int smoother = public_xtra->smoother;
  int raptype = public_xtra->raptype;

  Grid               *mat_grid;
  Subgrid            *subgrid;
  int sg;

  Vector      *top = ProblemDataIndexOfDomainTop(problem_data);               //DOK
  Subvector      *top_sub = NULL;

  Submatrix          *pfB_sub, *pfC_sub;
  double             *cp, *wp = NULL, *ep, *sop = NULL, *np, *lp = NULL, *up = NULL;
  double             *cp_c, *wp_c = NULL, *ep_c = NULL, *sop_c = NULL, *np_c = NULL, *top_dat;

  double coeffs[7];
  double coeffs_symm[4];

  int i, j, k, itop, k1, ktop;
  int ix, iy, iz;
  int nx, ny, nz;
  int nx_m, ny_m, nz_m, sy_m, sy_v;
  int im, io;
  int stencil_size;
  int symmetric;

  int full_ghosts[6] = { 1, 1, 1, 1, 1, 1 };
  int no_ghosts[6] = { 0, 0, 0, 0, 0, 0 };
  int stencil_indices[7] = { 0, 1, 2, 3, 4, 5, 6 };
  int stencil_indices_symm[4] = { 0, 1, 2, 3 };
  int index[3];
  int ilo[3];
  int ihi[3];

  (void)problem;
  (void)problem_data;
  (void)temp_data;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (grid != NULL)
  {
    /* Free the HYPRE grid */
    if (instance_xtra->hypre_grid)
    {
      HYPRE_StructGridDestroy(instance_xtra->hypre_grid);
      instance_xtra->hypre_grid = NULL;
    }

    /* Set the HYPRE grid */
    HYPRE_StructGridCreate(MPI_COMM_WORLD, 3, &(instance_xtra->hypre_grid));

    /* Set local grid extents as global grid values */
    ForSubgridI(sg, GridSubgrids(grid))
    {
      subgrid = GridSubgrid(grid, sg);

      ilo[0] = SubgridIX(subgrid);
      ilo[1] = SubgridIY(subgrid);
      ilo[2] = SubgridIZ(subgrid);
      ihi[0] = ilo[0] + SubgridNX(subgrid) - 1;
      ihi[1] = ilo[1] + SubgridNY(subgrid) - 1;
      ihi[2] = ilo[2] + SubgridNZ(subgrid) - 1;

      instance_xtra->dxyz[0] = SubgridDX(subgrid);
      instance_xtra->dxyz[1] = SubgridDY(subgrid);
      instance_xtra->dxyz[2] = SubgridDZ(subgrid);
    }
    HYPRE_StructGridSetExtents(instance_xtra->hypre_grid, ilo, ihi);
    HYPRE_StructGridAssemble(instance_xtra->hypre_grid);
  }

  /* Reset the HYPRE solver for each recompute of the PC matrix.
   * This reset will require a matrix copy from PF format to HYPRE format. */
  if (pf_Bmat != NULL)
  {
    /* Free old solver data because HYPRE requires a new solver if
     * matrix values change */
    if (instance_xtra->hypre_pfmg_data)
    {
      HYPRE_StructPFMGDestroy(instance_xtra->hypre_pfmg_data);
      instance_xtra->hypre_pfmg_data = NULL;
    }

    /* For remainder of routine, assume matrix is structured the same for
     * entire nonlinear solve process */
    /* Set stencil parameters */
    stencil_size = MatrixDataStencilSize(pf_Bmat);
    if (!(instance_xtra->hypre_stencil))
    {
      HYPRE_StructStencilCreate(3, stencil_size,
                                &(instance_xtra->hypre_stencil));

      for (i = 0; i < stencil_size; i++)
      {
        HYPRE_StructStencilSetElement(instance_xtra->hypre_stencil, i,
                                      &(MatrixDataStencil(pf_Bmat))[i * 3]);
      }
    }

    /* Set up new matrix */
    symmetric = MatrixSymmetric(pf_Bmat);
    if (!(instance_xtra->hypre_mat))
    {
      HYPRE_StructMatrixCreate(MPI_COMM_WORLD, instance_xtra->hypre_grid,
                               instance_xtra->hypre_stencil,
                               &(instance_xtra->hypre_mat));
      HYPRE_StructMatrixSetNumGhost(instance_xtra->hypre_mat, full_ghosts);
      HYPRE_StructMatrixSetSymmetric(instance_xtra->hypre_mat, symmetric);
      HYPRE_StructMatrixInitialize(instance_xtra->hypre_mat);
    }

    /* Set up new right-hand-side vector */
    if (!(instance_xtra->hypre_b))
    {
      HYPRE_StructVectorCreate(MPI_COMM_WORLD,
                               instance_xtra->hypre_grid,
                               &(instance_xtra->hypre_b));
      HYPRE_StructVectorSetNumGhost(instance_xtra->hypre_b, no_ghosts);
      HYPRE_StructVectorInitialize(instance_xtra->hypre_b);
    }

    /* Set up new solution vector */
    if (!(instance_xtra->hypre_x))
    {
      HYPRE_StructVectorCreate(MPI_COMM_WORLD,
                               instance_xtra->hypre_grid,
                               &(instance_xtra->hypre_x));
      HYPRE_StructVectorSetNumGhost(instance_xtra->hypre_x, full_ghosts);
      HYPRE_StructVectorInitialize(instance_xtra->hypre_x);
    }
    HYPRE_StructVectorSetConstantValues(instance_xtra->hypre_x, 0.0e0);
    HYPRE_StructVectorAssemble(instance_xtra->hypre_x);

    /* Copy the matrix entries */
    BeginTiming(public_xtra->time_index_copy_hypre);

    mat_grid = MatrixGrid(pf_Bmat);
    if (pf_Cmat == NULL) /* No overland flow */
    {
      ForSubgridI(sg, GridSubgrids(mat_grid))
      {
        subgrid = GridSubgrid(mat_grid, sg);

        pfB_sub = MatrixSubmatrix(pf_Bmat, sg);

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
            HYPRE_StructMatrixSetValues(instance_xtra->hypre_mat,
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
            HYPRE_StructMatrixSetValues(instance_xtra->hypre_mat,
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
        subgrid = GridSubgrid(mat_grid, sg);

        pfB_sub = MatrixSubmatrix(pf_Bmat, sg);
        pfC_sub = MatrixSubmatrix(pf_Cmat, sg);

        top_sub = VectorSubvector(top, sg);

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

        sy_m = nx_m;

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
            HYPRE_StructMatrixSetValues(instance_xtra->hypre_mat,
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
            HYPRE_StructMatrixSetValues(instance_xtra->hypre_mat,
                                        index,
                                        stencil_size,
                                        stencil_indices, coeffs);
          });
        }
      }   /* End subgrid loop */
    }  /* end if pf_Cmat==NULL */
    HYPRE_StructMatrixAssemble(instance_xtra->hypre_mat);

    EndTiming(public_xtra->time_index_copy_hypre);

    /* Set up the PFMG preconditioner */
    HYPRE_StructPFMGCreate(MPI_COMM_WORLD,
                           &(instance_xtra->hypre_pfmg_data));

    HYPRE_StructPFMGSetTol(instance_xtra->hypre_pfmg_data, 1.0e-30);
    /* Set user parameters for PFMG */
    HYPRE_StructPFMGSetMaxIter(instance_xtra->hypre_pfmg_data, max_iter);
    HYPRE_StructPFMGSetNumPreRelax(instance_xtra->hypre_pfmg_data,
                                   num_pre_relax);
    HYPRE_StructPFMGSetNumPostRelax(instance_xtra->hypre_pfmg_data,
                                    num_post_relax);
    /* Jacobi = 0; weighted Jacobi = 1; red-black GS symmetric = 2; red-black GS non-symmetric = 3 */
    HYPRE_StructPFMGSetRelaxType(instance_xtra->hypre_pfmg_data, smoother);

    /* Galerkin=0; non-Galkerkin=1 */
    HYPRE_StructPFMGSetRAPType(instance_xtra->hypre_pfmg_data, raptype);

    HYPRE_StructPFMGSetSkipRelax(instance_xtra->hypre_pfmg_data, 1);

    HYPRE_StructPFMGSetDxyz(instance_xtra->hypre_pfmg_data,
                            instance_xtra->dxyz);

    HYPRE_StructPFMGSetup(instance_xtra->hypre_pfmg_data,
                          instance_xtra->hypre_mat,
                          instance_xtra->hypre_b, instance_xtra->hypre_x);
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
#else
  return NULL;
#endif
}


/*--------------------------------------------------------------------------
 * PFMGFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  PFMGFreeInstanceXtra()
{
#ifdef HAVE_HYPRE
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    if (instance_xtra->hypre_pfmg_data)
      HYPRE_StructPFMGDestroy(instance_xtra->hypre_pfmg_data);
    if (instance_xtra->hypre_mat)
      HYPRE_StructMatrixDestroy(instance_xtra->hypre_mat);
    if (instance_xtra->hypre_b)
      HYPRE_StructVectorDestroy(instance_xtra->hypre_b);
    if (instance_xtra->hypre_x)
      HYPRE_StructVectorDestroy(instance_xtra->hypre_x);
    if (instance_xtra->hypre_stencil)
      HYPRE_StructStencilDestroy(instance_xtra->hypre_stencil);
    if (instance_xtra->hypre_grid)
      HYPRE_StructGridDestroy(instance_xtra->hypre_grid);

    tfree(instance_xtra);
  }
#endif
}

/*--------------------------------------------------------------------------
 * PFMGNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *PFMGNewPublicXtra(char *name)
{
#ifdef HAVE_HYPRE
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;
  char key[IDB_MAX_KEY_LEN];
  char          *smoother_name;
  NameArray smoother_switch_na;
  int smoother;
  char          *raptype_name;
  NameArray raptype_switch_na;
  int raptype;

  public_xtra = ctalloc(PublicXtra, 1);

  sprintf(key, "%s.MaxIter", name);
  public_xtra->max_iter = GetIntDefault(key, 1);

  sprintf(key, "%s.NumPreRelax", name);
  public_xtra->num_pre_relax = GetIntDefault(key, 1);

  sprintf(key, "%s.NumPostRelax", name);
  public_xtra->num_post_relax = GetIntDefault(key, 1);

  /* Use a dummy place holder so that cardinalities match
   * with what HYPRE expects */
  smoother_switch_na = NA_NewNameArray("Jacobi WJacobi RBGaussSeidelSymmetric RBGaussSeidelNonSymmetric");
  sprintf(key, "%s.Smoother", name);
  smoother_name = GetStringDefault(key, "RBGaussSeidelNonSymmetric");
  smoother = NA_NameToIndex(smoother_switch_na, smoother_name);
  if (smoother >= 0)
  {
    public_xtra->smoother = NA_NameToIndex(smoother_switch_na,
                                           smoother_name);
  }
  else
  {
    InputError("Error: Invalid value <%s> for key <%s>.\n",
               smoother_name, key);
  }
  NA_FreeNameArray(smoother_switch_na);

  raptype_switch_na = NA_NewNameArray("Galerkin NonGalerkin");
  sprintf(key, "%s.RAPType", name);
  raptype_name = GetStringDefault(key, "NonGalerkin");
  raptype = NA_NameToIndex(raptype_switch_na, raptype_name);
  if (raptype >= 0)
  {
    public_xtra->raptype = raptype;
  }
  else
  {
    InputError("Error: Invalid value <%s> for key <%s>.\n",
               raptype_name, key);
  }
  NA_FreeNameArray(raptype_switch_na);

  if (raptype == 0 && smoother > 1)
  {
    InputError("Error: Galerkin RAPType is not compatible with Smoother <%s>.\n",
               smoother_name, key);
  }

  public_xtra->time_index_pfmg = RegisterTiming("PFMG");
  public_xtra->time_index_copy_hypre = RegisterTiming("HYPRE_Copies");

  PFModulePublicXtra(this_module) = public_xtra;

  return this_module;
#else
  amps_Printf("Error: Parflow not compiled with hypre, can't use pfmg\n");
  return NULL;
#endif
}

/*-------------------------------------------------------------------------
 * PFMGFreePublicXtra
 *-------------------------------------------------------------------------*/

void  PFMGFreePublicXtra()
{
#ifdef HAVE_HYPRE
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
#endif
}

/*--------------------------------------------------------------------------
 * PFMGSizeOfTempData
 *--------------------------------------------------------------------------*/

int  PFMGSizeOfTempData()
{
  return 0;
}

