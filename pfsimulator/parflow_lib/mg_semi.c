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
* Multigrid with semi-coarsening strategy.
*
*****************************************************************************/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  PFModule  *smooth;
  PFModule  *solve;

  int max_iter;
  int max_levels;
  int min_NX, min_NY, min_NZ;

  int time_index;
} PublicXtra;

typedef struct {
  PFModule        **smooth_l;
  PFModule         *solve;

  /* InitInstanceXtra arguments */
  double           *temp_data;

  /* instance data */
  int num_levels;

  int              *coarsen_l;

  Grid            **grid_l;

  SubregionArray  **f_sra_l;
  SubregionArray  **c_sra_l;

  ComputePkg      **restrict_compute_pkg_l;
  ComputePkg      **prolong_compute_pkg_l;

  Matrix          **A_l;
  Matrix          **P_l;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * MGSemi:
 *   Solves A x = b.
 *--------------------------------------------------------------------------
 *
 * We use the following convergence test:
 *
 *       ||r||_2
 *       -------  < tol
 *       ||b||_2
 *
 * We implement the test as:
 *
 *       <r,r>  <  (tol^2)*<b,b> = eps
 *
 *--------------------------------------------------------------------------*/

void     MGSemi(
                Vector *x,
                Vector *b,
                double  tol,
                int     zero)
{
  PUSH_NVTX("MGSemi",2)
  
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int max_iter = (public_xtra->max_iter);

  PFModule        **smooth_l = (instance_xtra->smooth_l);
  PFModule         *solve = (instance_xtra->solve);

  int num_levels = (instance_xtra->num_levels);

  SubregionArray  **f_sra_l = (instance_xtra->f_sra_l);
  SubregionArray  **c_sra_l = (instance_xtra->c_sra_l);

  ComputePkg      **restrict_compute_pkg_l =
    (instance_xtra->restrict_compute_pkg_l);
  ComputePkg      **prolong_compute_pkg_l =
    (instance_xtra->prolong_compute_pkg_l);

  CommPkg         **restrict_comm_pkg_l = NULL;
  CommPkg         **prolong_comm_pkg_l = NULL;

  Matrix          **A_l = (instance_xtra->A_l);
  Vector          **x_l = NULL;
  Vector          **b_l = NULL;

  Matrix          **P_l = (instance_xtra->P_l);

  Vector          **temp_vec_l = NULL;

  Matrix           *A = A_l[0];

  double eps = 0.0;
  double b_dot_b = 0.0, r_dot_r = 0.0;
  int almost_converged = 0;

  int l;
  int i = 0;

  double           *norm_log = NULL;
  double           *rel_norm_log = NULL;


  if (tol > 0.0)
  {
    IfLogging(1)
    {
      norm_log = talloc(double, max_iter + 1);
      rel_norm_log = talloc(double, max_iter + 1);
    }
  }

  /*-----------------------------------------------------------------------
   * Begin timing
   *-----------------------------------------------------------------------*/

  BeginTiming(public_xtra->time_index);


  /*-----------------------------------------------------------------------
   * Allocate temp vectors
   *-----------------------------------------------------------------------*/

  x_l = talloc(Vector *, num_levels);
  b_l = talloc(Vector *, num_levels);
  temp_vec_l = talloc(Vector *, num_levels);

  restrict_comm_pkg_l = talloc(CommPkg *, (num_levels - 1));
  prolong_comm_pkg_l = talloc(CommPkg *, (num_levels - 1));

  temp_vec_l[0] = NewVector(instance_xtra->grid_l[0], 1, 1);
  for (l = 0; l < (num_levels - 1); l++)
  {
    /*-----------------------------------------------------------------
     * Set up temporary vectors: x_l, b_l, temp_vec_l
     *-----------------------------------------------------------------*/
    // Non SAMRAI grids
    x_l[l + 1] = NewVectorType(instance_xtra->grid_l[l + 1], 1, 1, vector_non_samrai);
    b_l[l + 1] = NewVectorType(instance_xtra->grid_l[l + 1], 1, 1, vector_non_samrai);
    temp_vec_l[l + 1] = NewVectorType(instance_xtra->grid_l[l + 1], 1, 1, vector_non_samrai);

    /* Set up comm_pkg's */

    /* SGS not done */
    restrict_comm_pkg_l[l] =
      NewVectorCommPkg(temp_vec_l[l],
                       (instance_xtra->restrict_compute_pkg_l[l]));
    prolong_comm_pkg_l[l] =
      NewVectorCommPkg(temp_vec_l[l],
                       (instance_xtra->prolong_compute_pkg_l[l]));
  }

  /*-----------------------------------------------------------------------
   * Do V-cycles:
   *   For each index l, "fine" = l, "coarse" = (l+1)
   *-----------------------------------------------------------------------*/

  if ((i + 1) > max_iter)
  {
    Copy(b, x);
    EndTiming(public_xtra->time_index);
    return;
  }

  if (tol > 0.0)
  {
    /* eps = (tol^2)*<b,b> */
    b_dot_b = InnerProd(b, b);
    eps = (tol * tol) * b_dot_b;
  }

  /* smooth (use `zero' to determine initial x) */
  PFModuleInvokeType(LinearSolverInvoke, smooth_l[0], (x, b, 0.0, zero));

  PUSH_NVTX("MGSemi_solveloop",4)
  while (++i)
  {
    /*--------------------------------------------------------------------
     * Down cycle
     *--------------------------------------------------------------------*/

    /* first smoothing is already done */

    /* compute residual (b - Ax) */
    Copy(b, temp_vec_l[0]);
    Matvec(-1.0, A, x, 1.0, temp_vec_l[0]);

    /* do preliminary convergence check */
    if (tol > 0.0)
    {
      if (!almost_converged)
      {
        r_dot_r = InnerProd(temp_vec_l[0], temp_vec_l[0]);
        if (r_dot_r < eps)
          almost_converged = 1;

        IfLogging(1)
        {
          norm_log[i - 1] = sqrt(r_dot_r);
          rel_norm_log[i - 1] = b_dot_b ? sqrt(r_dot_r / b_dot_b) : 0.0;
        }
      }
    }

    /* restrict residual */
    MGSemiRestrict(A, temp_vec_l[0], b_l[1], P_l[0],
                   f_sra_l[0], c_sra_l[0],
                   restrict_compute_pkg_l[0], restrict_comm_pkg_l[0]);

#if 0
    /* for debugging purposes */
    PrintVector("b.01", b_l[1]);
#endif

    for (l = 1; l <= (num_levels - 2); l++)
    {
      /* smooth (zero initial x) */
      PFModuleInvokeType(LinearSolverInvoke, smooth_l[l], (x_l[l], b_l[l], 0.0, 1));

      /* compute residual (b - Ax) */
      Copy(b_l[l], temp_vec_l[l]);
      Matvec(-1.0, A_l[l], x_l[l], 1.0, temp_vec_l[l]);

      /* restrict residual */
      MGSemiRestrict(A_l[l], temp_vec_l[l], b_l[l + 1], P_l[l],
                     f_sra_l[l], c_sra_l[l],
                     restrict_compute_pkg_l[l], restrict_comm_pkg_l[l]);
#if 0
      /* for debugging purposes */
      {
        char filename[255];

        sprintf(filename, "b.%02d", l + 1);
        PrintVector(filename, b_l[l + 1]);
      }
#endif
    }

    /*--------------------------------------------------------------------
     * Bottom
     *--------------------------------------------------------------------*/

    /* solve the coarse system */
    PFModuleInvokeType(LinearSolverInvoke, solve, (x_l[l], b_l[l], 1.0e-9, 1));

    /*--------------------------------------------------------------------
     * Up cycle
     *--------------------------------------------------------------------*/

    for (l = (num_levels - 2); l >= 1; l--)
    {
      /* prolong error */
      MGSemiProlong(A_l[l], temp_vec_l[l], x_l[l + 1], P_l[l],
                    f_sra_l[l], c_sra_l[l],
                    prolong_compute_pkg_l[l], prolong_comm_pkg_l[l]);
#if 0
      /* for debugging purposes */
      {
        char filename[255];

        sprintf(filename, "e.%02d", l);
        PrintVector(filename, temp_vec_l[l]);
      }
#endif

      /* update solution (x = x + e) */
      Axpy(1.0, temp_vec_l[l], x_l[l]);

      /* smooth (non-zero initial x) */
      PFModuleInvokeType(LinearSolverInvoke, smooth_l[l], (x_l[l], b_l[l], 0.0, 0));
    }

    /* prolong error */
    MGSemiProlong(A, temp_vec_l[0], x_l[1], P_l[0],
                  f_sra_l[0], c_sra_l[0],
                  prolong_compute_pkg_l[0], prolong_comm_pkg_l[0]);
#if 0
    /* for debugging purposes */
    PrintVector("e.00", temp_vec_l[0]);
#endif

    /* update solution (x = x + e) */
    Axpy(1.0, temp_vec_l[0], x);

    /* smooth (non-zero initial x) */
    PFModuleInvokeType(LinearSolverInvoke, smooth_l[0], (x, b, 0.0, 0));

    /*--------------------------------------------------------------------
     * Test for convergence or max_iter
     *--------------------------------------------------------------------*/

    if (tol > 0.0)
    {
      if (almost_converged)
      {
        Copy(b, temp_vec_l[0]);
        Matvec(-1.0, A, x, 1.0, temp_vec_l[0]);
        r_dot_r = InnerProd(temp_vec_l[0], temp_vec_l[0]);

#if 0
        if (!amps_Rank(amps_CommWorld))
          amps_Printf("Iteration (%d): ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
                      i, sqrt(r_dot_r), (b_dot_b ? sqrt(r_dot_r / b_dot_b) : 0.0));
#endif

        IfLogging(1)
        {
          norm_log[i] = sqrt(r_dot_r);
          rel_norm_log[i] = b_dot_b ? sqrt(r_dot_r / b_dot_b) : 0.0;
        }

        if (r_dot_r < eps)
          break;
      }
    }

    if ((i + 1) > max_iter)
      break;

    /* smooth (non-zero initial x) */
    PFModuleInvokeType(LinearSolverInvoke, smooth_l[0], (x, b, 0.0, 0));
  }
  POP_NVTX

  if (tol > 0.0)
  {
    if (!amps_Rank(amps_CommWorld))
      amps_Printf("Iterations = %d, ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
                  i, sqrt(r_dot_r), (b_dot_b ? sqrt(r_dot_r / b_dot_b) : 0.0));
  }

  /*-----------------------------------------------------------------------
   * Free temp vectors
   *-----------------------------------------------------------------------*/

  FreeVector(temp_vec_l[0]);
  for (l = 0; l < (num_levels - 1); l++)
  {
    FreeVector(x_l[l + 1]);
    FreeVector(b_l[l + 1]);
    FreeVector(temp_vec_l[l + 1]);

    FreeCommPkg(restrict_comm_pkg_l[l]);
    FreeCommPkg(prolong_comm_pkg_l[l]);
  }

  tfree(temp_vec_l);
  tfree(b_l);
  tfree(x_l);

  tfree(prolong_comm_pkg_l);
  tfree(restrict_comm_pkg_l);

  /*-----------------------------------------------------------------------
   * End timing
   *-----------------------------------------------------------------------*/

  IncFLOPCount(2);
  EndTiming(public_xtra->time_index);

  /*-----------------------------------------------------------------------
   * Print log.
   *-----------------------------------------------------------------------*/

  if (tol > 0.0)
  {
    IfLogging(1)
    {
      FILE  *log_file;
      int j;

      log_file = OpenLogFile("MGSemi");

      fprintf(log_file, "Iters       ||r||_2    ||r||_2/||b||_2\n");
      fprintf(log_file, "-----    ------------    ------------\n");

      for (j = 0; j <= i; j++)
      {
        fprintf(log_file, "% 5d    %e    %e\n",
                j, norm_log[j], rel_norm_log[j]);
      }

      CloseLogFile(log_file);

      tfree(norm_log);
      tfree(rel_norm_log);
    }
  }
  POP_NVTX
}


/*--------------------------------------------------------------------------
 * SetupCoarseOps
 *--------------------------------------------------------------------------*/

void              SetupCoarseOps(
                                 Matrix **        A_l,
                                 Matrix **        P_l,
                                 int              num_levels,
                                 SubregionArray **f_sra_l,
                                 SubregionArray **c_sra_l)
{
  SubregionArray *subregion_array;

  Subregion      *subregion;

  Submatrix      *P_sub;
  Submatrix      *A_sub;
  Submatrix      *Ac_sub;

  double               *p1, *p2;
  double         *a0, *a1, *a2, *a3, *a4, *a5, *a6;
  double         *ac0, *ac1, *ac2, *ac3, *ac4, *ac5, *ac6;

  Stencil        *P_stencil, *A_stencil;
  StencilElt     *P_ss, *A_ss;
  int P_sz, A_sz;
  int s_num[7];

  int nx, ny, nz;
  int nx_A, ny_A, nz_A;
  int nx_Ac, ny_Ac, nz_Ac;
  int nx_P, ny_P, nz_P;

  int ii, jj, kk;
  int ix, iy, iz;
  int sx, sy, sz;

  int iP, iP1, dP12 = 0;
  int iA, dA12 = 0;
  int iAc;

  int l, i, j, k;


  /*-----------------------------------------------------------------------
   * Set fine grid conductivity, current solution, and finest matrix.
   *-----------------------------------------------------------------------*/

  for (l = 0; l <= (num_levels - 2); l++)
  {
    /*--------------------------------------------------------------
     * Align prolongation stencil with matrix stencil
     *--------------------------------------------------------------*/

    P_stencil = MatrixStencil(P_l[l]);
    A_stencil = MatrixStencil(A_l[l]);
    P_ss = StencilShape(P_stencil);
    A_ss = StencilShape(A_stencil);
    P_sz = StencilSize(P_stencil);
    A_sz = StencilSize(A_stencil);
    for (j = 0; j < A_sz; j++)
      s_num[j] = j;
    for (j = 1; j < A_sz; j++)
    {
      for (k = 0; k < P_sz; k++)
      {
        if ((A_ss[j][0] == P_ss[k][0]) &&
            (A_ss[j][1] == P_ss[k][1]) &&
            (A_ss[j][2] == P_ss[k][2]))
        {
          s_num[j] = s_num[k + 1];
          s_num[k + 1] = j;
          break;
        }
      }
    }

    /*--------------------------------------------------------------------
     * Compute prolongation matrix
     *--------------------------------------------------------------------*/

    subregion_array = f_sra_l[l];

    ForSubregionI(i, subregion_array)
    {
      subregion = SubregionArraySubregion(subregion_array, i);

      nx = SubregionNX(subregion);
      ny = SubregionNY(subregion);
      nz = SubregionNZ(subregion);

      if (nx && ny && nz)
      {
        ix = SubregionIX(subregion);
        iy = SubregionIY(subregion);
        iz = SubregionIZ(subregion);

        sx = SubregionSX(subregion);
        sy = SubregionSY(subregion);
        sz = SubregionSZ(subregion);

        P_sub = MatrixSubmatrix(P_l[l], i);
        A_sub = MatrixSubmatrix(A_l[l], i);

        nx_P = SubmatrixNX(P_sub);
        ny_P = SubmatrixNY(P_sub);
        nz_P = SubmatrixNZ(P_sub);

        nx_A = SubmatrixNX(A_sub);
        ny_A = SubmatrixNY(A_sub);
        nz_A = SubmatrixNZ(A_sub);

        p1 = SubmatrixStencilData(P_sub, 0);
        p2 = SubmatrixStencilData(P_sub, 1);

        a0 = SubmatrixStencilData(A_sub, s_num[0]);
        a1 = SubmatrixStencilData(A_sub, s_num[1]);
        a2 = SubmatrixStencilData(A_sub, s_num[2]);
        a3 = SubmatrixStencilData(A_sub, s_num[3]);
        a4 = SubmatrixStencilData(A_sub, s_num[4]);
        a5 = SubmatrixStencilData(A_sub, s_num[5]);
        a6 = SubmatrixStencilData(A_sub, s_num[6]);

        iP = SubmatrixEltIndex(P_sub, ix, iy, iz);
        iA = SubmatrixEltIndex(A_sub, ix, iy, iz);

        BoxLoopI2(ii, jj, kk, ix, iy, iz, nx, ny, nz,
                  iP, nx_P, ny_P, nz_P, 1, 1, 1,
                  iA, nx_A, ny_A, nz_A, sx, sy, sz,
        {
          double ap0 = a0[iA] + a3[iA] + a4[iA] + a5[iA] + a6[iA];

          if (ap0)
          {
            p1[iP] = -a1[iA] / ap0;
            p2[iP] = -a2[iA] / ap0;
          }
          else
          {
            p1[iP] = 0.0;
            p2[iP] = 0.0;
          }
        });
      }
    }

    /*--------------------------------------------------------------------
     * Update prolongation matrix boundaries
     *--------------------------------------------------------------------*/

    FinalizeMatrixUpdate(InitMatrixUpdate(P_l[l]));

    /*--------------------------------------------------------------------
     * Compute coarse coefficient matrix
     *--------------------------------------------------------------------*/

    subregion_array = c_sra_l[l];

    ForSubregionI(i, subregion_array)
    {
      subregion = SubregionArraySubregion(subregion_array, i);

      nx = SubregionNX(subregion);
      ny = SubregionNY(subregion);
      nz = SubregionNZ(subregion);

      if (nx && ny && nz)
      {
        ix = SubregionIX(subregion);
        iy = SubregionIY(subregion);
        iz = SubregionIZ(subregion);

        sx = SubregionSX(subregion);
        sy = SubregionSY(subregion);
        sz = SubregionSZ(subregion);

        P_sub = MatrixSubmatrix(P_l[l], i);
        A_sub = MatrixSubmatrix(A_l[l], i);
        Ac_sub = MatrixSubmatrix(A_l[l + 1], i);

        nx_P = SubmatrixNX(P_sub);
        ny_P = SubmatrixNY(P_sub);
        nz_P = SubmatrixNZ(P_sub);

        nx_A = SubmatrixNX(A_sub);
        ny_A = SubmatrixNY(A_sub);
        nz_A = SubmatrixNZ(A_sub);

        nx_Ac = SubmatrixNX(Ac_sub);
        ny_Ac = SubmatrixNY(Ac_sub);
        nz_Ac = SubmatrixNZ(Ac_sub);

        p1 = SubmatrixStencilData(P_sub, 0);
        p2 = SubmatrixStencilData(P_sub, 1);

        a0 = SubmatrixStencilData(A_sub, s_num[0]);
        a1 = SubmatrixStencilData(A_sub, s_num[1]);
        a2 = SubmatrixStencilData(A_sub, s_num[2]);
        a3 = SubmatrixStencilData(A_sub, s_num[3]);
        a4 = SubmatrixStencilData(A_sub, s_num[4]);
        a5 = SubmatrixStencilData(A_sub, s_num[5]);
        a6 = SubmatrixStencilData(A_sub, s_num[6]);

        ac0 = SubmatrixStencilData(Ac_sub, s_num[0]);
        ac1 = SubmatrixStencilData(Ac_sub, s_num[1]);
        ac2 = SubmatrixStencilData(Ac_sub, s_num[2]);
        ac3 = SubmatrixStencilData(Ac_sub, s_num[3]);
        ac4 = SubmatrixStencilData(Ac_sub, s_num[4]);
        ac5 = SubmatrixStencilData(Ac_sub, s_num[5]);
        ac6 = SubmatrixStencilData(Ac_sub, s_num[6]);

        iP1 = SubmatrixEltIndex(P_sub,
                                (ix + P_ss[0][0]),
                                (iy + P_ss[0][1]),
                                (iz + P_ss[0][2]));

        iA = SubmatrixEltIndex(A_sub, ix, iy, iz);
        iAc = SubmatrixEltIndex(Ac_sub, ix / sx, iy / sy, iz / sz);

        if (s_num[2] == 2)
        {
          dP12 = 1;
          dA12 = 1;
        }
        else if (s_num[2] == 4)
        {
          dP12 = SubmatrixNX(P_sub);
          dA12 = SubmatrixNX(A_sub);
        }
        else if (s_num[2] == 6)
        {
          dP12 = SubmatrixNX(P_sub) * SubmatrixNY(P_sub);
          dA12 = SubmatrixNX(A_sub) * SubmatrixNY(A_sub);
        }

        BoxLoopI3(ii, jj, kk, ix, iy, iz, nx, ny, nz,
                  iP1, nx_P, ny_P, nz_P, 1, 1, 1,
                  iA, nx_A, ny_A, nz_A, sx, sy, sz,
                  iAc, nx_Ac, ny_Ac, nz_Ac, 1, 1, 1,
        {
          int iP2 = iP1 + dP12;
          int iA1 = iA - dA12;
          int iA2 = iA + dA12;

          ac3[iAc] = a3[iA] + 0.5 * a3[iA1] + 0.5 * a3[iA2];
          ac4[iAc] = a4[iA] + 0.5 * a4[iA1] + 0.5 * a4[iA2];
          ac5[iAc] = a5[iA] + 0.5 * a5[iA1] + 0.5 * a5[iA2];
          ac6[iAc] = a6[iA] + 0.5 * a6[iA1] + 0.5 * a6[iA2];

          ac1[iAc] = a1[iA] * p1[iP1];
          ac2[iAc] = a2[iA] * p2[iP2];

          ac0[iAc] =
            a0[iA] + a3[iA] + a4[iA] + a5[iA] + a6[iA] +
            a1[iA] * p2[iP1] + a2[iA] * p1[iP2];
        });
      }
    }

    /*--------------------------------------------------------------------
     * Update coefficient matrix boundaries
     *--------------------------------------------------------------------*/

    FinalizeMatrixUpdate(InitMatrixUpdate(A_l[l + 1]));

    /*--------------------------------------------------------------------
     * Complete computation of center coefficient
     *--------------------------------------------------------------------*/

    ForSubregionI(i, subregion_array)
    {
      subregion = SubregionArraySubregion(subregion_array, i);

      nx = SubregionNX(subregion);
      ny = SubregionNY(subregion);
      nz = SubregionNZ(subregion);

      if (nx && ny && nz)
      {
        ix = SubregionIX(subregion);
        iy = SubregionIY(subregion);
        iz = SubregionIZ(subregion);

        sx = SubregionSX(subregion);
        sy = SubregionSY(subregion);
        sz = SubregionSZ(subregion);

        Ac_sub = MatrixSubmatrix(A_l[l + 1], i);

        nx_Ac = SubmatrixNX(Ac_sub);
        ny_Ac = SubmatrixNY(Ac_sub);
        nz_Ac = SubmatrixNZ(Ac_sub);

        ac0 = SubmatrixStencilData(Ac_sub, s_num[0]);
        ac3 = SubmatrixStencilData(Ac_sub, s_num[3]);
        ac4 = SubmatrixStencilData(Ac_sub, s_num[4]);
        ac5 = SubmatrixStencilData(Ac_sub, s_num[5]);
        ac6 = SubmatrixStencilData(Ac_sub, s_num[6]);

        iAc = SubmatrixEltIndex(Ac_sub, ix / sx, iy / sy, iz / sz);

        BoxLoopI1(ii, jj, kk, ix, iy, iz, nx, ny, nz,
                  iAc, nx_Ac, ny_Ac, nz_Ac, 1, 1, 1,
        {
          ac0[iAc] -= (ac3[iAc] + ac4[iAc] +
                       ac5[iAc] + ac6[iAc]);
        });
      }
    }
  }

#if 0
  /* for debugging purposes */
  for (l = 0; l < num_levels; l++)
  {
    char filename[255];

    sprintf(filename, "A.%02d", l);
    PrintSortMatrix(filename, A_l[l], FALSE);
  }
  for (l = 0; l < (num_levels - 1); l++)
  {
    char filename[255];

    sprintf(filename, "P.%02d", l);
    PrintSortMatrix(filename, P_l[l], FALSE);
  }
#endif
}


/*--------------------------------------------------------------------------
 * MGSemiInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule     *MGSemiInitInstanceXtra(
                                     Problem *    problem,
                                     Grid *       grid,
                                     ProblemData *problem_data,
                                     Matrix *     A,
                                     double *     temp_data)
{
  PUSH_NVTX("MGSemiInitInstanceXtra",3)

  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;

  int max_levels = (public_xtra->max_levels);
  int min_NX = (public_xtra->min_NX);
  int min_NY = (public_xtra->min_NY);
  int min_NZ = (public_xtra->min_NZ);

  int num_levels;

  Grid           **grid_l;

  SubregionArray **f_sra_l;
  SubregionArray **c_sra_l;

  int             *coarsen_l;

  ComputePkg     **restrict_compute_pkg_l;
  ComputePkg     **prolong_compute_pkg_l;

  Matrix          **A_l;

  Matrix          **P_l;

  SubgridArray    *all_subgrids;
  SubgridArray    *subgrids;

  Subgrid         *subgrid;

  Subregion       *f_subregion;
  Subregion       *c_subregion;

  int c_index = 0;
  int f_index = 1;
  int sx = 0, sy = 0, sz = 0;

  double DX, DY, DZ;
  int NX, NY, NZ;
  double min_spacing;
  int min_spacing_direction;


  int l, i;

  int coarse_op_shape[7][3] = { { 0, 0, 0 },
                                { -1, 0, 0 },
                                { 1, 0, 0 },
                                { 0, -1, 0 },
                                { 0, 1, 0 },
                                { 0, 0, -1 },
                                { 0, 0, 1 } };
  int transfer_x_shape[][3] = { { -1, 0, 0 },
                                { 1, 0, 0 } };
  int transfer_y_shape[][3] = { { 0, -1, 0 },
                                { 0, 1, 0 } };
  int transfer_z_shape[][3] = { { 0, 0, -1 },
                                { 0, 0, 1 } };

  Stencil         *coarse_op_stencil;
  Stencil         *transfer_stencil = NULL;


  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
  {
    /* TODO */
    /* free old data */
    if ((instance_xtra->grid_l) != NULL)
    {
    }

    /*--------------------------------------------------------------------
     * Determine coarsening directions and num_levels
     *--------------------------------------------------------------------*/

    DX = RealSpaceDX(0);
    DY = RealSpaceDY(0);
    DZ = RealSpaceDZ(0);

    NX = IndexSpaceNX(0);
    NY = IndexSpaceNY(0);
    NZ = IndexSpaceNZ(0);

    /* reset min_N* to be the minimum size that can be coarsened */
    min_NX = pfmax(2 * min_NX - 1, 2);
    min_NY = pfmax(2 * min_NY - 1, 2);
    min_NZ = pfmax(2 * min_NZ - 1, 2);

    coarsen_l = talloc(int, (max_levels - 1));

    for (l = 0; l < (max_levels - 1); l++)
    {
      min_spacing = DX + DY + DZ;
      min_spacing_direction = -1;
      if ((NX >= min_NX) && (DX < min_spacing))
      {
        min_spacing = DX;
        min_spacing_direction = 0;
      }
      if ((NY >= min_NY) && (DY < min_spacing))
      {
        min_spacing = DY;
        min_spacing_direction = 1;
      }
      if ((NZ >= min_NZ) && (DZ < min_spacing))
      {
        min_spacing = DZ;
        min_spacing_direction = 2;
      }

      /* if cannot coarsen in any direction, stop */
      if (min_spacing_direction == -1)
        break;

      switch (min_spacing_direction)
      {
        case 0:
          DX *= 2;
          NX = (NX + 1) / 2;
          break;

        case 1:
          DY *= 2;
          NY = (NY + 1) / 2;
          break;

        case 2:
          DZ *= 2;
          NZ = (NZ + 1) / 2;
          break;
      }

      coarsen_l[l] = min_spacing_direction;
    }
    num_levels = l + 1;

    (instance_xtra->num_levels) = num_levels;
    (instance_xtra->coarsen_l) = coarsen_l;

    /*-----------------------------------------------------------------
     * Log coarsening strategy
     *-----------------------------------------------------------------*/

    IfLogging(1)
    {
      FILE  *log_file;

      log_file = OpenLogFile("MGSemi");

      fprintf(log_file, "coarsening direction\n");
      fprintf(log_file, "--------------------\n");
      for (l = 0; l < (num_levels - 1); l++)
      {
        switch (coarsen_l[l])
        {
          case 0:
            fprintf(log_file, "        x\n");
            break;

          case 1:
            fprintf(log_file, "         y\n");
            break;

          case 2:
            fprintf(log_file, "          z\n");
            break;
        }
      }

      CloseLogFile(log_file);
    }

    /*--------------------------------------------------------------------
     * Set up grids, fine and coarse regions, compute packages, and
     * matrix/vector structures.
     *--------------------------------------------------------------------*/

    grid_l = talloc(Grid *, num_levels);
    grid_l[0] = grid;

    c_sra_l = talloc(SubregionArray *, (num_levels - 1));
    f_sra_l = talloc(SubregionArray *, (num_levels - 1));

    restrict_compute_pkg_l = talloc(ComputePkg *, (num_levels - 1));
    prolong_compute_pkg_l = talloc(ComputePkg *, (num_levels - 1));


    A_l = talloc(Matrix *, num_levels);
    P_l = talloc(Matrix *, num_levels - 1);

    for (l = 0; l < (num_levels - 1); l++)
    {
      coarse_op_stencil = NewStencil(coarse_op_shape, 7);

      switch (coarsen_l[l])
      {
        case 0:
          sx = 2;
          sy = 1;
          sz = 1;
          transfer_stencil = NewStencil(transfer_x_shape, 2);
          break;

        case 1:
          sx = 1;
          sy = 2;
          sz = 1;
          transfer_stencil = NewStencil(transfer_y_shape, 2);
          break;

        case 2:
          sx = 1;
          sy = 1;
          sz = 2;
          transfer_stencil = NewStencil(transfer_z_shape, 2);
          break;
      }

      /*-----------------------------------------------------------------
       * Coarsen the `all_subgrids' array.
       *-----------------------------------------------------------------*/

      all_subgrids = NewSubgridArray();

      ForSubgridI(i, GridAllSubgrids(grid_l[l]))
      {
        subgrid = SubgridArraySubgrid(GridAllSubgrids(grid_l[l]), i);

        c_subregion = DuplicateSubregion(subgrid);
        ProjectSubgrid(c_subregion, sx, sy, sz, c_index, c_index, c_index);
        AppendSubgrid(ConvertToSubgrid(c_subregion), all_subgrids);
      }

      /*-----------------------------------------------------------------
       * Create the `subgrids' array
       *-----------------------------------------------------------------*/

      subgrids = GetGridSubgrids(all_subgrids);

      /*-----------------------------------------------------------------
       * Create the coarse grid
       *-----------------------------------------------------------------*/

      grid_l[l + 1] = NewGrid(subgrids, all_subgrids);
      CreateComputePkgs(grid_l[l + 1]);

      /*-----------------------------------------------------------------
       * Create `c_sra' and `f_sra':
       *   `c_sra' is the SubregionArray of the fine grid
       *   corresponding to coarse grid points.
       *   `f_sra' is the SubregionArray of the fine grid
       *   corresponding to grid points which are not coarse points.
       *-----------------------------------------------------------------*/

      f_sra_l[l] = NewSubregionArray();
      c_sra_l[l] = NewSubregionArray();

      ForSubgridI(i, GridSubgrids(grid_l[l]))
      {
        subgrid = SubgridArraySubgrid(GridSubgrids(grid_l[l]), i);

        f_subregion = DuplicateSubregion(subgrid);
        ProjectSubgrid(f_subregion, sx, sy, sz, f_index, f_index, f_index);
        AppendSubregion(f_subregion, f_sra_l[l]);

        c_subregion = DuplicateSubregion(subgrid);
        ProjectSubgrid(c_subregion, sx, sy, sz, c_index, c_index, c_index);
        AppendSubregion(c_subregion, c_sra_l[l]);
      }

      /*-----------------------------------------------------------------
       * Set up compute_pkg_l arrays
       *-----------------------------------------------------------------*/

      restrict_compute_pkg_l[l] =
        NewMGSemiRestrictComputePkg(grid_l[l], transfer_stencil,
                                    sx, sy, sz, c_index, f_index);
      prolong_compute_pkg_l[l] =
        NewMGSemiProlongComputePkg(grid_l[l], transfer_stencil,
                                   sx, sy, sz, c_index, f_index);

      /*-----------------------------------------------------------------
       * Set up A_l, P_l
       *-----------------------------------------------------------------*/

      A_l[l + 1] = NewMatrix(grid_l[l + 1], NULL, coarse_op_stencil,
                             ON, coarse_op_stencil);

      P_l[l] = NewMatrix(grid_l[l], f_sra_l[l], transfer_stencil,
                         OFF, transfer_stencil);
    }

    (instance_xtra->grid_l) = grid_l;

    (instance_xtra->f_sra_l) = f_sra_l;
    (instance_xtra->c_sra_l) = c_sra_l;

    (instance_xtra->restrict_compute_pkg_l) = restrict_compute_pkg_l;
    (instance_xtra->prolong_compute_pkg_l) = prolong_compute_pkg_l;

    (instance_xtra->A_l) = A_l;
    (instance_xtra->P_l) = P_l;
  }

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `A'
   *-----------------------------------------------------------------------*/

  if (A != NULL)
  {
    (instance_xtra->A_l[0]) = A;
    SetupCoarseOps((instance_xtra->A_l),
                   (instance_xtra->P_l),
                   (instance_xtra->num_levels),
                   (instance_xtra->f_sra_l),
                   (instance_xtra->c_sra_l));
  }

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `temp_data'
   *-----------------------------------------------------------------------*/

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;
  }

  /*-----------------------------------------------------------------------
   * Initialize module instances
   *-----------------------------------------------------------------------*/

  /* if null `grid', pass null `grid_l' to other modules */
  if (grid == NULL)
    grid_l = ctalloc(Grid *, (instance_xtra->num_levels));
  else
    grid_l = (instance_xtra->grid_l);

  /* if null `A', pass null `A_l' to other modules */
  if (A == NULL)
    A_l = ctalloc(Matrix *, (instance_xtra->num_levels));
  else
    A_l = (instance_xtra->A_l);

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->smooth_l) =
      talloc(PFModule *, (instance_xtra->num_levels));
    for (l = 0; l < ((instance_xtra->num_levels) - 1); l++)
    {
      (instance_xtra->smooth_l[l]) =
        PFModuleNewInstanceType(LinearSolverInitInstanceXtraInvoke,
                                (public_xtra->smooth),
                                (problem, grid_l[l], problem_data, A_l[l],
                                 temp_data));
    }
    (instance_xtra->solve) =
      PFModuleNewInstanceType(LinearSolverInitInstanceXtraInvoke,
                              (public_xtra->solve),
                              (problem, grid_l[l], problem_data, A_l[l],
                               temp_data));
  }
  else
  {
    for (l = 0; l < ((instance_xtra->num_levels) - 1); l++)
    {
      PFModuleReNewInstanceType(LinearSolverInitInstanceXtraInvoke,
                                (instance_xtra->smooth_l[l]),
                                (problem, grid_l[l], problem_data, A_l[l],
                                 temp_data));
    }
    PFModuleReNewInstanceType(LinearSolverInitInstanceXtraInvoke,
                              (instance_xtra->solve),
                              (problem, grid_l[l], problem_data, A_l[l],
                               temp_data));
  }

  if (grid == NULL)
  {
    tfree(grid_l);
  }
  if (A == NULL)
  {
    tfree(A_l);
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;

  POP_NVTX
  return this_module;
}


/*--------------------------------------------------------------------------
 * MGSemiFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void   MGSemiFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int l;


  if (instance_xtra)
  {
    PFModuleFreeInstance(instance_xtra->solve);
    for (l = 0; l < ((instance_xtra->num_levels) - 1); l++)
      PFModuleFreeInstance(instance_xtra->smooth_l[l]);
    tfree(instance_xtra->smooth_l);

    for (l = 0; l < ((instance_xtra->num_levels) - 1); l++)
      FreeMatrix((instance_xtra->P_l[l]));

    for (l = 1; l < (instance_xtra->num_levels); l++)
      FreeMatrix((instance_xtra->A_l[l]));

    for (l = 0; l < ((instance_xtra->num_levels) - 1); l++)
    {
      FreeComputePkg((instance_xtra->prolong_compute_pkg_l[l]));
      FreeComputePkg((instance_xtra->restrict_compute_pkg_l[l]));
    }

    for (l = 1; l < (instance_xtra->num_levels); l++)
      FreeGrid((instance_xtra->grid_l[l]));

    for (l = 0; l < ((instance_xtra->num_levels) - 1); l++)
    {
      FreeSubregionArray((instance_xtra->c_sra_l[l]));
      FreeSubregionArray((instance_xtra->f_sra_l[l]));
    }

    tfree(instance_xtra->P_l);
    tfree(instance_xtra->A_l);

    tfree(instance_xtra->prolong_compute_pkg_l);
    tfree(instance_xtra->restrict_compute_pkg_l);

    tfree(instance_xtra->coarsen_l);

    tfree(instance_xtra->c_sra_l);
    tfree(instance_xtra->f_sra_l);

    tfree(instance_xtra->grid_l);

    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * MGSemiNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *MGSemiNewPublicXtra(char *name)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  char key[IDB_MAX_KEY_LEN];

  char          *switch_name;
  int switch_value;

  NameArray smoother_na;

  NameArray coarse_solve_na;

  public_xtra = talloc(PublicXtra, 1);

  smoother_na = NA_NewNameArray("RedBlackGSPoint WJacobi");
  sprintf(key, "%s.Smoother", name);
  switch_name = GetStringDefault(key, "RedBlackGSPoint");
  switch_value = NA_NameToIndexExitOnError(smoother_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      public_xtra->smooth = PFModuleNewModuleType(LinearSolverNewPublicXtraInvoke, RedBlackGSPoint, (key));
      break;
    }

    case 1:
    {
      public_xtra->smooth = PFModuleNewModuleType(LinearSolverNewPublicXtraInvoke, WJacobi, (key));
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(smoother_na);

  coarse_solve_na = NA_NewNameArray("CGHS RedBlackGSPoint WJacobi");
  sprintf(key, "%s.CoarseSolve", name);
  switch_name = GetStringDefault(key, "RedBlackGSPoint");
  switch_value = NA_NameToIndexExitOnError(coarse_solve_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      public_xtra->solve = PFModuleNewModuleType(LinearSolverNewPublicXtraInvoke, CGHS, (key));
      break;
    }

    case 1:
    {
      public_xtra->solve = PFModuleNewModuleType(LinearSolverNewPublicXtraInvoke, RedBlackGSPoint, (key));
      break;
    }

    case 2:
    {
      public_xtra->solve = PFModuleNewModuleType(LinearSolverNewPublicXtraInvoke, WJacobi, (key));
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(coarse_solve_na);

  sprintf(key, "%s.MaxIter", name);
  public_xtra->max_iter = GetIntDefault(key, 1);

  sprintf(key, "%s.MaxLevels", name);
  public_xtra->max_levels = GetIntDefault(key, 100);

  sprintf(key, "%s.MaxMinNX", name);
  public_xtra->min_NX = GetIntDefault(key, 1);

  sprintf(key, "%s.MaxMinNY", name);
  public_xtra->min_NY = GetIntDefault(key, 1);

  sprintf(key, "%s.MaxMinNZ", name);
  public_xtra->min_NZ = GetIntDefault(key, 1);

  (public_xtra->time_index) = RegisterTiming("MGSemi");

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * MGSemiFreePublicXtra
 *--------------------------------------------------------------------------*/

void   MGSemiFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    PFModuleFreeModule(public_xtra->smooth);
    PFModuleFreeModule(public_xtra->solve);
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * MGSemiSizeOfTempData
 *--------------------------------------------------------------------------*/

int  MGSemiSizeOfTempData()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int sz = 0;

  int l;


  /* set `sz' to max of each of the called modules */
  for (l = 0; l < ((instance_xtra->num_levels) - 1); l++)
    sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->smooth_l[l]));

  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->solve));

  return sz;
}
