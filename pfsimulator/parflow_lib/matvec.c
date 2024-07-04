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
* Matrix-vector multply routine.
*
*****************************************************************************/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Matvec
 *--------------------------------------------------------------------------*/

void            Matvec(
                       double  alpha,
                       Matrix *A,
                       Vector *x,
                       double  beta,
                       Vector *y)
{
  VectorUpdateCommHandle *handle = NULL;

  Grid           *grid = MatrixGrid(A);
  Subgrid        *subgrid;

  SubregionArray *subregion_array;
  Subregion      *subregion;

  ComputePkg     *compute_pkg;

  Region         *compute_reg = NULL;

  Subvector      *y_sub = NULL;
  Subvector      *x_sub = NULL;
  Submatrix      *A_sub = NULL;

  Stencil        *stencil;
  int stencil_size;
  StencilElt     *s;

  int compute_i, sg, sra, sr, si, i, j, k;

  double temp;

  double         *ap;
  double         *xp;
  double         *yp;

  int vi, mi;

  int ix, iy, iz;
  int nx, ny, nz;
  int sx, sy, sz;

  int nx_v = 0, ny_v = 0, nz_v = 0;
  int nx_m = 0, ny_m = 0, nz_m = 0;

  /*-----------------------------------------------------------------------
   * Begin timing
   *-----------------------------------------------------------------------*/


  BeginTiming(MatvecTimingIndex);

#ifdef VECTOR_UPDATE_TIMING
  EventTiming[NumEvents][MatvecStart] = amps_Clock();
#endif

  /*-----------------------------------------------------------------------
   * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
   *-----------------------------------------------------------------------*/

  if (alpha == 0.0)
  {
    ForSubgridI(sg, GridSubgrids(grid))
    {
      subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      if (nx && ny && nz)
      {
        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        y_sub = VectorSubvector(y, sg);

        nx_v = SubvectorNX(y_sub);
        ny_v = SubvectorNY(y_sub);
        nz_v = SubvectorNZ(y_sub);

        yp = SubvectorElt(y_sub, ix, iy, iz);

        vi = 0;
        BoxLoopI1(i, j, k,
                  ix, iy, iz, nx, ny, nz,
                  vi, nx_v, ny_v, nz_v, 1, 1, 1,
        {
          yp[vi] *= beta;
        });
      }
    }

    IncFLOPCount(VectorSize(x));
    EndTiming(MatvecTimingIndex);

    return;
  }

  /*-----------------------------------------------------------------------
   * Do (alpha != 0.0) computation
   *-----------------------------------------------------------------------*/

  compute_pkg = GridComputePkg(grid, VectorUpdateAll);

  for (compute_i = 0; compute_i < 2; compute_i++)
  {
    switch (compute_i)
    {
      case 0:

#ifndef NO_VECTOR_UPDATE
#ifdef VECTOR_UPDATE_TIMING
        BeginTiming(VectorUpdateTimingIndex);
        EventTiming[NumEvents][InitStart] = amps_Clock();
#endif
        handle = InitVectorUpdate(x, VectorUpdateAll);

#ifdef VECTOR_UPDATE_TIMING
        EventTiming[NumEvents][InitEnd] = amps_Clock();
        EndTiming(VectorUpdateTimingIndex);
#endif
#endif

        compute_reg = ComputePkgIndRegion(compute_pkg);

        /*-----------------------------------------------------------------
         * initialize y= (beta/alpha)*y
         *-----------------------------------------------------------------*/

        ForSubgridI(sg, GridSubgrids(grid))
        {
          subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          if (nx && ny && nz)
          {
            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            y_sub = VectorSubvector(y, sg);

            nx_v = SubvectorNX(y_sub);
            ny_v = SubvectorNY(y_sub);
            nz_v = SubvectorNZ(y_sub);

            temp = beta / alpha;

            if (temp != 1.0)
            {
              yp = SubvectorElt(y_sub, ix, iy, iz);

              vi = 0;
              if (temp == 0.0)
              {
                BoxLoopI1(i, j, k,
                          ix, iy, iz, nx, ny, nz,
                          vi, nx_v, ny_v, nz_v, 1, 1, 1,
              {
                yp[vi] = 0.0;
              });
              }
              else
              {
                BoxLoopI1(i, j, k,
                          ix, iy, iz, nx, ny, nz,
                          vi, nx_v, ny_v, nz_v, 1, 1, 1,
              {
                yp[vi] *= temp;
              });
              }
            }
          }
        }

        break;

      case 1:

#ifndef NO_VECTOR_UPDATE
#ifdef VECTOR_UPDATE_TIMING
        BeginTiming(VectorUpdateTimingIndex);
        EventTiming[NumEvents][FinalizeStart] = amps_Clock();
#endif
        FinalizeVectorUpdate(handle);

#ifdef VECTOR_UPDATE_TIMING
        EventTiming[NumEvents][FinalizeEnd] = amps_Clock();
        EndTiming(VectorUpdateTimingIndex);
#endif
#endif

        compute_reg = ComputePkgDepRegion(compute_pkg);
        break;
    }

    ForSubregionArrayI(sra, compute_reg)
    {
      subregion_array = RegionSubregionArray(compute_reg, sra);

      if (SubregionArraySize(subregion_array))
      {
        y_sub = VectorSubvector(y, sra);
        x_sub = VectorSubvector(x, sra);

        A_sub = MatrixSubmatrix(A, sra);

        nx_v = SubvectorNX(y_sub);
        ny_v = SubvectorNY(y_sub);
        nz_v = SubvectorNZ(y_sub);

        nx_m = SubmatrixNX(A_sub);
        ny_m = SubmatrixNY(A_sub);
        nz_m = SubmatrixNZ(A_sub);
      }

      /*-----------------------------------------------------------------
       * y += A*x
       *-----------------------------------------------------------------*/

      ForSubregionI(sr, subregion_array)
      {
        subregion = SubregionArraySubregion(subregion_array, sr);

        ix = SubregionIX(subregion);
        iy = SubregionIY(subregion);
        iz = SubregionIZ(subregion);

        nx = SubregionNX(subregion);
        ny = SubregionNY(subregion);
        nz = SubregionNZ(subregion);

        sx = SubregionSX(subregion);
        sy = SubregionSY(subregion);
        sz = SubregionSZ(subregion);

        stencil = MatrixStencil(A);
        stencil_size = StencilSize(stencil);
        s = StencilShape(stencil);

        yp = SubvectorElt(y_sub, ix, iy, iz);

        for (si = 0; si < stencil_size; si++)
        {
          xp = SubvectorElt(x_sub,
                            (ix + s[si][0]),
                            (iy + s[si][1]),
                            (iz + s[si][2]));
          ap = SubmatrixElt(A_sub, si, ix, iy, iz);

          vi = 0; mi = 0;
          BoxLoopI2(i, j, k,
                    ix, iy, iz, nx, ny, nz,
                    vi, nx_v, ny_v, nz_v, sx, sy, sz,
                    mi, nx_m, ny_m, nz_m, 1, 1, 1,
          {
            yp[vi] += ap[mi] * xp[vi];
          });
        }

        if (alpha != 1.0)
        {
          yp = SubvectorElt(y_sub, ix, iy, iz);

          vi = 0;
          BoxLoopI1(i, j, k,
                    ix, iy, iz, nx, ny, nz,
                    vi, nx_v, ny_v, nz_v, 1, 1, 1,
          {
            yp[vi] *= alpha;
          });
        }
      }
    }
  }

  /*-----------------------------------------------------------------------
   * End timing
   *-----------------------------------------------------------------------*/

  IncFLOPCount(2 * (MatrixSize(A) + VectorSize(x)));
  EndTiming(MatvecTimingIndex);

#ifdef VECTOR_UPDATE_TIMING
  EventTiming[NumEvents++][MatvecEnd] = amps_Clock();
#endif
}


/*--------------------------------------------------------------------------
 * MatvecSubMat - A matvec operation involving the submatrices JA, JC, JE, JF
 *--------------------------------------------------------------------------*/

void            MatvecSubMat(
                             void *  current_state,
                             double  alpha,
                             Matrix *JB,
                             Matrix *JC,
                             Vector *x,
                             double  beta,
                             Vector *y
                             )
{
  ProblemData *problem_data = StateProblemData(((State*)current_state));

  VectorUpdateCommHandle *handle = NULL;

  Grid           *grid = MatrixGrid(JB);
  Subgrid        *subgrid;

  SubregionArray *subregion_array;
  Subregion      *subregion;

  ComputePkg     *compute_pkg;

  Region         *compute_reg = NULL;

  Vector      *top = ProblemDataIndexOfDomainTop(problem_data);               //DOK
  Subvector      *top_sub = NULL;

  Subvector      *y_sub = NULL;
  Subvector      *x_sub = NULL;
  Submatrix      *JB_sub = NULL;
  Submatrix      *JC_sub = NULL;

  Stencil        *stencil;
  int stencil_size;
  StencilElt     *s;

  int compute_i, sra, sr, si, sg, i, j, k;

  double temp;

  double         *bp, *cp;
  double         *xp;
  double         *yp;

  double         *top_dat = NULL;

  int vi, mi;

  int ix, iy, iz;
  int nx, ny, nz;
  int sx, sy, sz;

  int nx_v = 0, ny_v = 0, nz_v = 0;
  int nx_m = 0, ny_m = 0, nz_m = 0;
  int nx_mc = 0, ny_mc = 0, nz_mc = 0;

  /*-----------------------------------------------------------------------
   * Begin timing
   *-----------------------------------------------------------------------*/


  BeginTiming(MatvecTimingIndex);

#ifdef VECTOR_UPDATE_TIMING
  EventTiming[NumEvents][MatvecStart] = amps_Clock();
#endif

  /*-----------------------------------------------------------------------
   * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
   *-----------------------------------------------------------------------*/

  if (alpha == 0.0)
  {
    ForSubgridI(sg, GridSubgrids(grid))
    {
      subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      if (nx && ny && nz)
      {
        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        y_sub = VectorSubvector(y, sg);

        nx_v = SubvectorNX(y_sub);
        ny_v = SubvectorNY(y_sub);
        nz_v = SubvectorNZ(y_sub);

        yp = SubvectorElt(y_sub, ix, iy, iz);

        vi = 0;
        BoxLoopI1(i, j, k,
                  ix, iy, iz, nx, ny, nz,
                  vi, nx_v, ny_v, nz_v, 1, 1, 1,
        {
          yp[vi] *= beta;
        });
      }
    }

    IncFLOPCount(VectorSize(x));
    EndTiming(MatvecTimingIndex);

    return;
  }

  /*-----------------------------------------------------------------------
   * Do (alpha != 0.0) computation
   *-----------------------------------------------------------------------*/

  compute_pkg = GridComputePkg(grid, VectorUpdateAll);

  for (compute_i = 0; compute_i < 2; compute_i++)
  {
    switch (compute_i)
    {
      case 0:

#ifndef NO_VECTOR_UPDATE
#ifdef VECTOR_UPDATE_TIMING
        BeginTiming(VectorUpdateTimingIndex);
        EventTiming[NumEvents][InitStart] = amps_Clock();
#endif
        handle = InitVectorUpdate(x, VectorUpdateAll);

#ifdef VECTOR_UPDATE_TIMING
        EventTiming[NumEvents][InitEnd] = amps_Clock();
        EndTiming(VectorUpdateTimingIndex);
#endif
#endif

        compute_reg = ComputePkgIndRegion(compute_pkg);
        /*-----------------------------------------------------------------
         * initialize y= (beta/alpha)*y
         *-----------------------------------------------------------------*/

        ForSubgridI(sg, GridSubgrids(grid))
        {
          subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          if (nx && ny && nz)
          {
            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            y_sub = VectorSubvector(y, sg);

            nx_v = SubvectorNX(y_sub);
            ny_v = SubvectorNY(y_sub);
            nz_v = SubvectorNZ(y_sub);

            temp = beta / alpha;

            if (temp != 1.0)
            {
              yp = SubvectorElt(y_sub, ix, iy, iz);

              vi = 0;
              if (temp == 0.0)
              {
                BoxLoopI1(i, j, k,
                          ix, iy, iz, nx, ny, nz,
                          vi, nx_v, ny_v, nz_v, 1, 1, 1,
              {
                yp[vi] = 0.0;
              });
              }
              else
              {
                BoxLoopI1(i, j, k,
                          ix, iy, iz, nx, ny, nz,
                          vi, nx_v, ny_v, nz_v, 1, 1, 1,
              {
                yp[vi] *= temp;
              });
              }
            }
          }
        }

        break;

      case 1:

#ifndef NO_VECTOR_UPDATE
#ifdef VECTOR_UPDATE_TIMING
        BeginTiming(VectorUpdateTimingIndex);
        EventTiming[NumEvents][FinalizeStart] = amps_Clock();
#endif
        FinalizeVectorUpdate(handle);

#ifdef VECTOR_UPDATE_TIMING
        EventTiming[NumEvents][FinalizeEnd] = amps_Clock();
        EndTiming(VectorUpdateTimingIndex);
#endif
#endif

        compute_reg = ComputePkgDepRegion(compute_pkg);
        break;
    }

    /*-----------------------------------------------------------------
     * y += A*x
     *-----------------------------------------------------------------*/
    ForSubregionArrayI(sra, compute_reg)
    {
      subregion_array = RegionSubregionArray(compute_reg, sra);

      if (SubregionArraySize(subregion_array))
      {
        y_sub = VectorSubvector(y, sra);
        x_sub = VectorSubvector(x, sra);

        top_sub = VectorSubvector(top, sra);
        top_dat = SubvectorData(top_sub);

        JB_sub = MatrixSubmatrix(JB, sra);
        JC_sub = MatrixSubmatrix(JC, sra);

        nx_v = SubvectorNX(y_sub);
        ny_v = SubvectorNY(y_sub);
        nz_v = SubvectorNZ(y_sub);

        nx_m = SubmatrixNX(JB_sub);
        ny_m = SubmatrixNY(JB_sub);
        nz_m = SubmatrixNZ(JB_sub);

        nx_mc = SubmatrixNX(JC_sub);
        ny_mc = SubmatrixNY(JC_sub);
        nz_mc = SubmatrixNZ(JC_sub);
      }

/* ------------- First get contribution from JB (includes E and F parts) --------------- */
      ForSubregionI(sr, subregion_array)
      {
        subregion = SubregionArraySubregion(subregion_array, sr);

        ix = SubregionIX(subregion);
        iy = SubregionIY(subregion);
        iz = SubregionIZ(subregion);

        nx = SubregionNX(subregion);
        ny = SubregionNY(subregion);
        nz = SubregionNZ(subregion);

        sx = SubregionSX(subregion);
        sy = SubregionSY(subregion);
        sz = SubregionSZ(subregion);

        stencil = MatrixStencil(JB);
        stencil_size = StencilSize(stencil);
        s = StencilShape(stencil);

        yp = SubvectorElt(y_sub, ix, iy, iz);

        for (si = 0; si < stencil_size; si++)
        {
          xp = SubvectorElt(x_sub,
                            (ix + s[si][0]),
                            (iy + s[si][1]),
                            (iz + s[si][2]));
          bp = SubmatrixElt(JB_sub, si, ix, iy, iz);

          vi = 0; mi = 0;

          BoxLoopI2(i, j, k,
                    ix, iy, iz, nx, ny, nz,
                    vi, nx_v, ny_v, nz_v, sx, sy, sz,
                    mi, nx_m, ny_m, nz_m, 1, 1, 1,
          {
            yp[vi] += bp[mi] * xp[vi];
          });
        }

	/* Now compute matvec contributions from JC */
        yp = SubvectorData(y_sub);
        xp = SubvectorData(x_sub);
        for (si = 0; si < 5; si++)     /* loop over only c,w,e,s,n */
        {
          cp = SubmatrixElt(JC_sub, si, ix, iy, iz);

          vi = 0; mi = 0;

          /* Only JC involved here */
          BoxLoopI2(i, j, k,
                    ix, iy, iz, nx, ny, 1,
                    vi, nx_v, ny_v, nz_v, sx, sy, sz,
                    mi, nx_mc, ny_mc, nz_mc, 1, 1, 1,
          {
            int itop = SubvectorEltIndex(top_sub, i, j, 0);
            int k1 = (int)top_dat[itop];
            /* Since we are using a boxloop, we need to check for top index
             * to update with the surface contributions */
            if (k1 >= 0)
            {
              int y_index = SubvectorEltIndex(y_sub, i, j, k1);
              //itop   = SubvectorEltIndex(top_sub, (i+s[si][0]), (j+s[si][1]), 0);
              k1 = (int)top_dat[itop + s[si][0] + nx_v * s[si][1]];
              if (k1 >= 0)
              {
                int x_index = SubvectorEltIndex(x_sub, (i + s[si][0]), (j + s[si][1]), k1);

                yp[y_index] += cp[mi] * xp[x_index];
              }
            }
          });
        }     /*end si loop */

        /* Update vector y */
        if (alpha != 1.0)
        {
          yp = SubvectorElt(y_sub, ix, iy, iz);

          vi = 0;
          BoxLoopI1(i, j, k,
                    ix, iy, iz, nx, ny, nz,
                    vi, nx_v, ny_v, nz_v, 1, 1, 1,
          {
            yp[vi] *= alpha;
          });
        }
      }
    }
  }
  /*-----------------------------------------------------------------------
   * End timing
   *-----------------------------------------------------------------------*/

  IncFLOPCount(2 * (MatrixSize(JB) + VectorSize(x))); /* This may need some fixing - DOK */
  EndTiming(MatvecTimingIndex);

#ifdef VECTOR_UPDATE_TIMING
  EventTiming[NumEvents++][MatvecEnd] = amps_Clock();
#endif
}

/*--------------------------------------------------------------------------
 * MatvecJacF - Does the matvec for a submatrix JF with a vector x. Since the
 * submatrix JF contains subsurface-to-surface coupling, one can think of this
 * as mapping a (2D) vector corresponding to surface contributions, to a (3D)
 * vector corresponding to subsurface contributions. The result, y, is a 3D vector.
 *--------------------------------------------------------------------------*/
void            MatvecJacF(
                           ProblemData *problem_data,
                           double       alpha,
                           Matrix *     JF,
                           Vector *     x,
                           double       beta,
                           Vector *     y
                           )
{
  VectorUpdateCommHandle *handle = NULL;

  Grid           *grid = MatrixGrid(JF);
  Subgrid        *subgrid;

  SubregionArray *subregion_array;
  Subregion      *subregion;

  ComputePkg     *compute_pkg;

  Region         *compute_reg = NULL;

  Vector      *top = ProblemDataIndexOfDomainTop(problem_data);               //DOK
  Subvector      *top_sub = NULL;

  Subvector      *y_sub = NULL;
  Subvector      *x_sub = NULL;
  Submatrix      *JF_sub = NULL;

  Stencil        *stencil;
  StencilElt     *s;

  int compute_i, sra, sr, si, sg, i, j, k;

  double temp;

  double         *fp, *top_dat;
  double         *xp;
  double         *yp;

  int vi, mi;

  int ix, iy, iz;
  int nx, ny, nz;
  int sx, sy, sz;

  int nx_v = 0, ny_v = 0, nz_v = 0;
  int nx_mf = 0, ny_mf = 0, nz_mf = 0;

  /*-----------------------------------------------------------------------
   * Begin timing
   *-----------------------------------------------------------------------*/

  BeginTiming(MatvecTimingIndex);

#ifdef VECTOR_UPDATE_TIMING
  EventTiming[NumEvents][MatvecStart] = amps_Clock();
#endif

  /*-----------------------------------------------------------------------
   * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
   *-----------------------------------------------------------------------*/

  if (alpha == 0.0)
  {
    ForSubgridI(sg, GridSubgrids(grid))
    {
      subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      if (nx && ny && nz)
      {
        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        y_sub = VectorSubvector(y, sg);

        nx_v = SubvectorNX(y_sub);
        ny_v = SubvectorNY(y_sub);
        nz_v = SubvectorNZ(y_sub);

        yp = SubvectorElt(y_sub, ix, iy, iz);

        vi = 0;
        BoxLoopI1(i, j, k,
                  ix, iy, iz, nx, ny, nz,
                  vi, nx_v, ny_v, nz_v, 1, 1, 1,
        {
          yp[vi] *= beta;
        });
      }
    }

    IncFLOPCount(VectorSize(x));
    EndTiming(MatvecTimingIndex);

    return;
  }

  /*-----------------------------------------------------------------------
   * Do (alpha != 0.0) computation
   *-----------------------------------------------------------------------*/

  compute_pkg = GridComputePkg(grid, VectorUpdateAll);

  for (compute_i = 0; compute_i < 2; compute_i++)
  {
    switch (compute_i)
    {
      case 0:

#ifndef NO_VECTOR_UPDATE
#ifdef VECTOR_UPDATE_TIMING
        BeginTiming(VectorUpdateTimingIndex);
        EventTiming[NumEvents][InitStart] = amps_Clock();
#endif
        handle = InitVectorUpdate(x, VectorUpdateAll);

#ifdef VECTOR_UPDATE_TIMING
        EventTiming[NumEvents][InitEnd] = amps_Clock();
        EndTiming(VectorUpdateTimingIndex);
#endif
#endif

        compute_reg = ComputePkgIndRegion(compute_pkg);
        /*-----------------------------------------------------------------
         * initialize y= (beta/alpha)*y
         *-----------------------------------------------------------------*/

        ForSubgridI(sg, GridSubgrids(grid))
        {
          subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          if (nx && ny && nz)
          {
            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            y_sub = VectorSubvector(y, sg);

            nx_v = SubvectorNX(y_sub);
            ny_v = SubvectorNY(y_sub);
            nz_v = SubvectorNZ(y_sub);

            temp = beta / alpha;

            if (temp != 1.0)
            {
              yp = SubvectorElt(y_sub, ix, iy, iz);

              vi = 0;
              if (temp == 0.0)
              {
                BoxLoopI1(i, j, k,
                          ix, iy, iz, nx, ny, nz,
                          vi, nx_v, ny_v, nz_v, 1, 1, 1,
              {
                yp[vi] = 0.0;
              });
              }
              else
              {
                BoxLoopI1(i, j, k,
                          ix, iy, iz, nx, ny, nz,
                          vi, nx_v, ny_v, nz_v, 1, 1, 1,
              {
                yp[vi] *= temp;
              });
              }
            }
          }
        }

        break;

      case 1:

#ifndef NO_VECTOR_UPDATE
#ifdef VECTOR_UPDATE_TIMING
        BeginTiming(VectorUpdateTimingIndex);
        EventTiming[NumEvents][FinalizeStart] = amps_Clock();
#endif
        FinalizeVectorUpdate(handle);

#ifdef VECTOR_UPDATE_TIMING
        EventTiming[NumEvents][FinalizeEnd] = amps_Clock();
        EndTiming(VectorUpdateTimingIndex);
#endif
#endif

        compute_reg = ComputePkgDepRegion(compute_pkg);
        break;
    }

    /*-----------------------------------------------------------------
     * y += JF*x
     *-----------------------------------------------------------------*/

    ForSubregionArrayI(sra, compute_reg)
    {
      subregion_array = RegionSubregionArray(compute_reg, sra);

      if (SubregionArraySize(subregion_array))
      {
        y_sub = VectorSubvector(y, sra);
        x_sub = VectorSubvector(x, sra);

        top_sub = VectorSubvector(top, sra);
        top_dat = SubvectorData(top_sub);

        JF_sub = MatrixSubmatrix(JF, sra);

        nx_v = SubvectorNX(y_sub);
        ny_v = SubvectorNY(y_sub);
        nz_v = SubvectorNZ(y_sub);

        nx_mf = SubmatrixNX(JF_sub);
        ny_mf = SubmatrixNY(JF_sub);
        nz_mf = SubmatrixNZ(JF_sub);
      }

      ForSubregionI(sr, subregion_array)
      {
        subregion = SubregionArraySubregion(subregion_array, sr);

        ix = SubregionIX(subregion);
        iy = SubregionIY(subregion);
        iz = SubregionIZ(subregion);

        nx = SubregionNX(subregion);
        ny = SubregionNY(subregion);
        nz = SubregionNZ(subregion);

        sx = SubregionSX(subregion);
        sy = SubregionSY(subregion);
        sz = SubregionSZ(subregion);

        stencil = MatrixStencil(JF);
        s = StencilShape(stencil);

        yp = SubvectorData(y_sub);

        for (si = 1; si < 5; si++)     /* first loop over only w,e,s,n stencil nodes */
        {
          xp = SubvectorElt(x_sub,
                            (ix + s[si][0]),
                            (iy + s[si][1]),
                            (iz + s[si][2]));
          fp = SubmatrixElt(JF_sub, si, ix, iy, iz);

          vi = 0;
          mi = 0;

          BoxLoopI2(i, j, k,
                    ix, iy, iz, nx, ny, 1,
                    vi, nx_v, ny_v, nz_v, sx, sy, sz,
                    mi, nx_mf, ny_mf, nz_mf, 1, 1, 1,
          {
            int itop = SubvectorEltIndex(top_sub, (i + s[si][0]), (j + s[si][1]), 0);
            int k1 = (int)top_dat[itop];
            if (k1 >= 0)
            {
              int y_index = SubvectorEltIndex(y_sub, i, j, k1);
              yp[y_index] += fp[mi] * xp[vi];
            }
          });
        }

        /* Now do contribution for upper stencil node.
         * x is assumed to be 2D hence index of upper stencil
         * node corresponds to the value of x a that node.
         * No shifting is necessary
         */
        si = 6;
        xp = SubvectorElt(x_sub, ix, iy, iz);
        fp = SubmatrixElt(JF_sub, si, ix, iy, iz);

        vi = 0; mi = 0;
        BoxLoopI2(i, j, k,
                  ix, iy, iz, nx, ny, 1,
                  vi, nx_v, ny_v, nz_v, sx, sy, sz,
                  mi, nx_mf, ny_mf, nz_mf, 1, 1, 1,
        {
          int itop = SubvectorEltIndex(top_sub, i, j, 0);
          int k1 = (int)top_dat[itop];
          if (k1 >= 0)
          {
            int y_index = SubvectorEltIndex(y_sub, i, j, (k1 - s[si][2]));       /* (i,j,k-1) from top */
            yp[y_index] += fp[mi] * xp[vi];
          }
        });

        /* Update vector y */
        if (alpha != 1.0)
        {
          yp = SubvectorElt(y_sub, ix, iy, iz);

          vi = 0;
          BoxLoopI1(i, j, k,
                    ix, iy, iz, nx, ny, nz,
                    vi, nx_v, ny_v, nz_v, 1, 1, 1,
          {
            yp[vi] *= alpha;
          });
        }
      }
    }
  }
  /*-----------------------------------------------------------------------
   * End timing
   *-----------------------------------------------------------------------*/

  IncFLOPCount(2 * (MatrixSize(JF) + VectorSize(x)));
  EndTiming(MatvecTimingIndex);

#ifdef VECTOR_UPDATE_TIMING
  EventTiming[NumEvents++][MatvecEnd] = amps_Clock();
#endif
}


/*--------------------------------------------------------------------------
 * MatvecJacE - Does the matvec for a submatrix JE with a vector x. Since the
 * submatrix JE contains surface-to-subsurface coupling, one can think of this
 * as mapping a (3D) vector corresponding to subsurface contributions, to a (2D)
 * vector corresponding to surface contributions. The resulting vector, y, is 2D.
 *--------------------------------------------------------------------------*/
void            MatvecJacE(
                           ProblemData *problem_data,
                           double       alpha,
                           Matrix *     JE,
                           Vector *     x,
                           double       beta,
                           Vector *     y
                           )
{
  VectorUpdateCommHandle *handle = NULL;

  Grid           *grid = MatrixGrid(JE);
  Subgrid        *subgrid;

  SubregionArray *subregion_array;
  Subregion      *subregion;

  ComputePkg     *compute_pkg;

  Region         *compute_reg = NULL;

  Vector      *top = ProblemDataIndexOfDomainTop(problem_data);               //DOK
  Subvector      *top_sub = NULL;

  Subvector      *y_sub = NULL;
  Subvector      *x_sub = NULL;
  Submatrix      *JE_sub = NULL;

  Stencil        *stencil;
  StencilElt     *s;

  int compute_i, sra, sr, si, sg, i, j, k;

  double temp;

  double         *ep, *top_dat;
  double         *xp;
  double         *yp;

  int vi, mi;

  int ix, iy, iz;
  int nx, ny, nz;
  int sx, sy, sz;

  int nx_v = 0, ny_v = 0, nz_v = 0;
  int nx_me = 0, ny_me = 0, nz_me = 0;

  /*-----------------------------------------------------------------------
   * Begin timing
   *-----------------------------------------------------------------------*/


  BeginTiming(MatvecTimingIndex);

#ifdef VECTOR_UPDATE_TIMING
  EventTiming[NumEvents][MatvecStart] = amps_Clock();
#endif

  /*-----------------------------------------------------------------------
   * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
   *-----------------------------------------------------------------------*/

  if (alpha == 0.0)
  {
    ForSubgridI(sg, GridSubgrids(grid))
    {
      subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      if (nx && ny && nz)
      {
        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        y_sub = VectorSubvector(y, sg);

        nx_v = SubvectorNX(y_sub);
        ny_v = SubvectorNY(y_sub);
        nz_v = SubvectorNZ(y_sub);

        yp = SubvectorElt(y_sub, ix, iy, iz);

        vi = 0;
        BoxLoopI1(i, j, k,
                  ix, iy, iz, nx, ny, nz,
                  vi, nx_v, ny_v, nz_v, 1, 1, 1,
        {
          yp[vi] *= beta;
        });
      }
    }

    IncFLOPCount(VectorSize(x));
    EndTiming(MatvecTimingIndex);

    return;
  }

  /*-----------------------------------------------------------------------
   * Do (alpha != 0.0) computation
   *-----------------------------------------------------------------------*/

  compute_pkg = GridComputePkg(grid, VectorUpdateAll);

  for (compute_i = 0; compute_i < 2; compute_i++)
  {
    switch (compute_i)
    {
      case 0:

#ifndef NO_VECTOR_UPDATE
#ifdef VECTOR_UPDATE_TIMING
        BeginTiming(VectorUpdateTimingIndex);
        EventTiming[NumEvents][InitStart] = amps_Clock();
#endif
        handle = InitVectorUpdate(x, VectorUpdateAll);

#ifdef VECTOR_UPDATE_TIMING
        EventTiming[NumEvents][InitEnd] = amps_Clock();
        EndTiming(VectorUpdateTimingIndex);
#endif
#endif

        compute_reg = ComputePkgIndRegion(compute_pkg);
        /*-----------------------------------------------------------------
         * initialize y= (beta/alpha)*y
         *-----------------------------------------------------------------*/

        ForSubgridI(sg, GridSubgrids(grid))
        {
          subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          if (nx && ny && nz)
          {
            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            y_sub = VectorSubvector(y, sg);

            nx_v = SubvectorNX(y_sub);
            ny_v = SubvectorNY(y_sub);
            nz_v = SubvectorNZ(y_sub);

            temp = beta / alpha;

            if (temp != 1.0)
            {
              yp = SubvectorElt(y_sub, ix, iy, iz);

              vi = 0;
              if (temp == 0.0)
              {
                BoxLoopI1(i, j, k,
                          ix, iy, iz, nx, ny, nz,
                          vi, nx_v, ny_v, nz_v, 1, 1, 1,
              {
                yp[vi] = 0.0;
              });
              }
              else
              {
                BoxLoopI1(i, j, k,
                          ix, iy, iz, nx, ny, nz,
                          vi, nx_v, ny_v, nz_v, 1, 1, 1,
              {
                yp[vi] *= temp;
              });
              }
            }
          }
        }

        break;

      case 1:

#ifndef NO_VECTOR_UPDATE
#ifdef VECTOR_UPDATE_TIMING
        BeginTiming(VectorUpdateTimingIndex);
        EventTiming[NumEvents][FinalizeStart] = amps_Clock();
#endif
        FinalizeVectorUpdate(handle);

#ifdef VECTOR_UPDATE_TIMING
        EventTiming[NumEvents][FinalizeEnd] = amps_Clock();
        EndTiming(VectorUpdateTimingIndex);
#endif
#endif

        compute_reg = ComputePkgDepRegion(compute_pkg);
        break;
    }

    /*-----------------------------------------------------------------
     * y += JF*x
     *-----------------------------------------------------------------*/

    ForSubregionArrayI(sra, compute_reg)
    {
      subregion_array = RegionSubregionArray(compute_reg, sra);

      if (SubregionArraySize(subregion_array))
      {
        y_sub = VectorSubvector(y, sra);
        x_sub = VectorSubvector(x, sra);

        top_sub = VectorSubvector(top, sra);
        top_dat = SubvectorData(top_sub);

        JE_sub = MatrixSubmatrix(JE, sra);

        nx_v = SubvectorNX(y_sub);
        ny_v = SubvectorNY(y_sub);
        nz_v = SubvectorNZ(y_sub);

        nx_me = SubmatrixNX(JE_sub);
        ny_me = SubmatrixNY(JE_sub);
        nz_me = SubmatrixNZ(JE_sub);
      }

      ForSubregionI(sr, subregion_array)
      {
        subregion = SubregionArraySubregion(subregion_array, sr);

        ix = SubregionIX(subregion);
        iy = SubregionIY(subregion);
        iz = SubregionIZ(subregion);

        nx = SubregionNX(subregion);
        ny = SubregionNY(subregion);
        nz = SubregionNZ(subregion);

        sx = SubregionSX(subregion);
        sy = SubregionSY(subregion);
        sz = SubregionSZ(subregion);

        stencil = MatrixStencil(JE);
        s = StencilShape(stencil);

        yp = SubvectorElt(y_sub, ix, iy, iz);
        xp = SubvectorData(x_sub);

        for (si = 1; si < 5; si++)     /* first loop over only w,e,s,n stencil nodes */
        {
          ep = SubmatrixElt(JE_sub, si, ix, iy, iz);

          vi = 0;
          mi = 0;

          BoxLoopI2(i, j, k,
                    ix, iy, iz, nx, ny, 1,
                    vi, nx_v, ny_v, nz_v, sx, sy, sz,
                    mi, nx_me, ny_me, nz_me, 1, 1, 1,
          {
            int itop = SubvectorEltIndex(top_sub, i, j, 0);
            int k1 = (int)top_dat[itop];
            if (k1 >= 0)
            {
              int x_index = SubvectorEltIndex(x_sub, (i + s[si][0]), (j + s[si][1]), k1);
              yp[vi] += ep[mi] * xp[x_index];
            }
          });
        }

        /* Now do contribution for lower stencil node. */
        si = 5;
        ep = SubmatrixElt(JE_sub, si, ix, iy, iz);

        vi = 0; mi = 0;
        BoxLoopI2(i, j, k,
                  ix, iy, iz, nx, ny, 1,
                  vi, nx_v, ny_v, nz_v, sx, sy, sz,
                  mi, nx_me, ny_me, nz_me, 1, 1, 1,
        {
          int itop = SubvectorEltIndex(top_sub, i, j, 0);
          int k1 = (int)top_dat[itop];
          if (k1 >= 0)
          {
            int x_index = SubvectorEltIndex(x_sub, i, j, (k1 + s[si][2]));          /* (i,j,k-1) from top */
            yp[vi] += ep[mi] * xp[x_index];
          }
        });

        /* Update vector y */
        if (alpha != 1.0)
        {
          yp = SubvectorElt(y_sub, ix, iy, iz);

          vi = 0;
          BoxLoopI1(i, j, k,
                    ix, iy, iz, nx, ny, nz,
                    vi, nx_v, ny_v, nz_v, 1, 1, 1,
          {
            yp[vi] *= alpha;
          });
        }
      }
    }
  }
  /*-----------------------------------------------------------------------
   * End timing
   *-----------------------------------------------------------------------*/

  IncFLOPCount(2 * (MatrixSize(JE) + VectorSize(x)));
  EndTiming(MatvecTimingIndex);

#ifdef VECTOR_UPDATE_TIMING
  EventTiming[NumEvents++][MatvecEnd] = amps_Clock();
#endif
}
