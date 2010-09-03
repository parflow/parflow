/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER*/
/******************************************************************************
 *
 * Matrix-vector multply routine.
 * 
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Matvec
 *--------------------------------------------------------------------------*/

void            Matvec(
double          alpha,
Matrix         *A,
Vector         *x,
double          beta,
Vector         *y)
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
   int             stencil_size;
   StencilElt     *s;

   int             compute_i, sg, sra, sr, si, i, j, k;

   double          temp;

   double         *ap;
   double         *xp;
   double         *yp;

   int             vi, mi;

   int             ix, iy, iz;
   int             nx, ny, nz;
   int             sx, sy, sz;

   int             nx_v = 0, ny_v = 0, nz_v = 0;
   int             nx_m = 0, ny_m = 0, nz_m = 0;

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
	    BoxLoopI1(i,j,k,
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
      switch(compute_i)
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
			 mi, nx_m, ny_m, nz_m,  1,  1,  1,
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

   IncFLOPCount(2*(MatrixSize(A) + VectorSize(x)));
   EndTiming(MatvecTimingIndex);

#ifdef VECTOR_UPDATE_TIMING
   EventTiming[NumEvents++][MatvecEnd] = amps_Clock();
#endif

}
