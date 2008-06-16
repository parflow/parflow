/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * The Matrix vector multiplication routine
 *
 *****************************************************************************/

#include "parflow.h"


void     Scale(alpha, y)
double   alpha;
Vector  *y;
{
   Grid       *grid     = VectorGrid(y);
   Subgrid    *subgrid;
 
   Subvector  *y_sub;

   double     *yp;

   int         ix,  iy,  iz;
   int         nx,  ny,  nz;
   int         nx_v, ny_v, nz_v;

   int         i_s, i, j, k, iv;


   ForSubgridI(i_s, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, i_s);
      
      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);
      
      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);
      
      y_sub = VectorSubvector(y, i_s);
      
      nx_v = SubvectorNX(y_sub);
      ny_v = SubvectorNY(y_sub);
      nz_v = SubvectorNZ(y_sub);
      
      yp = SubvectorElt(y_sub, ix, iy, iz);
	 
      iv = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
		iv, nx_v, ny_v, nz_v, 1, 1, 1,
		{
		   yp[iv] *= alpha;
		});
   }

   IncFLOPCount(VectorSize(y));
}
