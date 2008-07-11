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


void     Copy(x, y)
Vector  *x;
Vector  *y;
{
   Grid       *grid     = VectorGrid(x);
   Subgrid    *subgrid;
 
   Subvector  *y_sub;
   Subvector  *x_sub;

   double     *yp, *xp;

   int         ix,   iy,   iz;
   int         nx,   ny,   nz;
   int         nx_x, ny_x, nz_x;
   int         nx_y, ny_y, nz_y;

   int         i_s, i, j, k, i_x, i_y;


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
      y_sub = VectorSubvector(y, i_s);
      
      nx_x = SubvectorNX(x_sub);
      ny_x = SubvectorNY(x_sub);
      nz_x = SubvectorNZ(x_sub);
      
      nx_y = SubvectorNX(y_sub);
      ny_y = SubvectorNY(y_sub);
      nz_y = SubvectorNZ(y_sub);
      
      yp = SubvectorElt(y_sub, ix, iy, iz);
      xp = SubvectorElt(x_sub, ix, iy, iz);
	 
      i_x = 0;
      i_y = 0;
      BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
		i_x, nx_x, ny_x, nz_x, 1, 1, 1,
		i_y, nx_y, ny_y, nz_y, 1, 1, 1,
		{
		   yp[i_y] = xp[i_x];
		});
   }
}
