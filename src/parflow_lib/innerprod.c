/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Inner Product of two vectors
 *
 *****************************************************************************/

#include "parflow.h"


double   InnerProd(x, y)
Vector  *x;
Vector  *y;
{
   Grid         *grid = VectorGrid(x);
   Subgrid      *subgrid;
 
   Subvector    *y_sub;
   Subvector    *x_sub;

   double       *yp, *xp;

   double        result = 0.0;

   int           ix,   iy,   iz;
   int           nx,   ny,   nz;
   int           nx_v, ny_v, nz_v;
                 
   int           i_s, i, j, k, iv;

   amps_Invoice  result_invoice;


   result_invoice = amps_NewInvoice("%d", &result);
   
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
      x_sub = VectorSubvector(x, i_s);

      nx_v = SubvectorNX(y_sub);
      ny_v = SubvectorNY(y_sub);
      nz_v = SubvectorNZ(y_sub);

      yp = SubvectorElt(y_sub, ix, iy, iz);
      xp = SubvectorElt(x_sub, ix, iy, iz);
 
      iv = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
		iv, nx_v, ny_v, nz_v, 1, 1, 1,
		{
		   result += yp[iv] * xp[iv];
		});
   }
   
   amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);

   amps_FreeInvoice(result_invoice);

   IncFLOPCount(2*VectorSize(x) - 1);
   
   return result;
}
