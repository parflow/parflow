/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Infinity Norm of two vectors
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * InfinityNorm
 *--------------------------------------------------------------------------*/

double  InfinityNorm(x)
Vector *x;
{
   Grid        	*grid     = VectorGrid(x);
   Subgrid     	*subgrid;
 	       	
   Subvector   	*x_sub;
	       	
   double      	*xp;
	       	
   double      	 infinity_norm, tmp;
	       	
   int         	 ix,   iy,   iz;
   int         	 nx,   ny,   nz;
   int         	 nx_v, ny_v, nz_v;
	       	
   int         	 i_s, i, j, k, iv;

   amps_Invoice  result_invoice;


   result_invoice = amps_NewInvoice("%d", &infinity_norm);

   infinity_norm = 0.0;

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

      nx_v = SubvectorNX(x_sub);
      ny_v = SubvectorNY(x_sub);
      nz_v = SubvectorNZ(x_sub);

      xp = SubvectorElt(x_sub, ix, iy, iz);
 
      iv = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
		iv, nx_v, ny_v, nz_v, 1, 1, 1,
		{
		   tmp = fabs( xp[iv] );
		   if ( tmp > infinity_norm )
		      infinity_norm = tmp;
		});
   }
   
   amps_AllReduce(amps_CommWorld, result_invoice, amps_Max);

   amps_FreeInvoice(result_invoice);

   return infinity_norm;
}
