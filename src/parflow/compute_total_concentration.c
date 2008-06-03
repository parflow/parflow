/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * This routine returns the total amount of a substance within the domain.
 * We apply a zero condition for the substance on cells in the first layer
 * around the boundary.  These cells do not have correct values due to the
 * false 
 *--------------------------------------------------------------------------*/

double       ComputeTotalConcen(gr_domain, grid, substance)
GrGeomSolid *gr_domain;
Grid        *grid;
Vector      *substance;
{
   Subgrid        *subgrid;
   double          cell_volume, field_sum;
   double          dx, dy, dz;

   Subvector      *s_sub;

   int             i, j, k, r, ix, iy, iz, nx, ny, nz, is, ips;
   int            *fdir;
   double         *data;
   amps_Invoice    result_invoice;

   field_sum = 0.0;
   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);

      s_sub = VectorSubvector(substance, is);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      dx = SubgridDX(subgrid);
      dy = SubgridDY(subgrid);
      dz = SubgridDZ(subgrid);

      /* RDF: assume resolution is the same in all 3 directions */
      r = SubgridRX(subgrid);
      
      data = SubvectorData(s_sub);

      cell_volume = dx * dy * dz;

      GrGeomSurfLoop(i, j, k, fdir, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {
	 ips = SubvectorEltIndex(s_sub, i, j, k);
         data[ips] = 0.0;
      });

      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {
	 ips = SubvectorEltIndex(s_sub, i, j, k);
         field_sum += data[ips] * cell_volume;
      });
   }

   result_invoice = amps_NewInvoice("%d", &field_sum);
   amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
   amps_FreeInvoice(result_invoice);

   return (field_sum);
}
