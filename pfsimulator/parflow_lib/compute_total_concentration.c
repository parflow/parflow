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
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * This routine returns the total amount of a substance within the domain.
 * We apply a zero condition for the substance on cells in the first layer
 * around the boundary.  These cells do not have correct values due to the
 * false
 *--------------------------------------------------------------------------*/

double       ComputeTotalConcen(
                                GrGeomSolid *gr_domain,
                                Grid *       grid,
                                Vector *     substance,
                                Vector *     porsat_inv)
{
  Subgrid        *subgrid;
  double cell_volume, field_sum;
  double dx, dy, dz;

  Subvector      *s_sub, *ps_sub;

  int i, j, k, r, ix, iy, iz, nx, ny, nz, is, ips, i_ps;
  int            *fdir;
  double         *data, *porsat_data;
  amps_Invoice result_invoice;

  field_sum = 0.0;
  if (porsat_inv == NULL)
  {
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
  }
  else
  {
    /* Reactive-transport path (react_trans): concentrations are per pore
     * volume, so weight each cell by porosity*saturation = 1/porsat_inv.
     * No surface zeroing here; chemistry consumes the field afterwards. */
    ForSubgridI(is, GridSubgrids(grid))
    {
      subgrid = GridSubgrid(grid, is);

      s_sub = VectorSubvector(substance, is);
      ps_sub = VectorSubvector(porsat_inv, is);

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
      porsat_data = SubvectorData(ps_sub);

      cell_volume = dx * dy * dz;

      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {
        ips = SubvectorEltIndex(s_sub, i, j, k);
        i_ps = SubvectorEltIndex(ps_sub, i, j, k);
        field_sum += data[ips] * cell_volume / porsat_data[i_ps];
      });
    }
  }

  result_invoice = amps_NewInvoice("%d", &field_sum);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
  amps_FreeInvoice(result_invoice);

  return(field_sum);
}
