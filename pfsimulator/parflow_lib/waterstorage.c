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

// Calculates the waterstorage in the whole domain!
double WaterStorage(ProblemData *problem_data,
                    Vector *     pressure,    /* Current pressure values */
                    Vector *     saturation)  /* Current saturation values */
{
  Vector      *porosity = ProblemDataPorosity(problem_data);

  // Specific storage
  Vector      *sstorage = ProblemDataSpecificStorage(problem_data);

  Vector      *z_mult = ProblemDataZmult(problem_data);

  Grid        *grid = VectorGrid(pressure);

  GrGeomSolid *gr_domain = ProblemDataGrDomain(problem_data);

  int is;

  double waterstorage = 0.0;

  ForSubgridI(is, GridSubgrids(grid))
  {
    Subgrid * subgrid = GridSubgrid(grid, is);

    Subvector * p_sub = VectorSubvector(pressure, is);
    Subvector * s_sub = VectorSubvector(saturation, is);
    Subvector * po_sub = VectorSubvector(porosity, is);
    Subvector * ss_sub = VectorSubvector(sstorage, is);
    Subvector * z_mult_sub = VectorSubvector(z_mult, is);

    /* RDF: assumes resolutions are the same in all 3 directions */
    int r = SubgridRX(subgrid);

    int ix = SubgridIX(subgrid);
    int iy = SubgridIY(subgrid);
    int iz = SubgridIZ(subgrid);

    int nx = SubgridNX(subgrid);
    int ny = SubgridNY(subgrid);
    int nz = SubgridNZ(subgrid);

    double dx = SubgridDX(subgrid);
    double dy = SubgridDY(subgrid);
    double dz = SubgridDZ(subgrid);

    double vol = dx * dy * dz;

    double * pressure_dat = SubvectorData(p_sub);
    double * saturation_dat = SubvectorData(s_sub);
    double * porosity_dat = SubvectorData(po_sub);
    double * sstorage_dat = SubvectorData(ss_sub);
    double * z_mult_dat = SubvectorData(z_mult_sub);

    int i, j, k;

    GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
    {
      int pressure_index = SubvectorEltIndex(p_sub, i, j, k);
      int saturation_index = SubvectorEltIndex(s_sub, i, j, k);
      int porosity_index = SubvectorEltIndex(po_sub, i, j, k);
      int sstorage_index = SubvectorEltIndex(ss_sub, i, j, k);
      int z_mult_index = SubvectorEltIndex(z_mult_sub, i, j, k);

      waterstorage += z_mult_dat[z_mult_index] * vol * saturation_dat[saturation_index] * porosity_dat[porosity_index];
      waterstorage += z_mult_dat[z_mult_index] * vol * saturation_dat[saturation_index] * porosity_dat[porosity_index] * pressure_dat[pressure_index] *
                      sstorage_dat[sstorage_index];

      // TODO: this is always left out in water_balance.c ComputeGWStorage/ComputeSubsurfaceStorage - so I leaf it out for the moment as well..
      //if (k = 0 && pressure_ptr[pressure_index] > 0)
      //{
      // waterstorage += vol * pressure_ptr[pressure_index];
      //}
    });
  }

  /* get sum of waterstorage of all nodes */
  amps_Invoice invoice;
  invoice = amps_NewInvoice("%d", &waterstorage);
  amps_AllReduce(amps_CommWorld, invoice, amps_Add);
  amps_FreeInvoice(invoice);

  return waterstorage;
}

