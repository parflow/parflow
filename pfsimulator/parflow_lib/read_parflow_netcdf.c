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
 * Routines to write a Vector to a file in full or scattered form.
 *
 *****************************************************************************/

#include "parflow_netcdf.h"

void ReadNetCDF_Subvector(int varid, Subvector *subvector, Subgrid *subgrid,
                          int timestep) {
  int ix = SubgridIX(subgrid);
  int iy = SubgridIY(subgrid);
  int iz = SubgridIZ(subgrid);

  int nx = SubgridNX(subgrid);
  int ny = SubgridNY(subgrid);
  int nz = SubgridNZ(subgrid);

  int nx_v = SubvectorNX(subvector);
  int ny_v = SubvectorNY(subvector);
  int nz_v = SubvectorNZ(subvector);

  int i, j, k, d, ai;

  size_t startp[4];
  size_t countp[4];

  double *data;
  double nc_data[nx * ny * nz];

  (void)subgrid;

  startp[0] = timestep, countp[0] = 1;
  startp[1] = iz, countp[1] = nz;
  startp[2] = iy, countp[2] = ny;
  startp[3] = ix, countp[3] = nx;

#ifdef HAVE_NETCDF
  nc_get_vara_double(ncid, varid, startp, countp, &nc_data[0]);

  data = SubvectorElt(subvector, ix, iy, iz);

  ai = 0, d = 0;
  BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz, ai, nx_v, ny_v, nz_v, 1, 1, 1, {
    data[ai] = nc_data[d];
    d++;
  });
#endif
}

void ReadNetCDF(char *varname, Vector *v, int timestep) {
  Grid *grid = VectorGrid(v);
  SubgridArray *subgrids = GridSubgrids(grid);
  Subgrid *subgrid;
  Subvector *subvector;

  int varid, g;

  BeginTiming(PFBTimingIndex);

#ifdef HAVE_NETCDF
  nc_inq_varid(ncid, varname, &varid);

  ForSubgridI(g, subgrids) {
    subgrid = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(v, g);
    ReadNetCDF_Subvector(varid, subvector, subgrid, timestep);
  }
#endif

  EndTiming(PFBTimingIndex);
}