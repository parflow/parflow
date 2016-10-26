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

/*
      WRITING RAW DATA
*/

void WriteNetCDF_Subvector(int varid, Subvector *subvector, Subgrid *subgrid) {
  int ix = SubgridIX(subgrid);
  int iy = SubgridIY(subgrid);
  int iz = SubgridIZ(subgrid);

  int nx = SubgridNX(subgrid);
  int ny = SubgridNY(subgrid);
  int nz = SubgridNZ(subgrid);

  int nx_v = SubvectorNX(subvector);
  int ny_v = SubvectorNY(subvector);

  size_t startp[dimlen];
  size_t countp[dimlen];

  int i, j, k, d, ai;
  double *data;
  double data_nc[nx * ny * nz];
  double start_t;

#ifdef HAVE_NETCDF
  data = SubvectorElt(subvector, ix, iy, iz);

  // remove the halos from the subvector double pointer
  // TODO try to make the halo removal more efficient
  ai = 0, d = 0;
  BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz, ai, nx_v, ny_v, nz_v, 1, 1, 1, {
    data_nc[d] = data[ai];
    d++;
  });

  // NC_COLLECTIVE and ROMIO hints will manage to do this nicely in parallel
  switch (dimlen) {
    case 2:
    {
    startp[0] = iz, startp[1] = iy, startp[2] = ix;
    countp[0] = nz, countp[1] = ny, countp[2] = nx;

    if ((retval = nc_put_vara_double(ncid, varid, startp, countp, &data_nc[0])))
      ERR(retval, __LINE__);
    break;
    }
    case 3:
    {
    startp[0] = iy, startp[1] = ix;
    countp[0] = ny, countp[1] = nx;

    if ((retval = nc_put_vara_double(ncid, varid, startp, countp, &data_nc[0])))
      ERR(retval, __LINE__);
    break;
    }
    case 4:
    {
    startp[0] = time_cont_offset + time_index - time_index_offset;
    countp[0] = 1;
    startp[1] = iz, startp[2] = iy, startp[3] = ix;
    countp[1] = nz, countp[2] = ny, countp[3] = nx;

    if ((retval = nc_put_vara_double(ncid, varid, startp, countp, &data_nc[0])))
      ERR(retval, __LINE__);
    break;
    }
    default:
    {
    break;
    }
  }
#endif
}

void WriteNetCDF(char *file_prefix, char *file_postfix, Vector *v) {
  Grid *grid = VectorGrid(v);
  SubgridArray *subgrids = GridSubgrids(grid);
  Subgrid *subgrid;
  Subvector *subvector;

  char file_extn[7] = "nc";

  int varid, i, g;
  char *varname, *timestep;
  size_t time_dimlen;
  double dummy_value = 1.0;

  BeginTiming(PFBTimingIndex);

#ifdef HAVE_NETCDF
  dimlen = 3;

  /* Compute number of patches to write */
  int num_subgrids = GridNumSubgrids(grid);
  {
    amps_Invoice invoice = amps_NewInvoice("%i", &num_subgrids);

    amps_AllReduce(amps_CommWorld, invoice, amps_Add);

    amps_FreeInvoice(invoice);
  }

  // process the varname to check for the time dim
  printf("%s\n", file_postfix);
  varname = strtok(file_postfix, ".");

  // this is dirty and should be resolved differently
  // strtok returns NULL if it didn't match anything anymore
  // else you get the current time step
  if (varname != NULL)
    timestep = strtok(NULL, ".");

  // set 3D or 4D
  if (timestep != NULL) {
    dimlen += 1;
  }

  // define the dimid arrays needed for var I/O
  free(dimids);
  dimids = (int *)malloc(dimlen);
  if (strcmp(file_postfix, "slope_x") == 0 ||
      strcmp(file_postfix, "slope_y") == 0 ||
      strcmp(file_postfix, "perm_x") == 0 ||
      strcmp(file_postfix, "perm_y") == 0 ||
      strcmp(file_postfix, "perm_z") == 0 ||
      strcmp(file_postfix, "dz_mult") == 0 ||
      strcmp(file_postfix, "mask") == 0 ||
      strcmp(file_postfix, "mannings") == 0) {
    dimids[0] = y_dimid;
    dimids[1] = x_dimid;

    dimlen = 2;

    // if we are in 3D we need to check if the 3D field was already written
    int test_varid;
    retval = nc_inq_varid(ncid, varname, &test_varid);
    if (retval == NC_NOERR)
      return;
  } else if (strcmp(file_postfix, "overlandsum") == 0 ||
      strcmp(file_postfix, "overland_bc_flux") == 0 ||
      strcmp(file_postfix, "eflx_lh_tot") == 0 ||
      strcmp(file_postfix, "eflx_lwrad_out") == 0 ||
      strcmp(file_postfix, "eflx_soil_grnd") == 0 ||
      strcmp(file_postfix, "qflx_evap_trans") == 0 ||
      strcmp(file_postfix, "qflx_evap_tot") == 0 ||
      strcmp(file_postfix, "qflx_evap_grnd") == 0 ||
      strcmp(file_postfix, "qflx_evap_soi") == 0 ||
      strcmp(file_postfix, "qflx_evap_veg") == 0 ||
      strcmp(file_postfix, "qflx_tran_veg") == 0 ||
      strcmp(file_postfix, "qflx_infl") == 0 ||
      strcmp(file_postfix, "swe_out") == 0 ||
      strcmp(file_postfix, "t_grnd") == 0) {
    dimids[0] = time_dimid;
    dimids[1] = y_dimid;
    dimids[2] = x_dimid;

    dimlen = 3;

    // convert the extracted time step from the varname to integer to process it
    time_index = atoi(timestep);

    // complete the dimension variable data
    // if time dim is not complete yet
    // make comparison to the length of the time dimension +1 because the time
    // dimension is not yet expanded for the current data
    nc_inq_dimlen(ncid, time_dimid, &time_dimlen);
    if (time_dimlen == time_cont_offset + time_index - time_index_offset)
      WriteNetCDF_Timestamp();
  } else if (strcmp(file_postfix, "press") == 0 ||
        strcmp(file_postfix, "satur") == 0 ||
        strcmp(file_postfix, "evaptrans") == 0 ||
        strcmp(file_postfix, "evaptranssum") == 0 ||
        strcmp(file_postfix, "t_soil") == 0) {
    dimids[0] = time_dimid;
    dimids[1] = z_dimid;
    dimids[2] = y_dimid;
    dimids[3] = x_dimid;

    dimlen = 4;

    // convert the extracted time step from the varname to integer to process it
    time_index = atoi(timestep);

    // complete the dimension variable data
    // if time dim is not complete yet
    // make comparison to the length of the time dimension +1 because the time
    // dimension is not yet expanded for the current data
    nc_inq_dimlen(ncid, time_dimid, &time_dimlen);
    if (time_dimlen == time_cont_offset + time_index - time_index_offset)
      WriteNetCDF_Timestamp();
  }
  printf("%d\n", dimlen);

  varid = WriteNetCDF_Variable(varname);

  // collective access optimisation
  if ((retval = nc_var_par_access(ncid, varid, NC_COLLECTIVE)))
    ERR(retval, __LINE__);

  // enter data mode to write stuff
  if ((retval = nc_enddef(ncid)))
    ERR(retval, __LINE__);

  ForSubgridI(g, subgrids) {
    subgrid = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(v, g);

    WriteNetCDF_Subvector(varid, subvector, subgrid);
  }
#endif

  EndTiming(PFBTimingIndex);
}
