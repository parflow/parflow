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
 * Routines to read a Vector from a distributed file.
 *
 *****************************************************************************/
#include "parflow.h"
#include "parflow_netcdf.h"
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>



void ReadPFNC(char *fileName, Vector *v)
{

  Grid           *grid     = VectorGrid(v);
  SubgridArray   *subgrids = GridSubgrids(grid);
  Subgrid        *subgrid;
  Subvector      *subvector;

  int             num_chars, g;
  int             p, P;

  amps_File       file;

  double          X, Y, Z;
  int             NX, NY, NZ;
  double          DX, DY, DZ;
  int 		   ncID, varID;

  OpenNCFile(fileName, &ncID);
  ForSubgridI(g, subgrids)
  {
    subgrid   = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(v, g);
    ReadNCFile(ncID, varID, subvector,subgrid);
  }
  nc_close(ncID);

}


void OpenNCFile(char *file_name, int *ncID)
{

  char *switch_name;
  char key[IDB_MAX_KEY_LEN];
  char *default_val = "None";

  sprintf(key, "NetCDF.ROMIOhints");
  switch_name = GetStringDefault(key, "None");
  if(strcmp(switch_name, default_val) != 0)
  {
    if (access(switch_name, F_OK|R_OK) == -1)
    {
      InputError("Error: check if the file is present and readable <%s> for key <%s>\n",
	  switch_name, key);
    }
    MPI_Info romio_info;
    FILE *fp;
    MPI_Info_create (&romio_info);
    char line[100], romio_key[100],value[100];
    fp = fopen(switch_name, "r");
    while ( fgets ( line, sizeof line, fp ) != NULL ) /* read a line */
    {	
      sscanf(line,"%s%s", romio_key, value);
      MPI_Info_set(romio_info, romio_key, value);
    }	
    int res = nc_open_par(file_name,NC_MPIIO, amps_CommWorld, romio_info, ncID);
  }
  else
  {
    int res = nc_open_par(file_name,NC_MPIIO, amps_CommWorld, MPI_INFO_NULL, ncID);
  }
}

void ReadNCFile(int ncID, int varID, Subvector *subvector, Subgrid *subgrid)
{
  nc_inq_varid(ncID, "pressure", &varID); 
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
  double *nc_data;
  nc_data = (double *)malloc(sizeof(double)*nx*ny*nz);

  (void)subgrid;

  startp[0] = 0, countp[0] = 1;
  startp[1] = iz, countp[1] = nz;
  startp[2] = iy, countp[2] = ny;
  startp[3] = ix, countp[3] = nx;

  nc_get_vara_double(ncID, varID, startp, countp, nc_data);

  data = SubvectorElt(subvector, ix, iy, iz);

  ai = 0, d = 0;
  BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz, ai, nx_v, ny_v, nz_v, 1, 1, 1, {
      data[ai] = nc_data[d];
      d++;
      });

}
