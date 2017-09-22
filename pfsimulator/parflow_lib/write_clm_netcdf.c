/*BHEADER**********************************************************************

  This file is part of Parflow. For details, see

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

#include "parflow.h"
#include "parflow_netcdf.h"
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>

void WriteCLMNC(char * file_prefix, char* file_postfix, double t, Vector  *v, int numVarTimeVariant,
			char *varName, int dimensionality)
{
#ifdef PARFLOW_HAVE_NETCDF

  static int numCLMStepsInFile = 0;
  int userSpecSteps = GetInt("NetCDF.CLMNumStepsPerFile");
  static char file_name[255];
  
  varNCData *myVarNCData;
  if( numCLMStepsInFile == userSpecSteps*numVarTimeVariant)
  {
    sprintf(file_name, "%s%s%s%s%s", file_prefix,".CLM",".",file_postfix,".nc");
    CloseCLMNC(ncCLMID);
    isCLM2Ddefined = false;
    isCLM3Ddefined = false;
    isCLMTdefined = false;
    CreateCLMNCFile(file_name);
    NCCLMDefDimensions(v, dimensionality);
    int myVarID=LookUpCLMInventory(varName, &myVarNCData);
    PutCLMDataInNC(myVarID, v, t, myVarNCData, dimensionality);
    numCLMStepsInFile = 1;
  }
  else
  {
    if(numCLMStepsInFile == 0)
    {
      sprintf(file_name, "%s%s%s%s%s", file_prefix,".CLM",".",file_postfix,".nc");
      CreateCLMNCFile(file_name);
      NCCLMDefDimensions(v, dimensionality);
      int myVarID=LookUpCLMInventory(varName, &myVarNCData);
      PutCLMDataInNC(myVarID,v, t, myVarNCData, dimensionality);
      numCLMStepsInFile++;
    }
    else
    {
      numCLMStepsInFile++;
      NCCLMDefDimensions(v, dimensionality);
      int myVarID=LookUpCLMInventory(varName, &myVarNCData);
      PutCLMDataInNC(myVarID,v, t, myVarNCData, dimensionality);
      if( numCLMStepsInFile == userSpecSteps*numVarTimeVariant)
      {
	CloseCLMNC(ncCLMID);
	isCLM2Ddefined = false;
	isCLM3Ddefined = false;
	isCLMTdefined = false;
      }
    }
  }
#else
  amps_Printf("Parflow not compiled with NetCDF, can't create NetCDF file\n");
#endif

}

void CreateCLMNCFile(char *file_name)
{
#ifdef PARFLOW_HAVE_NETCDF
  char *switch_name;
  char key[IDB_MAX_KEY_LEN];
  char *default_val = "None";
  int old_fill_mode;

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
    int res = nc_create_par(file_name,NC_NETCDF4|NC_MPIIO, amps_CommWorld, romio_info, &ncCLMID);
  }
  else
  {
    int res = nc_create_par(file_name,NC_NETCDF4|NC_MPIIO, amps_CommWorld, MPI_INFO_NULL, &ncCLMID);
  }
#else
  amps_Printf("Parflow not compiled with NetCDF, can't create NetCDF file\n");
#endif
}

void NCCLMDefDimensions(Vector *v, int dimensionality)
{
#ifdef PARFLOW_HAVE_NETCDF
  if (dimensionality == 1 && !isCLMTdefined)
  {
    int res = nc_def_dim(ncCLMID, "time",NC_UNLIMITED,&timCLMID);
    isCLMTdefined = true;
  }

  if (dimensionality == 2 && !isCLM2Ddefined)
  {
    Grid           *grid     = VectorGrid(v);
    SubgridArray   *subgrids = GridSubgrids(grid);
    Subgrid        *subgrid;
    Subvector      *subvector;

    int nX = SubgridNX(GridBackground(grid));
    int nY = SubgridNY(GridBackground(grid));
    int nZ = SubgridNZ(GridBackground(grid));

    int res = nc_def_dim(ncCLMID, "x", nX, &xCLMID);
    res = nc_def_dim(ncCLMID, "y", nY, &yCLMID);
    isCLM2Ddefined = true;  
  }

  if (dimensionality == 3 && !isCLM3Ddefined)
  {
    Grid           *grid     = VectorGrid(v);
    SubgridArray   *subgrids = GridSubgrids(grid);
    Subgrid        *subgrid;
    Subvector      *subvector;

    int nX = SubgridNX(GridBackground(grid));
    int nY = SubgridNY(GridBackground(grid));
    int nZ = SubgridNZ(GridBackground(grid));

    int res = nc_inq_dimid(ncCLMID, "x", &xCLMID);
    if (res != NC_NOERR) nc_def_dim(ncCLMID, "x", nX, &xCLMID);

    res = nc_inq_dimid(ncCLMID, "y", &yCLMID);
    if (res != NC_NOERR) nc_def_dim(ncCLMID, "y", nY, &yCLMID);

    res = nc_def_dim(ncCLMID, "z", nZ, &zCLMID);


    isCLM3Ddefined = true;
  }
#endif
}

int LookUpCLMInventory(char * varName, varNCData **myVarNCData)
{
#ifdef PARFLOW_HAVE_NETCDF
  if (strcmp(varName,"time")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 1;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
     int timCLMVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&timCLMVarID);
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&timCLMVarID);
      if (res != NC_NOERR) printf("Something went wrong in definition %d\n", res);
    }
    return timCLMVarID;
  }

  if (strcmp(varName,"t_soil")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = zCLMID;
    (*myVarNCData)->dimIDs[2] = yCLMID;
    (*myVarNCData)->dimIDs[3] = xCLMID;
     int tsoilCLMVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&tsoilCLMVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkZ");
	chunksize[2] = GetInt("NetCDF.ChunkY");
	chunksize[3] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, tsoilCLMVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&tsoilCLMVarID);
    }
    return tsoilCLMVarID;
  }

  if (strcmp(varName,"eflx_lh_tot")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = yCLMID;
    (*myVarNCData)->dimIDs[2] = xCLMID;
    int lhTotVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&lhTotVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkY");
	chunksize[2] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, lhTotVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&lhTotVarID);
    }
    return lhTotVarID;
  }
  
  if (strcmp(varName,"eflx_lwrad_out")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = yCLMID;
    (*myVarNCData)->dimIDs[2] = xCLMID;
    int lwradVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&lwradVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkY");
	chunksize[2] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, lwradVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&lwradVarID);
    }
    return lwradVarID;
  }
  
  if (strcmp(varName,"eflx_sh_tot")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = yCLMID;
    (*myVarNCData)->dimIDs[2] = xCLMID;
    int shTotVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&shTotVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkY");
	chunksize[2] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, shTotVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&shTotVarID);
    }
    return shTotVarID;
  }
  
  if (strcmp(varName,"eflx_soil_grnd")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = yCLMID;
    (*myVarNCData)->dimIDs[2] = xCLMID;
    int soilGrndVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&soilGrndVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkY");
	chunksize[2] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, soilGrndVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&soilGrndVarID);
    }
    return soilGrndVarID;
  }
  
  if (strcmp(varName,"qflx_evap_tot")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = yCLMID;
    (*myVarNCData)->dimIDs[2] = xCLMID;
    int qEvapTotVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&qEvapTotVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkY");
	chunksize[2] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, qEvapTotVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&qEvapTotVarID);
    }
    return qEvapTotVarID;
  }
  
  if (strcmp(varName,"qflx_evap_grnd")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = yCLMID;
    (*myVarNCData)->dimIDs[2] = xCLMID;
    int qEvapGrndVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&qEvapGrndVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkY");
	chunksize[2] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, qEvapGrndVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&qEvapGrndVarID);
    }
    return qEvapGrndVarID;
  }

  if (strcmp(varName,"qflx_evap_soi")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = yCLMID;
    (*myVarNCData)->dimIDs[2] = xCLMID;
    int qEvapSoiVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&qEvapSoiVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkY");
	chunksize[2] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, qEvapSoiVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&qEvapSoiVarID);
    }
    return qEvapSoiVarID;
  }
  
  if (strcmp(varName,"qflx_evap_veg")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = yCLMID;
    (*myVarNCData)->dimIDs[2] = xCLMID;
    int qEvapVegVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&qEvapVegVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkY");
	chunksize[2] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, qEvapVegVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&qEvapVegVarID);
    }
    return qEvapVegVarID;
  }
  
  if (strcmp(varName,"qflx_tran_veg")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = yCLMID;
    (*myVarNCData)->dimIDs[2] = xCLMID;
    int qTranVegVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&qTranVegVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkY");
	chunksize[2] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, qTranVegVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&qTranVegVarID);
    }
    return qTranVegVarID;
  }
  
  if (strcmp(varName,"qflx_infl")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = yCLMID;
    (*myVarNCData)->dimIDs[2] = xCLMID;
    int qInflVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&qInflVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkY");
	chunksize[2] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, qInflVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&qInflVarID);
    }
    return qInflVarID;
  }
  
  if (strcmp(varName,"swe_out")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = yCLMID;
    (*myVarNCData)->dimIDs[2] = xCLMID;
    int sweVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&sweVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkY");
	chunksize[2] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, sweVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&sweVarID);
    }
    return sweVarID;
  }
  
  if (strcmp(varName,"t_grnd")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = yCLMID;
    (*myVarNCData)->dimIDs[2] = xCLMID;
    int t_grndVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&t_grndVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkY");
	chunksize[2] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, t_grndVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&t_grndVarID);
    }
    return t_grndVarID;
  }
  
  if (strcmp(varName,"qflx_qirr")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = yCLMID;
    (*myVarNCData)->dimIDs[2] = xCLMID;
    int qQirrVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&qQirrVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkY");
	chunksize[2] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, qQirrVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&qQirrVarID);
    }
    return qQirrVarID;
  }
  
  if (strcmp(varName,"qflx_qirr_inst")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timCLMID;
    (*myVarNCData)->dimIDs[1] = zCLMID;
    (*myVarNCData)->dimIDs[2] = yCLMID;
    (*myVarNCData)->dimIDs[3] = xCLMID;
     int qQirrInstCLMVarID;
    int res = nc_def_var(ncCLMID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&qQirrInstCLMVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if(strcmp(switch_name, default_val) != 0)
      {
	size_t chunksize[(*myVarNCData)->dimSize];
	chunksize[0] = 1;
	chunksize[1] = GetInt("NetCDF.ChunkZ");
	chunksize[2] = GetInt("NetCDF.ChunkY");
	chunksize[3] = GetInt("NetCDF.ChunkX");
	nc_def_var_chunking(ncCLMID, qQirrInstCLMVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncCLMID,varName,&qQirrInstCLMVarID);
    }
    return qQirrInstCLMVarID;
  }
#endif
}

void PutCLMDataInNC(int varID, Vector *v, double t, varNCData *myVarNCData, int dimensionality)
{
#ifdef PARFLOW_HAVE_NETCDF
  static int counter = 0;
  if (strcmp(myVarNCData->varName,"time")==0)
  {
    long end[MAX_NC_VARS];
    nc_var_par_access(ncCLMID, varID, NC_COLLECTIVE);
    find_variable_length(ncCLMID, varID, end);
    size_t start[myVarNCData->dimSize], count[myVarNCData->dimSize];
    start[0] = end[0]; count[0] = 1;
    int status = nc_put_vara_double(ncCLMID, varID, start, count, &t);
  }
  else
  {
    if (dimensionality==3)
    {
      long end[MAX_NC_VARS];
      nc_var_par_access(ncCLMID, varID, NC_COLLECTIVE);
      find_variable_length(ncCLMID, varID, end);
      size_t start[myVarNCData->dimSize], count[myVarNCData->dimSize];

      Grid *grid = VectorGrid(v);
      SubgridArray *subgrids = GridSubgrids(grid);
      Subgrid *subgrid;
      Subvector *subvector;
      int g;

      ForSubgridI(g, subgrids)
      {
	subgrid = SubgridArraySubgrid(subgrids, g);
	subvector = VectorSubvector(v, g);
      }

      int ix = SubgridIX(subgrid);
      int iy = SubgridIY(subgrid);
      int iz = SubgridIZ(subgrid);

      int nx = SubgridNX(subgrid);
      int ny = SubgridNY(subgrid);
      int nz = SubgridNZ(subgrid);


      int nx_v = SubvectorNX(subvector);
      int ny_v = SubvectorNY(subvector);

      int i, j, k, d, ai; 
      double *data;
      double *data_nc;
      data_nc = (double *)malloc(sizeof(double)*nx*ny*nz);

      data = SubvectorElt(subvector, ix, iy, iz);
      ai = 0, d = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz, ai, nx_v, ny_v, nz_v, 1, 1, 1,{ data_nc[d] = data[ai]; d++;});
      start[0] = end[0]-1; start[1] = iz; start[2] = iy; start[3] = ix;
      count[0] = 1; count[1] = nz; count[2] = ny; count[3] = nx;
      int status = nc_put_vara_double(ncCLMID, varID, start, count, &data_nc[0]);
      free(data_nc);
    }
    else if (dimensionality==2)
    {

      long end[MAX_NC_VARS];
      nc_var_par_access(ncCLMID, varID, NC_COLLECTIVE);
      find_variable_length(ncCLMID, varID, end);
      size_t start[myVarNCData->dimSize], count[myVarNCData->dimSize];

      Grid *grid = VectorGrid(v);
      SubgridArray *subgrids = GridSubgrids(grid);
      Subgrid *subgrid;
      Subvector *subvector;
      int g;

      ForSubgridI(g, subgrids)
      {
	subgrid = SubgridArraySubgrid(subgrids, g);
	subvector = VectorSubvector(v, g);
      }

      int ix = SubgridIX(subgrid);
      int iy = SubgridIY(subgrid);
      int iz = SubgridIZ(subgrid);

      int nx = SubgridNX(subgrid);
      int ny = SubgridNY(subgrid);
      int nz = SubgridNZ(subgrid);


      int nx_v = SubvectorNX(subvector);
      int ny_v = SubvectorNY(subvector);

      int i, j, k, d, ai; 
      double *data;
      double *data_nc;
      data_nc = (double *)malloc(sizeof(double)*nx*ny*nz);

      data = SubvectorElt(subvector, ix, iy, iz);
      ai = 0, d = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz, ai, nx_v, ny_v, nz_v, 1, 1, 1,{ data_nc[d] = data[ai]; d++;});
      start[0] = end[0]-1; start[1] = iy; start[2] = ix;
      count[0] = 1; count[1] = ny; count[2] = nx;
      int status = nc_put_vara_double(ncCLMID, varID, start, count, &data_nc[0]);
      free(data_nc);

    }
  }
#endif
}

void CloseCLMNC(int ncCLMID)
{
#ifdef PARFLOW_HAVE_NETCDF
  nc_close(ncCLMID);
#endif
}
