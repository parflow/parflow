/*BHEADER**********************************************************************

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

#include "parflow.h"
#include "parflow_netcdf.h"
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void WritePFNC(char * file_prefix, char* file_postfix, double t, Vector  *v, int numVarTimeVariant,
			char *varName, int dimensionality, int timDimensionality)
{
#ifdef PARFLOW_HAVE_NETCDF

  char *default_val = "False";
  char *switch_name;
  char key[IDB_MAX_KEY_LEN];
  sprintf(key, "NetCDF.NodeLevelIO");
  switch_name = GetStringDefault(key, "False");
  if(strcmp(switch_name, default_val) != 0)
  {
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

    int *nodeXIndices=NULL, *nodeYIndices=NULL, *nodeZIndices=NULL;
    int *nodeXCount=NULL, *nodeYCount=NULL, *nodeZCount=NULL;
    int nodeXTotal, nodeYTotal, nodeZTotal;
    if(amps_node_rank == 0)
    {
      nodeXIndices =  (int *)malloc(sizeof(int)*amps_node_size);
      nodeYIndices =  (int *)malloc(sizeof(int)*amps_node_size);
      nodeZIndices =  (int *)malloc(sizeof(int)*amps_node_size);

      nodeXCount =  (int *)malloc(sizeof(int)*amps_node_size);
      nodeYCount =  (int *)malloc(sizeof(int)*amps_node_size);
      nodeZCount =  (int *)malloc(sizeof(int)*amps_node_size);

    }
    MPI_Gather(&ix, 1, MPI_INT, nodeXIndices, 1, MPI_INT, 0, amps_CommNode);
    MPI_Gather(&iy, 1, MPI_INT, nodeYIndices, 1, MPI_INT, 0, amps_CommNode);
    MPI_Gather(&iz, 1, MPI_INT, nodeZIndices, 1, MPI_INT, 0, amps_CommNode);

    MPI_Gather(&nx, 1, MPI_INT, nodeXCount, 1, MPI_INT, 0, amps_CommNode);
    MPI_Gather(&ny, 1, MPI_INT, nodeYCount, 1, MPI_INT, 0, amps_CommNode);
    MPI_Gather(&nz, 1, MPI_INT, nodeZCount, 1, MPI_INT, 0, amps_CommNode);

    MPI_Reduce(&nx, &nodeXTotal, 1, MPI_INT, MPI_SUM, 0, amps_CommNode);
    MPI_Reduce(&ny, &nodeYTotal, 1, MPI_INT, MPI_SUM, 0, amps_CommNode);
    MPI_Reduce(&nz, &nodeZTotal, 1, MPI_INT, MPI_SUM, 0, amps_CommNode);


    double *data_nc_node=NULL;
    int i, j, k, d, ai;

    if(amps_node_rank == 0)
    {
      //data_nc_node = (double *)malloc(sizeof(double)*nodeXTotal*nodeYTotal*nodeZTotal);
      data_nc_node = (double *)malloc(sizeof(double)*nodeXTotal*nodeYTotal*nz);
    }
    double *data;
    double *data_nc;
    data_nc = (double *)malloc(sizeof(double)*nx*ny*nz);

    data = SubvectorElt(subvector, ix, iy, iz);
    ai = 0, d = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz, ai, nx_v, ny_v, nz_v, 1, 1, 1,{ data_nc[d] = data[ai]; d++;});

    int *recvCounts=NULL, *disps=NULL;
    if(amps_node_rank == 0)
    {
      recvCounts = (int *)malloc(sizeof(int)*amps_node_size);
      disps = (int *)malloc(sizeof(int)*amps_node_size);
      for(i=0; i<amps_node_size; i++)
      {
	recvCounts[i] = nodeXCount[i]*nodeYCount[i]*nodeZCount[i];
	disps[i] = 0;
	for(j=0; j<i; j++)
	{
	  disps[i] += nodeXCount[j]*nodeYCount[j]*nodeZCount[j];
	}
      }
    }

    MPI_Gatherv(data_nc, nx*ny*nz, MPI_DOUBLE, data_nc_node, recvCounts, disps, MPI_DOUBLE, 0, amps_CommNode);
    free(data_nc);

    if (amps_node_rank == 0)
    {
      static int numStepsInFile = 0;
      int userSpecSteps = GetInt("NetCDF.NumStepsPerFile");
      static char file_name[255];
      static int numOfDefVars=0;

      varNCData *myVarNCData;

      if( numStepsInFile == userSpecSteps*numVarTimeVariant)
      {
	sprintf(file_name, "%s%s%s%s", file_prefix,".",file_postfix,".nc");
	CloseNC(ncID);
	CreateNCFileNodeFromVector(file_name, v);
	int myVarID=LookUpInventory(varName, &myVarNCData);
	PutDataInNCNode(myVarID, data_nc_node, nodeXIndices, nodeYIndices, nodeZIndices,
	    nodeXCount, nodeYCount, nodeZCount, t, myVarNCData, amps_node_size);
	numStepsInFile = 1;
	numOfDefVars=1;
      }
      else
      {
	if(numStepsInFile == 0)
	{
	  sprintf(file_name, "%s%s%s%s", file_prefix,".",file_postfix,".nc");
	  CreateNCFileNodeFromVector(file_name, v);
	  int myVarID=LookUpInventory(varName, &myVarNCData);
	  PutDataInNCNode(myVarID, data_nc_node, nodeXIndices, nodeYIndices, nodeZIndices,
	      nodeXCount, nodeYCount, nodeZCount, t, myVarNCData, amps_node_size);
	  numOfDefVars++;
	  //	CloseNC(ncID);
	  numStepsInFile++;
	}
	else
	{
	  numStepsInFile++;
	  int myVarID=LookUpInventory(varName, &myVarNCData);
	  PutDataInNCNode(myVarID, data_nc_node, nodeXIndices, nodeYIndices, nodeZIndices,
	      nodeXCount, nodeYCount, nodeZCount, t, myVarNCData, amps_node_size);
	  if( numStepsInFile == userSpecSteps*numVarTimeVariant)
	  {
	    CloseNC(ncID);
	  }
	}
      }
      free(data_nc_node);
    }

  }
  else
  {

    static int numStepsInFile = 0;
    int userSpecSteps = GetInt("NetCDF.NumStepsPerFile");
    static char file_name[255];
    static int numOfDefVars=0;

    varNCData *myVarNCData;

    if( numStepsInFile == userSpecSteps*numVarTimeVariant)
    {
      sprintf(file_name, "%s%s%s%s", file_prefix,".",file_postfix,".nc");
      CloseNC(ncID);
      CreateNCFile(file_name, v);
      int myVarID=LookUpInventory(varName, &myVarNCData);
      PutDataInNC(myVarID, v, t, myVarNCData);
      numStepsInFile = 1;
      numOfDefVars=1;
    }
    else
    {
      if(numStepsInFile == 0)
      {
	sprintf(file_name, "%s%s%s%s", file_prefix,".",file_postfix,".nc");
	CreateNCFile(file_name, v);
	int myVarID=LookUpInventory(varName, &myVarNCData);
	PutDataInNC(myVarID,v, t, myVarNCData);
	numOfDefVars++;
	numStepsInFile++;
      }
      else
      {
	numStepsInFile++;
	int myVarID=LookUpInventory(varName, &myVarNCData);
	PutDataInNC(myVarID,v, t, myVarNCData);
	if( numStepsInFile == userSpecSteps*numVarTimeVariant)
	{
	  CloseNC(ncID);
	}

      }
    }
  }

#else
   amps_Printf("Parflow not compiled with NetCDF, can't create NetCDF file\n");
#endif

}

void CreateNCFile(char *file_name, Vector *v)
{
#ifdef PARFLOW_HAVE_NETCDF
  Grid           *grid     = VectorGrid(v);

  char *switch_name;
  char key[IDB_MAX_KEY_LEN];
  char *default_val = "None";
  int old_fill_mode;

  int nX = SubgridNX(GridBackground(grid));
  int nY = SubgridNY(GridBackground(grid));
  int nZ = SubgridNZ(GridBackground(grid));

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
    int res = nc_create_par(file_name,NC_NETCDF4|NC_MPIIO, amps_CommWorld, romio_info, &ncID);
  }
  else
  {
    int res = nc_create_par(file_name,NC_NETCDF4|NC_MPIIO, amps_CommWorld, MPI_INFO_NULL, &ncID);
  }
  int res = nc_def_dim(ncID, "x", nX, &xID);
  res = nc_def_dim(ncID, "y", nY, &yID);
  res = nc_def_dim(ncID, "z", nZ, &zID);
  //res = nc_def_dim(ncID, "time",GetInt("NetCDF.NumStepsPerFile"),&timID);
  res = nc_def_dim(ncID, "time",NC_UNLIMITED,&timID);
#else
   amps_Printf("Parflow not compiled with NetCDF, can't create NetCDF file\n");
#endif
}

void CreateNCFileNode(char *file_name, int nX, int nY, int nZ, int *ncidp)
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
    int res = nc_create_par(file_name,NC_NETCDF4|NC_MPIIO, amps_CommWrite, romio_info, ncidp);
  }
  else
  {
    int res = nc_create_par(file_name,NC_NETCDF4|NC_MPIIO, amps_CommWrite, MPI_INFO_NULL, ncidp);
  }
  int res = nc_def_dim(*ncidp, "x", nX, &xID);
  res = nc_def_dim(*ncidp, "y", nY, &yID);
  res = nc_def_dim(*ncidp, "z", nZ, &zID);
  //res = nc_def_dim(*ncidp, "time",GetInt("NetCDF.NumStepsPerFile"),&timID);
  res = nc_def_dim(*ncidp, "time",NC_UNLIMITED,&timID);

#else
   amps_Printf("Parflow not compiled with NetCDF, can't create NetCDF file\n");
#endif
}

void CreateNCFileNodeFromVector(char *file_name, Vector *v)
{
#ifdef PARFLOW_HAVE_NETCDF
  Grid           *grid     = VectorGrid(v);

  int nX = SubgridNX(GridBackground(grid));
  int nY = SubgridNY(GridBackground(grid));
  int nZ = SubgridNZ(GridBackground(grid));
  CreateNCFileNode(file_name, nX, nY, nZ, &ncID);

#else
   amps_Printf("Parflow not compiled with NetCDF, can't create NetCDF file\n");
#endif

}

void CloseNC(int ncID)
{
#ifdef PARFLOW_HAVE_NETCDF
  nc_close(ncID);
#endif
}

int LookUpInventory(char * varName, varNCData **myVarNCData)
{
#ifdef PARFLOW_HAVE_NETCDF
  if (strcmp(varName,"time")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 1;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timID;
    static int timVarID;
    int res = nc_def_var(ncID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&timVarID);
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncID,varName,&timVarID);
      if (res != NC_NOERR) printf("Something went wrong in definition %d\n", res);
    }
    return timVarID;
  }
  if (strcmp(varName,"pressure")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timID;
    (*myVarNCData)->dimIDs[1] = zID;
    (*myVarNCData)->dimIDs[2] = yID;
    (*myVarNCData)->dimIDs[3] = xID;
    static int pressVarID;
    int res = nc_def_var(ncID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&pressVarID);
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
	nc_def_var_chunking(ncID, pressVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncID,varName,&pressVarID);
    }
    return pressVarID;
  }
  if (strcmp(varName,"saturation")==0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize*sizeof(int));
    (*myVarNCData)->dimIDs[0] = timID;
    (*myVarNCData)->dimIDs[1] = zID;
    (*myVarNCData)->dimIDs[2] = yID;
    (*myVarNCData)->dimIDs[3] = xID;
    static int satVarID;
    int res = nc_def_var(ncID, varName,(*myVarNCData)->ncType,(*myVarNCData)->dimSize,
	(*myVarNCData)->dimIDs,&satVarID);
    //    if (res != NC_ENAMEINUSE)
    //    {
    //      nc_def_var_chunking(ncID, satVarID, NC_CHUNKED, chunksize);
    //    }
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
	nc_def_var_chunking(ncID, satVarID, NC_CHUNKED, chunksize);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res= nc_inq_varid(ncID,varName,&satVarID);
    }
    return satVarID;
  }

#endif
}

void PutDataInNC(int varID, Vector *v, double t, varNCData *myVarNCData)
{
#ifdef PARFLOW_HAVE_NETCDF
  static int counter = 0;
  if (strcmp(myVarNCData->varName,"time")==0)
  {
    long end[MAX_NC_VARS];
    nc_var_par_access(ncID, varID, NC_COLLECTIVE);
    find_variable_length(ncID, varID, end);
    size_t start[myVarNCData->dimSize], count[myVarNCData->dimSize];
    start[0] = end[0]; count[0] = 1;
    //start[0] = counter; count[0] = 1;
    //int status = nc_put_vara_double(ncID, varID, &start, &count, &t);
    int status = nc_put_vara_double(ncID, varID, start, count, &t);
    //		counter++;
    //		if(counter == GetInt("NetCDF.NumStepsPerFile"))
    //		{
    //			counter = 0;
    //		}

  }
  //else if (strcmp(myVarNCData->varName,"pressure")==0)
  else
  {
    long end[MAX_NC_VARS];
    nc_var_par_access(ncID, varID, NC_COLLECTIVE);
    find_variable_length(ncID, varID, end);
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
    //		double data_nc[nx * ny * nz];
    double *data_nc;
    data_nc = (double *)malloc(sizeof(double)*nx*ny*nz);

    data = SubvectorElt(subvector, ix, iy, iz);
    ai = 0, d = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz, ai, nx_v, ny_v, nz_v, 1, 1, 1,{ data_nc[d] = data[ai]; d++;});
    start[0] = end[0]-1; start[1] = iz; start[2] = iy; start[3] = ix;
    //start[0] = counter; start[1] = iz; start[2] = iy; start[3] = ix;
    count[0] = 1; count[1] = nz; count[2] = ny; count[3] = nx;
    int status = nc_put_vara_double(ncID, varID, start, count, &data_nc[0]);
    free(data_nc);
  }
#endif
}

void PutDataInNCNode(int varID, double *data_nc_node, int *nodeXIndices, int *nodeYIndices, int *nodeZIndices,
    int *nodeXCount, int *nodeYCount, int *nodeZCount, double t, varNCData *myVarNCData, int nodeSize)
{
#ifdef PARFLOW_HAVE_NETCDF
  long end[MAX_NC_VARS];
  nc_var_par_access(ncID, varID, NC_COLLECTIVE);
  find_variable_length(ncID, varID, end);
  size_t start[myVarNCData->dimSize], count[myVarNCData->dimSize];
  if (strcmp(myVarNCData->varName, "time")==0)
  {
    start[0] = end[0]; count[0] = 1;
    int status = nc_put_vara_double(ncID, varID, start, count, &t);
  }
  else
  {

    int i, j, index, status;
    start[0] = end[0]-1; count[0] = 1;
    for(i=0; i<nodeSize; i++)
    {
      start[1] = nodeZIndices[i]; start[2] =nodeYIndices[i]; start[3] = nodeXIndices[i];
      //start[1] = 0; start[2] =nodeYIndices[i]; start[3] = nodeXIndices[i];
      count[1] = nodeZCount[i]; count[2] =nodeYCount[i]; count[3] = nodeXCount[i];
      index=0; // TODO: omg. fix it!
      for(j=0; j<i; j++)
      {
        index += nodeZCount[j]*nodeYCount[j]*nodeXCount[j];
      }
      status = nc_put_vara_double(ncID, varID, start, count, &data_nc_node[index]);
    }
  }
#endif
}


void find_variable_length( int nid, int varid, long dim_lengths[MAX_NC_VARS] )
{
#ifdef PARFLOW_HAVE_NETCDF
  int dim_ids[MAX_VAR_DIMS];
  char dim_name[MAX_NC_NAME];
  int ndims, natts, i;
  nc_type type;
  /* inquire on this variable to determine number of dimensions
   *   and dimension ids */
  nc_inq_var( nid, varid, 0, &type, &ndims, dim_ids, &natts );
  /* get the sizes of each dimension */
  for( i=0; i < ndims; i++ )
  {
    nc_inq_dim( nid, dim_ids[i], dim_name, &dim_lengths[i] );
  }
#endif
}
