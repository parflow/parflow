/*BHEADER*********************************************************************
 *
 *  This file is part of Parflow. For details, see
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

#ifdef PARFLOW_HAVE_NETCDF
static bool is2Ddefined = false;
static bool is3Ddefined = false;
static bool isTdefined = false;
#endif

void FreeVarNCData(varNCData* myVarNCData)
{
  if(myVarNCData)
  {
    free(myVarNCData -> dimIDs);
    free(myVarNCData);
  }
}

void WritePFNC(char * file_prefix, char* file_postfix, double t, Vector  *v, int numVarTimeVariant,
               char *varName, int dimensionality, bool init, int numVarIni)
{
#ifdef PARFLOW_HAVE_NETCDF
  char *default_val = "False";
  char *switch_name;
  char key[IDB_MAX_KEY_LEN];
  static int netCDFIDs[5]; /* Here we store file and dimension IDs */
  sprintf(key, "NetCDF.NodeLevelIO");
  switch_name = GetStringDefault(key, "False");
  if (strcmp(switch_name, default_val) != 0)
  {
    Grid *grid = VectorGrid(v);
    SubgridArray *subgrids = GridSubgrids(grid);
    Subgrid *subgrid = NULL;
    Subvector *subvector = NULL;
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
    int nz_v = SubvectorNZ(subvector);

    int *nodeXIndices = NULL, *nodeYIndices = NULL, *nodeZIndices = NULL;
    int *nodeXCount = NULL, *nodeYCount = NULL, *nodeZCount = NULL;
    int nodeXTotal, nodeYTotal, nodeZTotal;
    if (amps_node_rank == 0)
    {
      nodeXIndices = (int*)malloc(sizeof(int) * amps_node_size);
      nodeYIndices = (int*)malloc(sizeof(int) * amps_node_size);
      nodeZIndices = (int*)malloc(sizeof(int) * amps_node_size);

      nodeXCount = (int*)malloc(sizeof(int) * amps_node_size);
      nodeYCount = (int*)malloc(sizeof(int) * amps_node_size);
      nodeZCount = (int*)malloc(sizeof(int) * amps_node_size);
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


    double *data_nc_node = NULL;
    int i, j, k, d, ai;

    if (amps_node_rank == 0)
    {
      data_nc_node = (double*)malloc(sizeof(double) * nodeXTotal * nodeYTotal * nz);
    }
    double *data;
    double *data_nc;
    data_nc = (double*)malloc(sizeof(double) * nx * ny * nz);

    data = SubvectorElt(subvector, ix, iy, iz);
    ai = 0, d = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz, ai, nx_v, ny_v, nz_v, 1, 1, 1, { data_nc[d] = data[ai]; d++; });

    int *recvCounts = NULL, *disps = NULL;
    if (amps_node_rank == 0)
    {
      recvCounts = (int*)malloc(sizeof(int) * amps_node_size);
      disps = (int*)malloc(sizeof(int) * amps_node_size);
      for (i = 0; i < amps_node_size; i++)
      {
        recvCounts[i] = nodeXCount[i] * nodeYCount[i] * nodeZCount[i];
        disps[i] = 0;
        for (j = 0; j < i; j++)
        {
          disps[i] += nodeXCount[j] * nodeYCount[j] * nodeZCount[j];
        }
      }
    }

    MPI_Gatherv(data_nc, nx * ny * nz, MPI_DOUBLE, data_nc_node, recvCounts, disps, MPI_DOUBLE, 0, amps_CommNode);
    free(data_nc);

    if (amps_node_rank == 0)
    {
      static int numStepsInFile = 0;
      int userSpecSteps = GetInt("NetCDF.NumStepsPerFile");
      static char file_name[255];

      varNCData *myVarNCData=NULL;

      if (numStepsInFile == userSpecSteps * numVarTimeVariant)
      {
        sprintf(file_name, "%s%s%s%s", file_prefix, ".", file_postfix, ".nc");
        CloseNC(netCDFIDs[0]);
        CreateNCFileNode(file_name, v, netCDFIDs);
        int myVarID = LookUpInventory(varName, &myVarNCData, netCDFIDs);
        PutDataInNCNode(myVarID, data_nc_node, nodeXIndices, nodeYIndices, nodeZIndices,
                        nodeXCount, nodeYCount, nodeZCount, t, myVarNCData, netCDFIDs);
        numStepsInFile = 1;
      }
      else
      {
        if (numStepsInFile == 0)
        {
          sprintf(file_name, "%s%s%s%s", file_prefix, ".", file_postfix, ".nc");
          CreateNCFileNode(file_name, v, netCDFIDs);
          int myVarID = LookUpInventory(varName, &myVarNCData, netCDFIDs);
          PutDataInNCNode(myVarID, data_nc_node, nodeXIndices, nodeYIndices, nodeZIndices,
                          nodeXCount, nodeYCount, nodeZCount, t, myVarNCData, netCDFIDs);
          numStepsInFile++;
        }
        else
        {
          numStepsInFile++;
          int myVarID = LookUpInventory(varName, &myVarNCData, netCDFIDs);
          PutDataInNCNode(myVarID, data_nc_node, nodeXIndices, nodeYIndices, nodeZIndices,
                          nodeXCount, nodeYCount, nodeZCount, t, myVarNCData, netCDFIDs);
          if (numStepsInFile == userSpecSteps * numVarTimeVariant)
          {
            CloseNC(netCDFIDs[0]);
          }
        }
      }
      free(data_nc_node);
      FreeVarNCData(myVarNCData);
    }
  }
  else
  {
    static int numStepsInFile = 0;
    int userSpecSteps = GetInt("NetCDF.NumStepsPerFile");
    static char file_name[255];
    static int numOfDefVars = 0;

    varNCData *myVarNCData = NULL;
    if (init)
    {
      sprintf(file_name, "%s%s%s%s", file_prefix, ".", file_postfix, ".nc");
      if (numOfDefVars == 0)
      {
        CreateNCFile(file_name, netCDFIDs);
      }
      NCDefDimensions(v, dimensionality, netCDFIDs);
      int myVarID = LookUpInventory(varName, &myVarNCData, netCDFIDs);
      PutDataInNC(myVarID, v, t, myVarNCData, dimensionality, netCDFIDs);
      numOfDefVars++;
      if (numOfDefVars == numVarIni)
      {
        CloseNC(netCDFIDs[0]);
        is2Ddefined = false;
        is3Ddefined = false;
        isTdefined = false;
        numOfDefVars = 0;
      }
    }
    else
    {
      if (numStepsInFile == userSpecSteps * numVarTimeVariant)
      {
        sprintf(file_name, "%s%s%s%s", file_prefix, ".", file_postfix, ".nc");
        CloseNC(netCDFIDs[0]);
        is2Ddefined = false;
        is3Ddefined = false;
        isTdefined = false;
        CreateNCFile(file_name, netCDFIDs);
        NCDefDimensions(v, dimensionality, netCDFIDs);
        int myVarID = LookUpInventory(varName, &myVarNCData, netCDFIDs);
        PutDataInNC(myVarID, v, t, myVarNCData, dimensionality, netCDFIDs);
        numStepsInFile = 1;
        numOfDefVars = 1;
      }
      else
      {
        if (numStepsInFile == 0)
        {
          sprintf(file_name, "%s%s%s%s", file_prefix, ".", file_postfix, ".nc");
          CreateNCFile(file_name, netCDFIDs);
          NCDefDimensions(v, dimensionality, netCDFIDs);
          int myVarID = LookUpInventory(varName, &myVarNCData, netCDFIDs);
          PutDataInNC(myVarID, v, t, myVarNCData, dimensionality, netCDFIDs);
          numOfDefVars++;
          numStepsInFile++;
        }
        else
        {
          numStepsInFile++;
          NCDefDimensions(v, dimensionality, netCDFIDs);
          int myVarID = LookUpInventory(varName, &myVarNCData, netCDFIDs);
          PutDataInNC(myVarID, v, t, myVarNCData, dimensionality, netCDFIDs);
          if (numStepsInFile == userSpecSteps * numVarTimeVariant)
          {
            CloseNC(netCDFIDs[0]);
            is2Ddefined = false;
            is3Ddefined = false;
            isTdefined = false;
          }
        }
      }
    }
    FreeVarNCData(myVarNCData);
  }
#else
  amps_Printf("Parflow not compiled with NetCDF, can't create NetCDF file\n");
#endif
}

void CreateNCFile(char *file_name, int *netCDFIDs)
{
#ifdef PARFLOW_HAVE_NETCDF
  char *switch_name;
  char key[IDB_MAX_KEY_LEN];
  char *default_val = "None";

  sprintf(key, "NetCDF.ROMIOhints");
  switch_name = GetStringDefault(key, "None");
  if (strcmp(switch_name, default_val) != 0)
  {
    if (access(switch_name, F_OK | R_OK) == -1)
    {
      InputError("Error: check if the file is present and readable <%s> for key <%s>\n",
                 switch_name, key);
    }
    MPI_Info romio_info;
    FILE *fp;
    MPI_Info_create(&romio_info);
    char line[100], romio_key[100], value[100];
    fp = fopen(switch_name, "r");
    while (fgets(line, sizeof line, fp) != NULL)      /* read a line */
    {
      sscanf(line, "%s%s", romio_key, value);
      MPI_Info_set(romio_info, romio_key, value);
    }
    int res = nc_create_par(file_name, NC_NETCDF4 | NC_MPIIO, amps_CommWorld, romio_info, &netCDFIDs[0]);
    if (res != NC_NOERR)
    {
      printf("Error: nc_create_par failed for file <%s>, error code=%d\n", file_name, res);
    }
  }
  else
  {
    int res = nc_create_par(file_name, NC_NETCDF4 | NC_MPIIO, amps_CommWorld, MPI_INFO_NULL, &netCDFIDs[0]);
    if (res != NC_NOERR)
    {
      printf("Error: nc_create_par failed for file <%s>, error code=%d\n", file_name, res);
    }
  }
#else
  amps_Printf("Parflow not compiled with NetCDF, can't create NetCDF file\n");
#endif
}

void CreateNCFileNode(char *file_name, Vector *v, int *netCDFIDs)
{
#ifdef PARFLOW_HAVE_NETCDF
  Grid           *grid = VectorGrid(v);

  char *switch_name;
  char key[IDB_MAX_KEY_LEN];
  char *default_val = "None";

  int nX = SubgridNX(GridBackground(grid));
  int nY = SubgridNY(GridBackground(grid));
  int nZ = SubgridNZ(GridBackground(grid));

  sprintf(key, "NetCDF.ROMIOhints");
  switch_name = GetStringDefault(key, "None");
  if (strcmp(switch_name, default_val) != 0)
  {
    if (access(switch_name, F_OK | R_OK) == -1)
    {
      InputError("Error: check if the file is present and readable <%s> for key <%s>\n",
                 switch_name, key);
    }
    MPI_Info romio_info;
    FILE *fp;
    MPI_Info_create(&romio_info);
    char line[100], romio_key[100], value[100];
    fp = fopen(switch_name, "r");
    while (fgets(line, sizeof line, fp) != NULL)      /* read a line */
    {
      sscanf(line, "%s%s", romio_key, value);
      MPI_Info_set(romio_info, romio_key, value);
    }
    int res = nc_create_par(file_name, NC_NETCDF4 | NC_MPIIO, amps_CommWrite, romio_info, &netCDFIDs[0]);
    if (res != NC_NOERR)
    {
      printf("Error: nc_create_par failed for file <%s>, error code=%d\n", file_name, res);
    }

  }
  else
  {
    int res = nc_create_par(file_name, NC_NETCDF4 | NC_MPIIO, amps_CommWrite, MPI_INFO_NULL, &netCDFIDs[0]);
    if (res != NC_NOERR)
    {
      printf("Error: nc_create_par failed for file <%s>, error code=%d\n", file_name, res);
    }

  }
  nc_def_dim(netCDFIDs[0], "x", nX, &netCDFIDs[4]);
  nc_def_dim(netCDFIDs[0], "y", nY, &netCDFIDs[3]);
  nc_def_dim(netCDFIDs[0], "z", nZ, &netCDFIDs[2]);
  nc_def_dim(netCDFIDs[0], "time", NC_UNLIMITED, &netCDFIDs[1]);
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

int LookUpInventory(char * varName, varNCData **myVarNCData, int *netCDFIDs)
{
#ifdef PARFLOW_HAVE_NETCDF
  // Read NetCDF compression configuration settings
  int enable_netcdf_compression = 0;
  {
    char key[IDB_MAX_KEY_LEN];
    sprintf(key, "NetCDF.Compression");
    char *switch_name = GetStringDefault(key, "False");
    char *default_val = "False";
    enable_netcdf_compression = strcmp(switch_name, default_val);
  }
  int compression_level = 1;
  {
    char key[IDB_MAX_KEY_LEN];
    sprintf(key, "NetCDF.CompressionLevel");
    compression_level = GetIntDefault(key, 1);
  }

  if (strcmp(varName, "time") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 1;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    int timVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &timVarID);
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &timVarID);
      if (res != NC_NOERR)
        printf("Something went wrong in definition %d\n", res);
    }
    return timVarID;
  }
  if (strcmp(varName, "pressure") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[2];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[3] = netCDFIDs[4];
    int pressVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &pressVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkZ");
        chunksize[2] = GetInt("NetCDF.ChunkY");
        chunksize[3] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], pressVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],pressVarID,0,1,compression_level);
      }

    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &pressVarID);
    }
    return pressVarID;
  }

  if (strcmp(varName, "saturation") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[2];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[3] = netCDFIDs[4];
    int satVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &satVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkZ");
        chunksize[2] = GetInt("NetCDF.ChunkY");
        chunksize[3] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], satVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],satVarID,0,1,compression_level);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &satVarID);
    }
    return satVarID;
  }

  if (strcmp(varName, "mask") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[2];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[3] = netCDFIDs[4];
    int maskVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &maskVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkZ");
        chunksize[2] = GetInt("NetCDF.ChunkY");
        chunksize[3] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], maskVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],maskVarID,0,1,compression_level);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &maskVarID);
    }
    return maskVarID;
  }

  if (strcmp(varName, "mannings") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[4];
    int manningsVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &manningsVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkY");
        chunksize[2] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], manningsVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],manningsVarID,0,1,compression_level);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &manningsVarID);
    }
    return manningsVarID;
  }

  if (strcmp(varName, "perm_x") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[2];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[3] = netCDFIDs[4];
    int perm_xVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &perm_xVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkZ");
        chunksize[2] = GetInt("NetCDF.ChunkY");
        chunksize[3] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], perm_xVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],perm_xVarID,0,1,compression_level);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &perm_xVarID);
    }
    return perm_xVarID;
  }

  if (strcmp(varName, "perm_y") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[2];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[3] = netCDFIDs[4];
    int perm_yVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &perm_yVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkZ");
        chunksize[2] = GetInt("NetCDF.ChunkY");
        chunksize[3] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], perm_yVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],perm_yVarID,0,1,compression_level);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &perm_yVarID);
    }
    return perm_yVarID;
  }

  if (strcmp(varName, "perm_z") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[2];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[3] = netCDFIDs[4];
    int perm_zVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &perm_zVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkZ");
        chunksize[2] = GetInt("NetCDF.ChunkY");
        chunksize[3] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], perm_zVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],perm_zVarID,0,1,compression_level);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &perm_zVarID);
    }
    return perm_zVarID;
  }

  if (strcmp(varName, "porosity") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[2];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[3] = netCDFIDs[4];
    int porosityVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &porosityVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkZ");
        chunksize[2] = GetInt("NetCDF.ChunkY");
        chunksize[3] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], porosityVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],porosityVarID,0,1,compression_level);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &porosityVarID);
    }
    return porosityVarID;
  }

  if (strcmp(varName, "specific_storage") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[2];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[3] = netCDFIDs[4];
    int specStorageVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &specStorageVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkZ");
        chunksize[2] = GetInt("NetCDF.ChunkY");
        chunksize[3] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], specStorageVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],specStorageVarID,0,1,compression_level);
      }

    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &specStorageVarID);
    }
    return specStorageVarID;
  }

  if (strcmp(varName, "slopex") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[4];
    int slopexVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &slopexVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkY");
        chunksize[2] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], slopexVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],slopexVarID,0,1,compression_level);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &slopexVarID);
    }
    return slopexVarID;
  }
  if (strcmp(varName, "slopey") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[4];
    int slopeyVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &slopeyVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkY");
        chunksize[2] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], slopeyVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],slopeyVarID,0,1,compression_level);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &slopeyVarID);
    }
    return slopeyVarID;
  }
  if (strcmp(varName, "DZ_Multiplier") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[2];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[3] = netCDFIDs[4];
    int dzmultVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &dzmultVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkZ");
        chunksize[2] = GetInt("NetCDF.ChunkY");
        chunksize[3] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], dzmultVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],dzmultVarID,0,1,compression_level);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &dzmultVarID);
    }
    return dzmultVarID;
  }

  if (strcmp(varName, "evaptrans") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[2];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[3] = netCDFIDs[4];
    int evaptransVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &evaptransVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkZ");
        chunksize[2] = GetInt("NetCDF.ChunkY");
        chunksize[3] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], evaptransVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],evaptransVarID,0,1,compression_level);
      }

    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &evaptransVarID);
    }
    return evaptransVarID;
  }

  if (strcmp(varName, "evaptrans_sum") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 4;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[2];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[3] = netCDFIDs[4];
    int evaptrans_sumVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &evaptrans_sumVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkZ");
        chunksize[2] = GetInt("NetCDF.ChunkY");
        chunksize[3] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], evaptrans_sumVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],evaptrans_sumVarID,0,1,compression_level);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &evaptrans_sumVarID);
    }
    return evaptrans_sumVarID;
  }
  if (strcmp(varName, "overland_sum") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[4];
    int overland_sumVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &overland_sumVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkY");
        chunksize[2] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], overland_sumVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],overland_sumVarID,0,1,compression_level);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &overland_sumVarID);
    }
    return overland_sumVarID;
  }

  if (strcmp(varName, "overland_bc_flux") == 0)
  {
    *myVarNCData = malloc(sizeof(varNCData));
    (*myVarNCData)->varName = varName;
    (*myVarNCData)->ncType = NC_DOUBLE;
    (*myVarNCData)->dimSize = 3;
    (*myVarNCData)->dimIDs = malloc((*myVarNCData)->dimSize * sizeof(int));
    (*myVarNCData)->dimIDs[0] = netCDFIDs[1];
    (*myVarNCData)->dimIDs[1] = netCDFIDs[3];
    (*myVarNCData)->dimIDs[2] = netCDFIDs[4];
    int overland_bc_fluxVarID;
    int res = nc_def_var(netCDFIDs[0], varName, (*myVarNCData)->ncType, (*myVarNCData)->dimSize,
                         (*myVarNCData)->dimIDs, &overland_bc_fluxVarID);
    if (res != NC_ENAMEINUSE)
    {
      char *switch_name;
      char key[IDB_MAX_KEY_LEN];
      char *default_val = "None";
      sprintf(key, "NetCDF.Chunking");
      switch_name = GetStringDefault(key, "None");
      if (strcmp(switch_name, default_val) != 0)
      {
        size_t chunksize[(*myVarNCData)->dimSize];
        chunksize[0] = 1;
        chunksize[1] = GetInt("NetCDF.ChunkY");
        chunksize[2] = GetInt("NetCDF.ChunkX");
        nc_def_var_chunking(netCDFIDs[0], overland_bc_fluxVarID, NC_CHUNKED, chunksize);
      }
      if (enable_netcdf_compression) {
        nc_def_var_deflate(netCDFIDs[0],overland_bc_fluxVarID,0,1,compression_level);
      }
    }
    if (res == NC_ENAMEINUSE)
    {
      res = nc_inq_varid(netCDFIDs[0], varName, &overland_bc_fluxVarID);
    }
    return overland_bc_fluxVarID;
  }

  return 0;
#else
  return 0;
#endif
}

void PutDataInNC(int varID, Vector *v, double t, varNCData *myVarNCData, int dimensionality, int *netCDFIDs)
{
#ifdef PARFLOW_HAVE_NETCDF
  if (strcmp(myVarNCData->varName, "time") == 0)
  {
    unsigned long end[MAX_NC_VARS];
    nc_var_par_access(netCDFIDs[0], varID, NC_COLLECTIVE);
    find_variable_length(netCDFIDs[0], varID, end);
    size_t start[myVarNCData->dimSize], count[myVarNCData->dimSize];
    start[0] = end[0]; count[0] = 1;
    int status = nc_put_vara_double(netCDFIDs[0], varID, start, count, &t);
    if (status != NC_NOERR)
    {
      printf("Error: nc_put_vara_double failed, error code=%d\n", status);
    }
  }
  else
  {
    if (dimensionality == 3)
    {
      unsigned long end[MAX_NC_VARS];
      nc_var_par_access(netCDFIDs[0], varID, NC_COLLECTIVE);
      find_variable_length(netCDFIDs[0], varID, end);
      size_t start[myVarNCData->dimSize], count[myVarNCData->dimSize];

      Grid *grid = VectorGrid(v);
      SubgridArray *subgrids = GridSubgrids(grid);
      Subgrid *subgrid = NULL;
      Subvector *subvector = NULL;
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
      int nz_v = SubvectorNZ(subvector);

      int i, j, k, d, ai;
      double *data;
      double *data_nc;
      data_nc = (double*)malloc(sizeof(double) * nx * ny * nz);

      data = SubvectorElt(subvector, ix, iy, iz);
      ai = 0, d = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz, ai, nx_v, ny_v, nz_v, 1, 1, 1, { data_nc[d] = data[ai]; d++; });
      start[0] = end[0] - 1; start[1] = iz; start[2] = iy; start[3] = ix;
      count[0] = 1; count[1] = nz; count[2] = ny; count[3] = nx;
      int status = nc_put_vara_double(netCDFIDs[0], varID, start, count, &data_nc[0]);
      if (status != NC_NOERR)
      {
	printf("Error: nc_put_vara_double failed, error code=%d\n", status);
      }
      free(data_nc);
    }
    else if (dimensionality == 2)
    {
      unsigned long end[MAX_NC_VARS];
      nc_var_par_access(netCDFIDs[0], varID, NC_COLLECTIVE);
      find_variable_length(netCDFIDs[0], varID, end);
      size_t start[myVarNCData->dimSize], count[myVarNCData->dimSize];

      Grid *grid = VectorGrid(v);
      SubgridArray *subgrids = GridSubgrids(grid);
      Subgrid *subgrid = NULL;
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
      int nz_v = SubvectorNZ(subvector);

      int i, j, k, d, ai;
      double *data;
      double *data_nc;
      data_nc = (double*)malloc(sizeof(double) * nx * ny * nz);

      data = SubvectorElt(subvector, ix, iy, iz);
      ai = 0, d = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz, ai, nx_v, ny_v, nz_v, 1, 1, 1, { data_nc[d] = data[ai]; d++; });
      start[0] = end[0] - 1; start[1] = iy; start[2] = ix;
      count[0] = 1; count[1] = ny; count[2] = nx;
      int status = nc_put_vara_double(netCDFIDs[0], varID, start, count, &data_nc[0]);
      if (status != NC_NOERR)
      {
	printf("Error: nc_put_vara_double failed, error code=%d\n", status);
      }
      free(data_nc);
    }
  }
#endif
}

void PutDataInNCNode(int varID, double *data_nc_node, int *nodeXIndices, int *nodeYIndices, int *nodeZIndices,
                     int *nodeXCount, int *nodeYCount, int *nodeZCount, double t, varNCData *myVarNCData, int *netCDFIDs)
{
#ifdef PARFLOW_HAVE_NETCDF
  if (strcmp(myVarNCData->varName, "time") == 0)
  {
    unsigned long end[MAX_NC_VARS];
    nc_var_par_access(netCDFIDs[0], varID, NC_COLLECTIVE);
    find_variable_length(netCDFIDs[0], varID, end);
    size_t start[myVarNCData->dimSize], count[myVarNCData->dimSize];
    start[0] = end[0]; count[0] = 1;
    int status = nc_put_vara_double(netCDFIDs[0], varID, start, count, &t);
    if (status != NC_NOERR)
    {
      printf("Error: nc_put_vara_double failed, error code=%d\n", status);
    }

  }
  else
  {
    unsigned long end[MAX_NC_VARS];
    nc_var_par_access(netCDFIDs[0], varID, NC_COLLECTIVE);
    find_variable_length(netCDFIDs[0], varID, end);
    size_t start[myVarNCData->dimSize], count[myVarNCData->dimSize];

    int i, j, index, status;
    start[0] = end[0] - 1; count[0] = 1;
    for (i = 0; i < amps_node_size; i++)
    {
      start[1] = nodeZIndices[i]; start[2] = nodeYIndices[i]; start[3] = nodeXIndices[i];
      count[1] = nodeZCount[i]; count[2] = nodeYCount[i]; count[3] = nodeXCount[i];
      index = 0;
      for (j = 0; j < i; j++)
      {
        index += nodeZCount[j] * nodeYCount[j] * nodeXCount[j];
      }
      status = nc_put_vara_double(netCDFIDs[0], varID, start, count, &data_nc_node[index]);
      if (status != NC_NOERR)
      {
	printf("Error: nc_put_vara_double failed, error code=%d\n", status);
      }
    }
  }
#endif
}


void find_variable_length(int nid, int varid, unsigned long dim_lengths[MAX_NC_VARS])
{
#ifdef PARFLOW_HAVE_NETCDF
  int dim_ids[MAX_VAR_DIMS];
  char dim_name[MAX_NC_NAME];
  int ndims, natts, i;
  nc_type type;
  /* inquire on this variable to determine number of dimensions
   *   and dimension ids */
  nc_inq_var(nid, varid, 0, &type, &ndims, dim_ids, &natts);
  /* get the sizes of each dimension */
  for (i = 0; i < ndims; i++)
  {
    nc_inq_dim(nid, dim_ids[i], dim_name, &dim_lengths[i]);
  }
#endif
}

void NCDefDimensions(Vector *v, int dimensionality, int *netCDFIDs)
{
#ifdef PARFLOW_HAVE_NETCDF
  if (dimensionality == 1 && !isTdefined)
  {
    nc_def_dim(netCDFIDs[0], "time", NC_UNLIMITED, &netCDFIDs[1]);
    isTdefined = true;
  }

  if (dimensionality == 2 && !is2Ddefined)
  {
    Grid           *grid = VectorGrid(v);

    int nX = SubgridNX(GridBackground(grid));
    int nY = SubgridNY(GridBackground(grid));

    nc_def_dim(netCDFIDs[0], "x", nX, &netCDFIDs[4]);
    nc_def_dim(netCDFIDs[0], "y", nY, &netCDFIDs[3]);
    is2Ddefined = true;
  }

  if (dimensionality == 3 && !is3Ddefined)
  {
    Grid           *grid = VectorGrid(v);

    int nX = SubgridNX(GridBackground(grid));
    int nY = SubgridNY(GridBackground(grid));
    int nZ = SubgridNZ(GridBackground(grid));

    int res = nc_inq_dimid(netCDFIDs[0], "x", &netCDFIDs[4]);
    if (res != NC_NOERR)
      nc_def_dim(netCDFIDs[0], "x", nX, &netCDFIDs[4]);

    res = nc_inq_dimid(netCDFIDs[0], "y", &netCDFIDs[3]);
    if (res != NC_NOERR)
      nc_def_dim(netCDFIDs[0], "y", nY, &netCDFIDs[3]);

    res = nc_def_dim(netCDFIDs[0], "z", nZ, &netCDFIDs[2]);


    is3Ddefined = true;
  }
#endif
}
