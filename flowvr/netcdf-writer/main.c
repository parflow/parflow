#include <fca/fca.h>
#include "parflow_config.h"
#include <parflow.h>
#include <parflow_netcdf.h>
#include <messages.h>
#include <netcdf_par.h>
#include <netcdf.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <amps.h>

// REM: when it receives a message:
// take first gridmessage found in it and get nX, nY, nZ, time, variable name from it.
// for the first and all further grid messages: copy all the data in the netcdf file specified by the filename stamp.
// REM: you cannot have multiple writers for one file!

#ifdef __DEBUG
#define D(x ...) printf("++++++++ "); printf(x); printf(" %s:%d\n", __FILE__, __LINE__)
#else
#define D(...)
#endif

// creates dimension or only returns its id if already existent.
int getDim(int ncID, const char *name, size_t len, int *idp)
{
  int res = nc_def_dim(ncID, name, len, idp);

  if (res == NC_ENAMEINUSE)
  {
    res = nc_inq_varid(ncID, name, idp);
  }
  return res;
}

// creates dimension or only returns its id if already existent.
void getVar(int ncID, const char *name, int ndims, const int dimids[], int *idp)
{
  int res = nc_def_var(ncID, name, NC_DOUBLE, ndims, dimids, idp);

  if (res == NC_ENAMEINUSE)
  {
    if (nc_inq_varid(ncID, name, idp) != NC_NOERR)
    {
      PARFLOW_ERROR("Could not find variable");
    }
  }
}

int CreateFile(const char* file_name, size_t nX, size_t nY, size_t nZ, int *pxID, int *pyID, int *pzID, int *ptimeID)
{
  // Opens/ creates file if not already there.  REFACTOR: rename
  int ncID = 0;

  if (access(file_name, F_OK) == -1)
  {
    if (nc_create_par(file_name, NC_NETCDF4 | NC_MPIIO, amps_CommWorld, MPI_INFO_NULL, &ncID) != NC_NOERR)
    {
      D("Error creating file!");
      PARFLOW_ERROR("Could not create file!");
    }
  }
  else
  {
    // open it
    D("file exists already. Opening it!");
    if (nc_open_par(file_name, NC_MPIIO | NC_WRITE, amps_CommWorld, MPI_INFO_NULL, &ncID) != NC_NOERR)
    {
      D("Error opening existing file file!");
      PARFLOW_ERROR("Could not open file!");
    }
  }

  // create file. if it exists already, just load it.

  // add/get Dimensions
  getDim(ncID, "x", nX, pxID);
  getDim(ncID, "y", nY, pyID);
  getDim(ncID, "z", nZ, pzID);
  getDim(ncID, "time", NC_UNLIMITED, ptimeID);


  return ncID;
}

int main(int argc, char *argv [])
{
  char * prefix = "";

  if (argc > 1)
  {
    prefix = argv[1];
  }

  printf("starting netcdf-writer\n");
  MPI_Init(&argc, &argv);
  // TODO: probably we will need to call flowvr fca_init_parallel here too ;)


  /***********************
   * init FlowVR Module
   */
  fca_module flowvr = fca_new_empty_module();
  fca_port portIn = fca_new_port("in", fca_IN, 0, NULL);

  fca_append_port(flowvr, portIn);

  const fca_stamp stampFileName = fca_register_stamp(portIn, "stampFileName", fca_STRING);

  if (!fca_init_module(flowvr))
  {
    printf("ERROR : init_module failed!\n");
  }

  int currentFileID;
  int xID, yID, zID, timeID;
  D("now Waiting\n");
  while (fca_wait(flowvr))  // low: use our reader loop here maybe? Not atm as this version should be faster ;)
  {
    D("got some stuff to write\n");
    fca_message msg = fca_get(portIn);
    char file_name[1024];
    sprintf(file_name, "%s%s.nc", prefix, (char*)fca_read_stamp(msg, stampFileName));

    // to access the variables in the ncFile...
    int variableVarID;
    int timeVarID;

    D("number of segments: %d", fca_number_of_segments(msg));
    D("size: %d", fca_get_segment_size(msg, 0));
    D("writing to %s", file_name);

    void* buffer = fca_get_read_access(msg, 0);
    void* end = buffer + fca_get_segment_size(msg, 0);

    // the current Grid dimensions that are used in the file
    GridMessageMetadata *m = (GridMessageMetadata*)buffer;
    GridDefinition *file_grid = &(m->grid);
    double curTime = m->time;
    Variable last_var = VARIABLE_LAST;
    currentFileID = CreateFile(file_name, m->grid.nX, m->grid.nY, m->grid.nZ, &xID, &yID, &zID, &timeID);
    // add variable Time and variable "variable":
    nc_def_var(currentFileID, "time", NC_DOUBLE, 1, &timeID, &timeVarID);
    int variable_dims[4] = { timeID, zID, yID, xID }; // low: why in this order? I guess so it will count in the right order ;)
    D("Adding Time %f for %dx%dx%d, variable %s", m->time, m->grid.nX, m->grid.nY, m->grid.nZ, VARIABLE_TO_NAME[m->variable]);

    size_t start[1], count[1];
    nc_var_par_access(currentFileID, timeVarID, NC_COLLECTIVE);
    find_variable_length(currentFileID, timeVarID, start);
    D("start writing timestep %f into file %s at %d\n", m->time, file_name, start[0]);
    count[0] = 1;  // writing one value
    int status = nc_put_vara_double(currentFileID, timeVarID, start, count, &(m->time));

    while (buffer < end)
    {
      // low: we might also just ignore wrong messages!
      assert(file_grid->nX == m->grid.nX);
      assert(file_grid->nY == m->grid.nY);
      assert(file_grid->nZ == m->grid.nZ);
      assert(curTime == m->time);
      if (last_var != m->variable)
      {
        getVar(currentFileID, VARIABLE_TO_NAME[m->variable], 4, variable_dims,
               &variableVarID);
      }

      buffer += sizeof(GridMessageMetadata);
      nc_var_par_access(currentFileID, variableVarID, NC_COLLECTIVE);
      // write next timestep
      size_t start[4] = { 0, m->iz, m->iy, m->ix };
      size_t count[4] = { 1, m->nz, m->ny, m->nx };
      find_variable_length(currentFileID, timeVarID, &(start[0]));
      start[0] = start[0] - 1;
      double const * const data = (double*)buffer;
      // now do a write to the cdf! (! all the preparations up there are necessary!
      int status = nc_put_vara_double(currentFileID, variableVarID, start, count, data);
      D("putting doubles");

      buffer += sizeof(double) * m->nx * m->ny * m->nz;
      m = (GridMessageMetadata*)buffer;
    }

    nc_close(currentFileID);
    D("wrote %s\n", file_name);

    fca_free(msg);
  }

  fca_free(flowvr);
  MPI_Finalize();
}
