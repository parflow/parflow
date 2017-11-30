#include <fca/fca.h>
#include "parflow_config.h"
// TODO: those paths could be changed right?:
#include <parflow.h>
#include <parflow_netcdf.h>
#include <messages.h>
#include <netcdf_par.h>
#include <netcdf.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>

#include "../../pfsimulator/amps/mpi1/amps.h"

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

int CreateFile(const char* file_name, size_t nX, size_t nY, size_t nZ, int *pxID, int *pyID, int *pzID, int *ptimeID)
{
  // Opens/ creates file if not already there.
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
  printf("starting netcdf-writer\n");
  MPI_Init(&argc, &argv);
  // TODO: probably we will need to call flowvr fca_init_parallel here too ;)


  /***********************
   * init FlowVR Module
   */
  fca_module moduleNetCDFWriter = fca_new_empty_module();
  fca_port portPressureIn = fca_new_port("pressureIn", fca_IN, 0, NULL);

  fca_append_port(moduleNetCDFWriter, portPressureIn);

  fca_port portOut = fca_new_port("out", fca_OUT, 0, NULL);
  fca_append_port(moduleNetCDFWriter, portOut);

  const fca_stamp stampTime = fca_register_stamp(portPressureIn, "stampTime", fca_FLOAT); // TODO good idea to use float? or should we put the double in the messages payload??
  const fca_stamp stampFileName = fca_register_stamp(portPressureIn, "stampFileName", fca_STRING);

  if (!fca_init_module(moduleNetCDFWriter))
  {
    printf("ERROR : init_module failed!\n");
  }

  int currentFileID;
  int xID, yID, zID, timeID;
  D("now Waiting\n");
  while (fca_wait(moduleNetCDFWriter))  // TODO: use our reader loop here maybe?
  {
    D("got some stuff to write\n");
    fca_message msg = fca_get(portPressureIn);
    char *file_name = (char*)fca_read_stamp(msg, stampFileName);

    // to access the variables in the ncFile...
    int pressureVarID;
    int timeVarID;

    D("number of segments: %d", fca_number_of_segments(msg));
    D("size: %d", fca_get_segment_size(msg, 0));
    D("writing to %s", file_name);

    void* buffer = fca_get_read_access(msg, 0);
    void* end = buffer + fca_get_segment_size(msg, 0);

    GridMessageMetadata* m = (GridMessageMetadata*)buffer;
    currentFileID = CreateFile(file_name, m->grid.nX, m->grid.nY, m->grid.nZ, &xID, &yID, &zID, &timeID);
    // add variable Time and pressure:
    nc_def_var(currentFileID, "time", NC_DOUBLE, 1, &timeID, &timeVarID);
    int pressure_dims[4] = { timeID, zID, yID, xID }; // low: why in this order? I guess so it will count in the right order ;)
    nc_def_var(currentFileID, "pressure", NC_DOUBLE, 4, pressure_dims, &pressureVarID);
    D("Adding Time");

    size_t start[1], count[1];
    nc_var_par_access(currentFileID, timeVarID, NC_COLLECTIVE);
    find_variable_length(currentFileID, timeVarID, start);
    D("start writing timestep %f into file %s at %d\n", m->time, file_name, start[0]);
    count[0] = 1;  // writing one value
    int status = nc_put_vara_double(currentFileID, timeVarID, start, count, &(m->time));

    while (buffer < end)
    {
      buffer += sizeof(GridMessageMetadata);
      nc_var_par_access(currentFileID, pressureVarID, NC_COLLECTIVE);
      // write next timestep
      size_t start[4] = { 0, m->iz, m->iy, m->ix };
      size_t count[4] = { 1, m->nz, m->ny, m->nx };
      find_variable_length(currentFileID, timeVarID, &(start[0]));
      start[0] = start[0] - 1;
      double const * const data = (double*)buffer;
      // now do a write to the cdf! (! all the preparations up there are necessary!
      int status = nc_put_vara_double(currentFileID, pressureVarID, start, count, data);
      D("putting doubles");

      buffer += sizeof(double) * m->nx * m->ny * m->nz;
      m = (GridMessageMetadata*)buffer;
    }

    nc_close(currentFileID);
    D("wrote %s\n", file_name);


    fca_free(msg);
  }

  fca_free(moduleNetCDFWriter);
  MPI_Finalize();
}
