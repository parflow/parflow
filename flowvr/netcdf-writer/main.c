#include <fca/fca.h>
#include "parflow_config.h"
#include "../../pfsimulator/parflow_lib/parflow.h"
#include "../../pfsimulator/parflow_lib/parflow_netcdf.h"
#include "../../pfsimulator/parflow_lib/GridMessage.h"
#include <netcdf_par.h>
#include <netcdf.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>

#include "../../pfsimulator/amps/mpi1/amps.h"

#ifdef __DEBUG
#define D(x...) printf("++++++++ "); printf(x); printf("\n");
#else
#define D(...)
#endif

// creates dimension or only returns its id if already existent.
int getDim(int ncID, const char *name, size_t len, int *idp)
{
  int res = nc_def_dim(ncID, name, len, idp);
  if (res == NC_ENAMEINUSE) {
    res= nc_inq_varid(ncID, name, idp);
  }
  return res;
}

int CreateFile(const char* file_name, size_t nX, size_t nY, size_t nZ, int *pxID, int *pyID, int *pzID, int *ptimeID) {
  // Opens/ creates file if not already there.
  int ncID = 0;

  if (access( file_name, F_OK ) == -1)
  {
    if (nc_create_par(file_name, NC_NETCDF4|NC_MPIIO, amps_CommWrite, MPI_INFO_NULL, &ncID) != NC_NOERR)
    {
      D("Error creating file!")
      PARFLOW_ERROR("Could not create file!");
    }
  }
  else
  {
    // open it
    D("file exists already. Opening it!");
    if (nc_open_par(file_name, NC_MPIIO|NC_WRITE, amps_CommWorld, MPI_INFO_NULL, &ncID) != NC_NOERR)
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
  getDim(ncID, "time", NC_UNLIMITED, ptimeID);  // unlimited dim should be first?! TODO?


  return ncID;
}

int main (int argc , char *argv [])
{

  printf("starting netcdf-writer\n");
  MPI_Init(&argc, &argv);


  /***********************
   * init FlowVR Module
   */
  fca_module moduleNetCDFWriter = fca_new_empty_module();
  fca_port portPressureIn = fca_new_port("pressureIn", fca_IN, 0, NULL);

  fca_append_port(moduleNetCDFWriter, portPressureIn);

  fca_port outPort = fca_new_port("outPort", fca_OUT, 0, NULL);
  fca_append_port(moduleNetCDFWriter, outPort);

  const fca_stamp stampTime = fca_register_stamp(portPressureIn, "stampTime", fca_FLOAT);// TODO good idea to use float? or should we put the double in the messages payload??
  const fca_stamp stampFileName = fca_register_stamp(portPressureIn, "stampFileName", fca_STRING);

  if(!fca_init_module(moduleNetCDFWriter)){
    printf("ERROR : init_module failed!\n");
  }

  int currentFileID;
  int xID, yID, zID, timeID;
  D("nowWaiting\n");
  while (fca_wait(moduleNetCDFWriter)) {
    D("got some stuff to write\n");
    fca_message msg = fca_get(portPressureIn);
    double time = (double) *((float*) fca_read_stamp(msg, stampTime));
    char *file_name = (char*) fca_read_stamp(msg, stampFileName);

    // to access the variables in the ncFile...
    int pressureVarID;
    int timeVarID;

    assert(fca_number_of_segments(msg)%2 == 0);
    for (int i = 0; i < fca_number_of_segments(msg); i+=2) {
      GridMessageMetadata* m = (GridMessageMetadata*)fca_get_read_access(msg, i);
      if (i == 0) {
        currentFileID = CreateFile(file_name, m->nX, m->nY, m->nZ, &xID, &yID, &zID, &timeID);
        // add variable Time and pressure:
        nc_def_var(currentFileID, "time", NC_DOUBLE, 1, &timeID, &timeVarID);
        int pressure_dims[4] = {timeID, zID, yID, xID}; // low: why in this order? I guess so it will count in the right order ;)
        nc_def_var(currentFileID, "pressure", NC_DOUBLE, 4, pressure_dims, &pressureVarID);
        D("Adding Time");

        size_t start[1], count[1];
        nc_var_par_access(currentFileID, timeVarID, NC_COLLECTIVE);
        find_variable_length(currentFileID, timeVarID, start);
        D("start writing timestep %f into file %s at %d\n", time, file_name, start[0]);
        count[0] = 1;  // writing one value
        int status = nc_put_vara_double(currentFileID, timeVarID, start, count, &time);
      }

      nc_var_par_access(currentFileID, pressureVarID, NC_COLLECTIVE);
      // write next timestep
      size_t start[4] = {0, m->iz, m->iy, m->ix};
      size_t count[4] = {1, m->nz, m->ny, m->nx};
      find_variable_length(currentFileID, timeVarID, &(start[0]));
      start[0] = start[0] - 1;
      double const * const data = (double*)fca_get_read_access(msg, i+1);
      // now do a write to the cdf! (! all the preparations up there are necessary!
      int status = nc_put_vara_double(currentFileID, pressureVarID, start, count, data);
      D("putting doubles");

    }

    nc_close(currentFileID);
    D("wrote %s\n", file_name);


    fca_free(msg);
  }

  fca_free(moduleNetCDFWriter);
  MPI_Finalize();
}
