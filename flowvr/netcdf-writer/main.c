#include <fca/fca.h>
#include "parflow_config.h"
#include "../../pfsimulator/parflow_lib/parflow.h"
#include "../../pfsimulator/parflow_lib/parflow_netcdf.h"
#include "../../pfsimulator/parflow_lib/GridMessage.h"
#include <netcdf_par.h>
#include <netcdf.h>
#include <stdio.h>
#include <assert.h>

#include "../../pfsimulator/amps/mpi1/amps.h"

#ifdef __DEBUG
#define D(x...) printf("++++++++ "); printf(x); printf("\n");
#else
#define D(x)
#endif

int CreateFile(const char* filename, int nX, int nY, int nZ, int *pxID, int *pyID, int *pzID, int *ptimeID) {
  int ncID = 0;
  if (nc_create_par(filename, NC_NETCDF4|NC_MPIIO, amps_CommWrite, MPI_INFO_NULL, &ncID) != NC_NOERR)
  {
    PARFLOW_ERROR("Could not create File!");
  }

  // Add Dimensions
  nc_def_dim(ncID, "x", nX, pxID);
  nc_def_dim(ncID, "y", nY, pyID);
  nc_def_dim(ncID, "z", nZ, pzID);
  nc_def_dim(ncID, "time", NC_UNLIMITED, ptimeID);  // unlimited dim should be first?! TODO?


  return ncID;
}

int main (int argc , char *argv [])
{
  printf("starting netcdf-writer\n");
  fca_module moduleNetCDFWriter = fca_new_empty_module();
  fca_port portPressureIn = fca_new_port("pressureIn", fca_IN, 0, NULL);

  fca_append_port(moduleNetCDFWriter, portPressureIn);

  fca_port outPort = fca_new_port("outPort", fca_OUT, 0, NULL);
  fca_append_port(moduleNetCDFWriter, outPort);

  const fca_stamp stampTime = fca_register_stamp(portPressureIn, "stampTime", fca_FLOAT);
  /*const fca_stamp stampN = fca_register_stamp(portPressureIn, "N", fca_INT);  // contains amount of cores that must be merged*/
  const fca_stamp stampFileNumber = fca_register_stamp(portPressureIn, "stampFileNumber",fca_INT);
//  const fca_stamp stampStartTime = fca_register_stamp(outPort, "stampStartTime", fca_FLOAT);
//  const fca_stamp stampStop = fca_register_stamp(outPort, "stampStopTime", fca_FLOAT);  // TODO good idea to use float? or should we put the double in the messages payload??

  if(!fca_init_module(moduleNetCDFWriter)){
    printf("ERROR : init_module failed!\n");
  }

  int it = 0;
  int currentFileID;
  int xID, yID, zID, timeID;
  D("nowWaiting\n");
  while (fca_wait(moduleNetCDFWriter)) {
    D("got some stuff to write\n"); // TODO: implement debug messages for more performance in release?
    ++it;
    fca_message msg = fca_get(portPressureIn);
    float *time = (float*)fca_read_stamp(msg, stampTime);
    int *filenumber = (int*)fca_read_stamp(msg, stampFileNumber);



    char file_name[1024];  // TODO: better filename generation!
    sprintf(file_name, "out%d.nc", it); // atm: create a new file for each message group.

    // to access the variables in the ncFile...
    int pressureVarID;
    int timeVarID;

    assert(fca_number_of_segments(msg)%2 == 0);
    for (int i = 0; i < fca_number_of_segments(msg); i+=2) {
      GridMessageMetadata* m = (GridMessageMetadata*)fca_get_read_access(msg, i);
      if (i == 0) {
        currentFileID = CreateFile(file_name, m->nX, m->nY, m->nZ, &xID, &yID, &zID, &timeID); // TODO: does not work when opening an existing file!
        // add variable Time and pressure:
        nc_def_var(currentFileID, "time", NC_DOUBLE, 1, &timeID, &timeVarID);
        int pressure_dims[4] = {timeID, zID, yID, xID}; // low: why in this order? I guess so it will count in the right order ;)
        nc_def_var(currentFileID, "pressure", NC_DOUBLE, 4, pressure_dims, &pressureVarID);
        D("Adding Time");

        size_t start[1], count[1];
        nc_var_par_access(currentFileID, timeVarID, NC_COLLECTIVE);
        find_variable_length(currentFileID, timeVarID, start);
        D("start writing time at %d\n", start[0]);
        count[0] = 1;  // writing one value
        int status = nc_put_vara_double(currentFileID, timeVarID, start, count, time);
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

    // one file per timestep...
    nc_close(currentFileID);
    D("wrote %s\n", file_name);


    fca_free(msg);

    // TODO send out new message with new time!
    // TODO: have iterator to give numbers to file/ create different file names!
  }


  fca_free(moduleNetCDFWriter);
}
