#include <fca/fca.h>
#include "../../pfsimulator/parflow_lib/parflow.h"
#include "../../pfsimulator/parflow_lib/parflow_netcdf.h"
#include "../../pfsimulator/parflow_lib/GridMessage.h"

void writeTime(double *time, int ncID) {
  long end[MAX_NC_VARS];
  varNCData *myVarNCData;
  int myVarID=LookUpInventory("time", &myVarNCData);  //TODO: set variable!
  size_t start[myVarNCData->dimSize], count[myVarNCData->dimSize];
  nc_var_par_access(ncID, myVarID, NC_COLLECTIVE);
  find_variable_length(ncID, myVarID, end);
  start[0] = end[0]; count[0] = 1;
  //start[0] = counter; count[0] = 1;
  int status = nc_put_vara_double(ncID, myVarID, start, count, time);
}

int main (int argc , char *argv [])
{
  // TODO

  // - Premessage sends out start message with standard timing ;)
  // - get results
  // - write them to file
  // quit
  //
  fca_module moduleNetCDFWriter = fca_new_empty_module();
  fca_port portPressureIn = fca_new_port("pressureIn", fca_IN, 0, NULL);

  fca_append_port(moduleNetCDFWriter, portPressureIn);
  fca_port outPort = fca_new_port("outPort", fca_OUT, 0, NULL);
  fca_append_port(moduleNetCDFWriter, outPort);

  const fca_stamp stampTime = fca_register_stamp(portPressureIn, "stampTime", fca_FLOAT);
  /*const fca_stamp stampN = fca_register_stamp(portPressureIn, "N", fca_INT);  // contains amount of cores that must be merged*/
  const fca_stamp stampFileNumber = fca_register_stamp(portPressureIn, "stampFileNumber",fca_INT);
  const fca_stamp stampStartTime = fca_register_stamp(outPort, "stampStartTime", fca_FLOAT);
  const fca_stamp stampStop = fca_register_stamp(outPort, "stampStopTime", fca_FLOAT);  // TODO good idea to use float? or should we put the double in the messages payload??


  while (fca_wait(moduleNetCDFWriter)) {
    fca_message msg = fca_get(portPressureIn);
    float *time = (float*)fca_read_stamp(msg, stampTime);
    int *filenumber = (int*)fca_read_stamp(msg, stampFileNumber);

    int ncID;


    long end[MAX_NC_VARS];


    const char* file_name = "out.nc"; // TODO: generate more intelligently
    size_t pos = 0;
    int myVarID;
    do {
      GridMessageMetadata* m = (GridMessageMetadata*)fca_get_read_access(msg, pos);
      varNCData *myVarNCData;
      if (pos == 0) {
        CreateNCFileNode(file_name, m->nX, m->nY, m->nZ, &ncID);
        writeTime(time, ncID);
        int myVarID=LookUpInventory("pressure", &myVarNCData);  //TODO: set variable!
      }

      nc_var_par_access(ncID, myVarID, NC_COLLECTIVE);
      find_variable_length(ncID, myVarID, end);
      size_t start[myVarNCData->dimSize], count[myVarNCData->dimSize];
      start[0] = end[0]-1;
      count[0] = 1;
      start[1] = m->iz; start[2] =m->iy; start[3] = m->ix;
      count[1] = m->nz; count[2] =m->ny; count[3] = m->nx;
      pos += sizeof(GridMessageMetadata);
      double const * const data = (double*)fca_get_read_access(msg, pos);
      // now do a write to the cdf! (! all the preparations up there are necessary!
      int status = nc_put_vara_double(ncID, myVarID, start, count, data);

      pos += m->nx * m->ny * m->nz;
    } while( pos < fca_get_segment_size(msg, 0) );

    // one file per timestep...
    nc_close(ncID);




    /*static int numStepsInFile = 1;*/
    /*int userSpecSteps = GetInt("NetCDF.NumStepsPerFile");*/
    /*static char file_name[255];*/
    /*static int numOfDefVars=0;*/


    //    all the steps go into one file :P

    varNCData *myVarNCData;


    // we are working directly with file nodes as this should be faster in the end when more than one process will be writing at the same time


    fca_free(msg);

    // TODO send out new message with new time!
    // TODO: have iterator to give numbers to file/ create different file names!
  }



}
