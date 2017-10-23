#include "flowvr.h"
#include "GridMessage.h"
#include <fca/fca.h>

#include <string.h>  // for memcpy
#include <stdlib.h>  // for malloc

void fillGridMessageMetadata(Vector const * const v, GridMessageMetadata *m)
{
  Grid *grid = VectorGrid(v);
  SubgridArray *subgrids = GridSubgrids(grid);
  Subgrid *subgrid;
  int g;
  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
  }

  m->ix = SubgridIX(subgrid);
  m->iy = SubgridIY(subgrid);
  m->iz = SubgridIZ(subgrid);

  m->nx = SubgridNX(subgrid);
  m->ny = SubgridNY(subgrid);
  m->nz = SubgridNZ(subgrid);

  m->nX = SubgridNX(GridBackground(grid));
  m->nY = SubgridNY(GridBackground(grid));
  m->nZ = SubgridNZ(GridBackground(grid));
}


void initFlowVR()
{ // TODO call it from the right place!
  const char* outportnamelist[] = {
    "pressure",
    "porosity",
    "saturation"
      /*"subsurf_data",         [> permeability/porosity <]*/
      /*"press",                [> pressures <]*/
      /*"slopes",               [> slopes <]*/
      /*"mannings",             [> mannings <]*/
      /*"top",                  [> top <]*/
      /*"velocities",           [> velocities <]*/
      /*"satur",                [> saturations <]*/
      /*"mask",                 [> mask <]*/
      /*"concen",               [> concentrations <]*/
      /*"wells",                [> well data <]*/
      /*"dzmult",               [> dz multiplier<]*/
      /*"evaptrans",            [> evaptrans <]*/
      /*"evaptrans_sum",        [> evaptrans_sum <]*/
      /*"overland_sum",         [> overland_sum <]*/
      /*"overland_bc_flux"      [> overland outflow boundary condition flux <]*/
      // TODO: ask BH: porosity is missing?
  };

#define n_outportnamelist (sizeof (outportnamelist) / sizeof (const char *))


  moduleParflow = fca_new_empty_module();

  for (unsigned int i = 0; i < n_outportnamelist; ++i)
  {
    fca_port port = fca_new_port(outportnamelist[i], fca_OUT, 0, NULL);
    fca_register_stamp(port, "stampTime", fca_FLOAT);
    fca_register_stamp(port, "stampFileNumber", fca_INT);
    /*fca_register_stamp(port, "stampMetadata", fca_BINARY, sizeof(GridMessageMetadata));*/
//    fca_register_stamp(port, "N", fca_INT);  // always 1 to count in merge ;)

    fca_append_port(moduleParflow, port);
  }
  // low: name ports? pressureOut...
  /*fca_trace trace = fca_new_trace("beginTrace", fca_trace_INT, NULL);*/
  /*if(trace != NULL) printf("Creation of trace succeded.\n"); else printf("Failed to create a trace.\n");*/

  /*fca_trace trace2 = fca_new_trace("endTrace", fca_trace_INT, NULL);*/
  /*if(trace2 != NULL) printf("Creation of trace succeded.\n"); else printf("Failed to create a trace.\n");*/

  /*fca_module modulePut = fca_new_empty_module();*/
  /*fca_append_port(modulePut, portText);*/
  /*fca_append_trace(modulePut, trace);*/
  /*fca_append_trace(modulePut, trace2);*/


  // in-port beginPort
  fca_port beginPort = fca_new_port("beginPort", fca_IN, 0, NULL);
  fca_register_stamp(beginPort, "stampStartTime", fca_FLOAT);
  fca_register_stamp(beginPort, "stampStopTime", fca_FLOAT);  // TODO good idea to use float? or should we put the double in the messages payload??
  fca_append_port(moduleParflow, beginPort);

  if(!fca_init_module(moduleParflow)){
    PARFLOW_ERROR("ERROR : init_module failed!\n");
  }

  /*fca_trace testTrace = fca_get_trace(modulePut,"beginTrace");*/
  /*if(testTrace == NULL) printf("ERROR : Test Trace FAIL!!\n"); else printf("Test Trace OK.\n");*/
}

void freeFlowVR()
{
  fca_free(moduleParflow);
}

void vectorToMessage(Vector* v, fca_message *result, fca_port *port) {
  // normally really generic. low: in common with write_parflow_netcdf
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

  *result = fca_new_message(moduleParflow, sizeof(GridMessageMetadata) + sizeof(double)*nx*ny*nz);
  /*const fca_stamp stampMetadata = fca_get_stamp(*port, "Metadata");*/
  /*fca_write_stamp(result, stampMetadata, (void*) &stampMetadata);*/
  /*const fca_stamp stampN = fca_get_stamp(*port, "N");*/
  /*fca_write_stamp(result, stampN, 1);*/
  // write to the beginning of our memory segment
  GridMessageMetadata* m = (GridMessageMetadata*) fca_get_write_access(*result, 0);
  fillGridMessageMetadata(v, m);

  double* buffer = (double*) fca_get_write_access(*result, sizeof(GridMessageMetadata));

  double *data;
  data = SubvectorElt(subvector, m->ix, m->iy, m->iz);

  // some iterators
  int i, j, k, d = 0, ai = 0;
  BoxLoopI1(i, j, k, m->ix, m->iy, m->iz, m->nx, m->ny, m->nz, ai, nx_v, ny_v, nz_v, 1, 1, 1,{ buffer[d] = data[ai]; d++;});
  // TODO: would be more performant if we could read the things not cell by cell I guess
}
// TODO: implement swap: do not do the memcpy but have to buffers one for read and wone for write. Change the buffers after one simulation step! (here a simulation step consists of multiple timesteps!


// REM: We are better than the nodelevel netcdf feature because during file write the other nodes are already calculating ;)
// REM: structure of nodelevel netcdf: one process per node gathers everything that has to be written and does the filesystem i/o
void dumpRichardsToFlowVR(float time, Vector const * const pressure_out,
    Vector const * const porosity_out, Vector const * const saturation_out)
{

  static int filenumber = 0;
  int it=0; // TODO: unused?


  // Build data
  typedef struct
  {
    const char *name;
    Vector const * const data;
  } PortNameData;

  const PortNameData portnamedatas[] =
  {
    {"pressure",  pressure_out},
    {"porosity",  porosity_out},
    {"saturation",  saturation_out}
  };
#define n_portnamedata (sizeof (portnamedatas) / sizeof (const PortNameData))

  // TODO: write reasonable values into stamps!
  for (unsigned int i = 0; i < n_portnamedata; ++i)
  {


    // Prepare the port
    fca_port port = fca_get_port(moduleParflow, portnamedatas[i].name); // TODO: maybe save all this in an array for faster acc

    // Prepare the Message
    fca_message msg;
    vectorToMessage(portnamedatas[i].data, &msg, &port);


    // TODO: do I need to send the rank too? and some other metadata like nx... (see vectorToMessage)
    const fca_stamp stampTime = fca_get_stamp(port, "stampTime");
    const fca_stamp stampFileNumber = fca_get_stamp(port, "stampFileNumber");
    fca_write_stamp(msg, stampTime, (void*) &time);
    fca_write_stamp(msg, stampFileNumber, (void*) &filenumber);

    // finally send message!
    if(!fca_put(port,  msg))
    {
      PARFLOW_ERROR("Could not send FlowVR-Message!");
    }

    //fca_free(buffer);  // TODO: do we really have to do this? I guess no. Example shows that it should be fine to free messages.

    // TODO: test with silo/hdf5 writer output...

  }

  //	++it; TODO: unused?
}


