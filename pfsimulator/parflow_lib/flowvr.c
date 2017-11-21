#include "flowvr.h"
#include "GridMessage.h"
#include <fca/fca.h>

#include <string.h>  // for memcpy
#include <stdlib.h>  // for malloc

FLOWVR_EVENT_ACTIVE = 0;
FLOWVR_ACTIVE = 0;
fca_module moduleParflow;

static fca_module moduleParflowEvent;
static fca_port triggerSnapPort;

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


void NewFlowVR(void)
{
  // Refactor: shouldn't there be a GetBooleanDefault?
  NameArray switch_na = NA_NewNameArray("False True");
  char* switch_name = GetStringDefault("FlowVR", "False");

  FLOWVR_ACTIVE = NA_NameToIndex(switch_na, switch_name);
  if (FLOWVR_ACTIVE < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, "FlowVR");
    FLOWVR_ACTIVE = 0;
  }

  if (!FLOWVR_ACTIVE)
  {
    return;
  }

  switch_name = GetStringDefault("FlowVR.Event", "False");
  FLOWVR_EVENT_ACTIVE = NA_NameToIndex(switch_na, switch_name);
  if (FLOWVR_ACTIVE < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, "FlowVR.Event");
    FLOWVR_ACTIVE = 0;
  }

#ifndef HAVE_FLOWVR
  PARFLOW_ERROR("Parflow was not compiled with FlowVR but FlowVR was the input file was set to True");
  return;
#else
  D("Modname: %s, Parent: %s\n", getenv("FLOWVR_MODNAME"), getenv("FLOWVR_PARENT"));
  if (amps_size > 1)
  {
    fca_init_parallel(amps_rank, amps_size);  // TODO: amps size or amps_node_size
  }
  D("Modname: %s, Parent: %s\n", getenv("FLOWVR_MODNAME"), getenv("FLOWVR_PARENT"));
  const char* outportnamelist[] = {
    "pressure",
    "porosity",    // REM: does not really change..
    "saturation",
    "pressureSnap"
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

#define n_outportnamelist (sizeof(outportnamelist) / sizeof(const char *))


  moduleParflow = fca_new_empty_module();



  for (unsigned int i = 0; i < n_outportnamelist; ++i)
  {
    fca_port port = fca_new_port(outportnamelist[i], fca_OUT, 0, NULL);
    fca_register_stamp(port, "stampTime", fca_FLOAT);
    fca_register_stamp(port, "stampFileName", fca_STRING);

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
  fca_port beginItPort = fca_new_port("in", fca_IN, 0, NULL);
  fca_register_stamp(beginItPort, "stampStartTime", fca_FLOAT);
  fca_register_stamp(beginItPort, "stampStopTime", fca_FLOAT);
  // TODO ^^good idea to use float? or should we put the double in the messages payload??
  fca_append_port(moduleParflow, beginItPort);
  if (!fca_init_module(moduleParflow))
  {
    PARFLOW_ERROR("ERROR : init_module for moduleParflow failed!\n");
  }

  if (FLOWVR_EVENT_ACTIVE)
  {
    moduleParflowEvent = fca_new_empty_module();
    triggerSnapPort = fca_new_port("triggerSnap", fca_IN, fca_NON_BLOCKING, NULL);
    fca_append_port(moduleParflowEvent, triggerSnapPort);
    fca_set_modulename(moduleParflowEvent, "Event");
    if (!fca_init_module(moduleParflowEvent))
    {
      PARFLOW_ERROR("ERROR : init_module for moduleParflowEvent failed!\n");
    }
    // show that we are there. this call should be nonblocking as no blocking ports are connected
    fca_wait(moduleParflowEvent);
  }

  D("flowvr initialisiert.");
//  char modulename[256];
//
//  sprintf(modulename, "parflow/%d", amps_Rank(amps_CommWorld));
//
//  fca_set_modulename(moduleParflow, modulename);


  /*fca_trace testTrace = fca_get_trace(modulePut,"beginTrace");*/
  /*if(testTrace == NULL) printf("ERROR : Test Trace FAIL!!\n"); else printf("Test Trace OK.\n");*/
#endif
}

#ifdef HAVE_FLOWVR


int FlowVR_wait()
{
  if (FLOWVR_ACTIVE)
  {
    D("now waiting");
    if (FLOWVR_EVENT_ACTIVE)
      fca_wait(moduleParflowEvent);
    return fca_wait(moduleParflow);
  }
  else
    return 0;
}

void FreeFlowVR()
{
  if (!FLOWVR_ACTIVE)
    return;
  fca_free(moduleParflow);
  if (FLOWVR_EVENT_ACTIVE)
    fca_free(moduleParflowEvent);
}

void vectorToMessage(Vector* v, fca_message *result, fca_port *port)
{
  // normally really generic. low: in common with write_parflow_netcdf
  Grid *grid = VectorGrid(v);
  SubgridArray *subgrids = GridSubgrids(grid);
  Subvector *subvector;

  int g;

  ForSubgridI(g, subgrids)
  {
    subvector = VectorSubvector(v, g);
  }

  int nx_v = SubvectorNX(subvector);
  int ny_v = SubvectorNY(subvector);

  /*const fca_stamp stampMetadata = fca_get_stamp(*port, "Metadata");*/
  /*fca_write_stamp(result, stampMetadata, (void*) &stampMetadata);*/
  /*const fca_stamp stampN = fca_get_stamp(*port, "N");*/
  /*fca_write_stamp(result, stampN, 1);*/
  // write to the beginning of our memory segment
  GridMessageMetadata m;
  fillGridMessageMetadata(v, &m);
  size_t vector_size = sizeof(double) * m.nx * m.ny * m.nz;
  *result = fca_new_message(moduleParflow, sizeof(GridMessageMetadata) + vector_size);
  if (result == NULL)
  {
    D("Message_size: %d\n", sizeof(GridMessageMetadata) + vector_size);
    PARFLOW_ERROR("Could not create Message");
  }
  void *buffer = fca_get_write_access(*result, 0);
  D("Will write  %d bytes + %d bytes", sizeof(GridMessageMetadata), vector_size);

  memcpy(buffer, &m, sizeof(GridMessageMetadata));
  buffer += sizeof(GridMessageMetadata);


  double* buffer_double = (double*)buffer;

  double *data;
  data = SubvectorElt(subvector, m.ix, m.iy, m.iz);

  // some iterators
  int i, j, k, d = 0, ai = 0;
  BoxLoopI1(i, j, k, m.ix, m.iy, m.iz, m.nx, m.ny, m.nz, ai, nx_v, ny_v, nz_v, 1, 1, 1, { buffer_double[d] = data[ai]; d++; });
  // TODO: would be more performant if we could read the things not cell by cell I guess
}
// TODO: implement swap: do not do the memcpy but have to buffers one for read and wone for write. Change the buffers after one simulation step! (here a simulation step consists of multiple timesteps!

// Build data
typedef struct {
  const char *name;
  Vector const * const data;
} PortNameData;

void CreateAndSendMessage(SimulationSnapshot const * const snapshot, const PortNameData portnamedatas[], size_t n_portnamedata)
{
  for (unsigned int i = 0; i < n_portnamedata; ++i)
  {
    // Sometimes we do not have values for all the data...
    if (portnamedatas[i].data == NULL)
      continue;


    // Prepare the port
    fca_port port = fca_get_port(moduleParflow, portnamedatas[i].name); // TODO: maybe save all this in an array for faster acc

    // Prepare the Message
    fca_message msg;
    vectorToMessage(portnamedatas[i].data, &msg, &port);


    // Create Stamps...
    const fca_stamp stampTime = fca_get_stamp(port, "stampTime");
    fca_write_stamp(msg, stampTime, (void*)&snapshot->time);

    fca_stamp stampFileName;
    if (snapshot->filename != NULL)
    {
      stampFileName = fca_get_stamp(port, "stampFileName");
      fca_write_stamp(msg, stampFileName, (void*)snapshot->filename);
    }


    // Finally send message!
    if (!fca_put(port, msg))
    {
      PARFLOW_ERROR("Could not send FlowVR-Message!");
    }
    fca_free(msg);
    D("put message!%.8f\n", snapshot->time);
  }
}

// REM: we are better than the nodelevel netcdf feature because during file write the other nodes are already calculating ;)
// REM: structure of nodelevel netcdf: one process per node gathers everything that has to be written and does the filesystem i/o
void DumpRichardsToFlowVR(SimulationSnapshot const * const snapshot)
{
  if (!FLOWVR_ACTIVE)
    return;

  const PortNameData portnamedatas[] =
  {
    { "pressure", snapshot->pressure_out },
    { "porosity", snapshot->porosity_out },
    { "saturation", snapshot->saturation_out }
  };
#define n_portnamedata_ (sizeof(portnamedatas) / sizeof(const PortNameData))
  CreateAndSendMessage(snapshot, portnamedatas, n_portnamedata_);
}

void FlowVRSendSnapshot(SimulationSnapshot const * const snapshot)
{
  if (!FLOWVR_ACTIVE || !FLOWVR_EVENT_ACTIVE)
    return;

  // TODO: do we have to call fca_wait before we can receive the next nonblocking message?
  // TODO: wenn wait noetig: brauchen nen extra modul! weils sonst ja immer blockt :(
  if (!fca_wait(moduleParflowEvent))
    return;                                   // something bad happened... TODO: debug the case...
  fca_message msg = fca_get(triggerSnapPort);
  size_t s = fca_get_segment_size(msg, 0);
  fca_free(msg);
  if (s == 0)
    return;
  D("Got a trigger!");

  // send snapshot!
  const PortNameData portnamedatas[] =
  {
    { "pressureSnap", snapshot->pressure_out }//,
    /*{ "porosity", porosity_out },*/
    /*{ "saturation", saturation_out }*/
  };
#define n_portnamesnapdata (sizeof(portnamedatas) / sizeof(const PortNameData))
  CreateAndSendMessage(snapshot, portnamedatas, n_portnamesnapdata);
}

void FlowVRServeFinalState(SimulationSnapshot const * const snapshot)
{
  NameArray switch_na = NA_NewNameArray("False True");
  char* switch_name = GetStringDefault("FlowVR.ServeFinalState", "False");


  int serve_final_state = NA_NameToIndex(switch_na, switch_name);

  if (serve_final_state < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, "FlowVR.ServeFinalState");
    serve_final_state = 0;
  }

  if (serve_final_state)
  {
    while (1)
    {
      FlowVRSendSnapshot(snapshot);
      usleep(100000);
    }
  }
}

#endif
