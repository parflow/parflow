/*BHEADER*********************************************************************
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
#include "flowvr.h"

#include <string.h>  // for memcpy
#include <stdlib.h>  // for malloc

#include "globals.h"


int FLOWVR_ACTIVE;

#ifdef HAVE_FLOWVR
static fca_module module_parflow;
static fca_port port_in;

void fillGridDefinition(Grid const * const grid, GridDefinition *grid_def)
{
  grid_def->nX = SubgridNX(GridBackground(grid));
  grid_def->nY = SubgridNY(GridBackground(grid));
  grid_def->nZ = SubgridNZ(GridBackground(grid));
}

void fillGridMessageMetadata(Vector const * const v, double const * const time,
                             const Variable variable, GridMessageMetadata *m)
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

  m->grid.nX = SubgridNX(GridBackground(grid));
  m->grid.nY = SubgridNY(GridBackground(grid));
  m->grid.nZ = SubgridNZ(GridBackground(grid));

  m->time = *time;
  m->variable = variable;

  strcpy(m->run_name, GlobalsRunName);
}

typedef struct {
  const char * port_name;
  Variable variable;
  int offset;
  int periodicity;
} Contract;

Contract *contracts;
size_t n_contracts;

/**
 * Will init out ports and create contracts from it.
 */
void initContracts()
{
  NameArray port_names = NA_NewNameArray(GetStringDefault("FlowVR.Outports.Names", ""));

  n_contracts = NA_Sizeof(port_names);
  contracts = ctalloc(Contract, n_contracts);

  char key[256];
  for (size_t i = 0; i < n_contracts; ++i)
  {
    contracts[i].port_name = NA_IndexToName(port_names, i);
    sprintf(key, "FlowVR.Outports.%s.Periodicity", contracts[i].port_name);
    contracts[i].periodicity = GetInt(key);
    sprintf(key, "FlowVR.Outports.%s.Offset", contracts[i].port_name);
    contracts[i].offset = GetInt(key);
    sprintf(key, "FlowVR.Outports.%s.Variable", contracts[i].port_name);
    contracts[i].variable = NameToVariable(GetString(key));

    D("Add Contract %s executed with periodicity: %d (offset: %d)",
      contracts[i].port_name, contracts[i].periodicity, contracts[i].offset);
    // REM we do not need to free Strings obtained by GetString as they are just requestet from a database that was already cached.
  }
}

typedef enum {
  STEER_LOG_NONE = 0,
  STEER_LOG_VERY_SIMPLE,
  STEER_LOG_SIMPLE,
  STEER_LOG_FULL
} SteerLogMode;
static SteerLogMode steer_log_mode;

void initSteerLogMode(void)
{
  NameArray switch_na = NA_NewNameArray("None VerySimple Simple Full");
  char* switch_name = GetStringDefault("FlowVR.SteerLogMode", "None");

  steer_log_mode = NA_NameToIndex(switch_na, switch_name);
  if (steer_log_mode < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, "FlowVR.SteerLogMode");
    steer_log_mode = STEER_LOG_NONE;
  }
  NA_FreeNameArray(switch_na);
  D("steer_log_mode: %d", steer_log_mode);
}
#endif

void NewFlowVR(void)
{
  FLOWVR_ACTIVE = GetBooleanDefault("FlowVR", 0);

  if (!FLOWVR_ACTIVE)
  {
    return;
  }

#ifndef HAVE_FLOWVR
  PARFLOW_ERROR("Parflow was not compiled with FlowVR but FlowVR in the input file was set to True");
  return;
#else
  if (strcmp(GetString("Solver"), "Richards") != 0)
  {
    PARFLOW_ERROR("To use as parflow module in parFlowVR the Richards solver must be chosen!");
    return;
  }
  initSteerLogMode();
  initContracts();
  D("Modname: %s, Parent: %s\n", getenv("FLOWVR_MODNAME"), getenv("FLOWVR_PARENT"));
  if (amps_size > 1)
  {
    fca_init_parallel(amps_Rank(amps_CommWorld), amps_Size(amps_CommWorld));
  }
  D("Modname: %s, Parent: %s\n", getenv("FLOWVR_MODNAME"), getenv("FLOWVR_PARENT"));
  /*"pressure",*/
  /*"porosity",    // REM: does not really change..*/
  /*"saturation",*/
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


  module_parflow = fca_new_empty_module();

  fca_port port_pressure_snap = fca_new_port("snap", fca_OUT, 0, NULL);
  fca_register_stamp(port_pressure_snap, "stampTime", fca_FLOAT);
  fca_register_stamp(port_pressure_snap, "stampFileName", fca_STRING);
  fca_append_port(module_parflow, port_pressure_snap);

  for (size_t i = 0; i < n_contracts; ++i)
  {
    D("Add outport %s", contracts[i].port_name);
    fca_port port = fca_new_port(contracts[i].port_name, fca_OUT, 0, NULL);
    fca_register_stamp(port, "stampTime", fca_FLOAT);
    fca_register_stamp(port, "stampFileName", fca_STRING);

    fca_append_port(module_parflow, port);
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


  port_in = fca_new_port("in", fca_IN, 0, NULL);
  fca_append_port(module_parflow, port_in);
  if (!fca_init_module(module_parflow))
  {
    PARFLOW_ERROR("ERROR : init_module for module_parflow failed!\n");
  }

  D("flowvr initialized.");

  /*fca_trace testTrace = fca_get_trace(modulePut,"beginTrace");*/
  /*if(testTrace == NULL) printf("ERROR : Test Trace FAIL!!\n"); else printf("Test Trace OK.\n");*/
#endif
}

#ifdef HAVE_FLOWVR

static void* translation[VARIABLE_LAST];
void FlowVRInitTranslation(SimulationSnapshot *snapshot)
{
  translation[VARIABLE_PRESSURE] = snapshot->pressure_out;
  translation[VARIABLE_SATURATION] = snapshot->saturation_out;
  translation[VARIABLE_POROSITY] = ProblemDataPorosity(snapshot->problem_data);
  translation[VARIABLE_MANNING] = ProblemDataMannings(snapshot->problem_data);
  translation[VARIABLE_PERMEABILITY_X] = ProblemDataPermeabilityX(snapshot->problem_data);
  translation[VARIABLE_PERMEABILITY_Y] = ProblemDataPermeabilityY(snapshot->problem_data);
  translation[VARIABLE_PERMEABILITY_Z] = ProblemDataPermeabilityZ(snapshot->problem_data);
}

static inline int simple_intersect(int ix1, int nx1, int ix2, int nx2,
                                   int iy1, int ny1, int iy2, int ny2,
                                   int iz1, int nz1, int iz2, int nz2)
{
  int d;

  d = ix2 - ix1;
  if (-nx2 > d || d > nx1)
    return 0;
  d = iy2 - iy1;
  if (-ny2 > d || d > ny1)
    return 0;
  d = iz2 - iz1;
  if (-nz2 > d || d > nz1)
    return 0;
  return 1;
}

static inline LogSteer(Variable var, Action action, SteerMessageMetadata * s,
                       double *operand, double const * const ptime)
{
  ////////////////////////////////////////////////////////
  // Log:
  if (steer_log_mode != STEER_LOG_NONE)
  {
    const char * action_text;
    switch (action)
    {
      case ACTION_SET:
        action_text = "SET";
        break;

      case ACTION_ADD:
        action_text = "ADD";
        break;

      case ACTION_MULTIPLY:
        action_text = "MUL";
        break;

      default:
        PARFLOW_ERROR("unknown Steer Action!");
    }
    switch (steer_log_mode)
    {
      case STEER_LOG_VERY_SIMPLE:
        printf("Steer at %f: %s on %s\n", *ptime, action_text,
               VARIABLE_TO_NAME[var]);
        break;

      case STEER_LOG_SIMPLE:
        printf("Steer at %f: %s on %s\nfrom (%d, %d, %d): (%d, %d, %d) counts.\n",
               *ptime, action_text, VARIABLE_TO_NAME[var],
               s->ix, s->iy, s->iz, s->nx, s->ny, s->nz);
        break;

      case STEER_LOG_FULL:
        printf("Steer at %f: %s on %s\nfrom (%d, %d, %d): (%d, %d, %d) counts.\n",
               *ptime, action_text, VARIABLE_TO_NAME[var],
               s->ix, s->iy, s->iz, s->nx, s->ny, s->nz);
        for (size_t i = 0; i < s->nx * s->ny * s->nz; ++i)
        {
          if (i % s->nx == 0)
            printf("[");
          if (i % (s->nx * s->ny) == 0)
            printf("[");
          if (i % (s->nx * s->ny * s->nz) == 0)
            printf("[");
          printf("%f,", *(operand++));
          if ((i + 1) % s->nx == 0)
            printf("],");
          if ((i + 1) % (s->nx * s->ny) == 0)
            printf("],\n");
          if ((i + 1) % (s->nx * s->ny * s->nz) == 0)
            printf("],\n\n");
        }
        break;
    }
  }
}

/// returns how much we read from buffer
size_t Steer(Variable var, Action action, const void *buffer, double const * const ptime)
{
  D("Steer");
  SteerMessage sm = ReadSteerMessage(buffer);

  Vector *v = translation[var];

  Grid *grid = VectorGrid(v);
  SubgridArray *subgrids = GridSubgrids(grid);
  Subvector *subvector;
  Subgrid *subgrid;
  int g;

  ForSubgridI(g, subgrids)
  {
    subvector = VectorSubvector(v, g);
    subgrid = SubgridArraySubgrid(subgrids, g);
  }

  int nx_v = SubvectorNX(subvector);
  int ny_v = SubvectorNY(subvector);

  // TODO:speedoptimize the BoxLoop! loop. switch to outside. maybe I can do it with avx on whole oxes? maybe I have to change 1,1,1
  // probably one can use IntersectSubgrids and loop only over intersction! Maybe this could influence the vectorupdate too!
  int ix = SubgridIX(subgrid);
  int iy = SubgridIY(subgrid);
  int iz = SubgridIZ(subgrid);

  int nx = SubgridNX(subgrid);
  int ny = SubgridNY(subgrid);
  int nz = SubgridNZ(subgrid);

  int i, j, k, ai = 0;
  double *data;
  data = SubvectorElt(subvector, ix, iy, iz);

  size_t read_out_size = sizeof(SteerMessageMetadata) + sizeof(double) * sm.m->nx * sm.m->ny * sm.m->nz;

  // Check if box in this thread! only then start box loop!
  if (!simple_intersect(ix, nx, sm.m->ix, sm.m->nx,
                        iy, ny, sm.m->iy, sm.m->ny,
                        iz, nz, sm.m->iz, sm.m->nz))
  {
    D("No intersect found!");
    return read_out_size;
  }

  BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz, ai, nx_v, ny_v, nz_v, 1, 1, 1, {
    int xn = i - sm.m->ix;
    int yn = j - sm.m->iy;
    int zn = k - sm.m->iz;
    // is the requested point in this chunk?
    if (xn < 0 || yn < 0 || zn < 0)
      continue;                                  // too small.
    if (xn >= sm.m->nx || yn >= sm.m->ny || zn >= sm.m->nz)
      continue;                                              // too big.

    size_t index = xn + yn * sm.m->nx + zn * sm.m->nx * sm.m->ny;
    switch (action)
    {
      case ACTION_SET:
        /*printf("err? %.12e == %.12e\n", data[ai], sm.data[index]);*/
        data[ai] = sm.data[index];
        break;

      case ACTION_ADD:
        data[ai] += sm.data[index];
        break;

      case ACTION_MULTIPLY:
        data[ai] *= sm.data[index];
        break;

      default:
        PARFLOW_ERROR("unknown Steer Action!");
    }
  });

  // InitVectorUpdate!
  // TODO: necessary?:
  VectorUpdateCommHandle *handle;
  handle = InitVectorUpdate(v, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  D("Applied SendSteerMessage (%d) + %d + %d", sizeof(ActionMessageMetadata), sizeof(SteerMessageMetadata), sizeof(double) * sm.m->nx * sm.m->ny * sm.m->nz);
  LogSteer(var, action, sm.m, sm.data, ptime);
  return read_out_size;
}


void SendGridDefinition(SimulationSnapshot const * const snapshot)
{
  fca_message msg = fca_new_message(module_parflow, sizeof(GridDefinition));
  GridDefinition *g = (GridDefinition*)fca_get_write_access(msg, 0);

  fillGridDefinition(snapshot->grid, g);
  fca_put(fca_get_port(module_parflow, "snap"), msg);
  fca_free(msg);
}


/// returns how much we read from buffer
MergeMessageParser(Interact)
{
  D("Interact %d > %d ?", size, sizeof(ActionMessageMetadata));

  if (size < sizeof(ActionMessageMetadata))
    return size;  // does not contain an action.

  SimulationSnapshot *snapshot = (SimulationSnapshot*)cbdata;
  ActionMessage am = ReadActionMessage(buffer);
  size_t s = sizeof(ActionMessageMetadata);

  switch (am.m->action)
  {
    case ACTION_GET_GRID_DEFINITION:
      SendGridDefinition(snapshot);
      // s+= 0
      break;

    case ACTION_TRIGGER_SNAPSHOT:
      SendSnapshot(snapshot, am.m->variable);
      // s += 0;
      break;

    case ACTION_SET:
    case ACTION_ADD:
    case ACTION_MULTIPLY:
      s += Steer(am.m->variable, am.m->action, am.data, snapshot->time);
      break;

    default:
      PARFLOW_ERROR("TODO: Unimplemented Probably somewhere adding the wrong size to s!");
  }
  /*D("processed %d / %d", s, size);*/
  return s;
}

/**
 * Executes a flowvr wait. Does the requested changes on the simulation state.
 * Returns 0 if abort was requested.
 */
int FlowVRInteract(SimulationSnapshot *snapshot)
{
  if (FLOWVR_ACTIVE)
  {
    D("now waiting");
    BeginTiming(FlowVRWaitTimingIndex);
    int wait_res = fca_wait(module_parflow);
    EndTiming(FlowVRWaitTimingIndex);
    if (!wait_res)
      return 0;

    D("now simulating");
    // Read out message on in port.
    // Do all actions that are listed there(steerings, trigger snaps...)
    ParseMergedMessage(port_in, Interact, (void*)snapshot);
  }
  return 1;
}

void FreeFlowVR()
{
  if (!FLOWVR_ACTIVE)
    return;

  fca_free(module_parflow);
  tfree(contracts);
}

static inline void vectorToMessage(const Variable variable, double const * const time, fca_message *result, fca_port *port)
{
  Vector *v = translation[variable];
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

  // write to the beginning of our memory segment
  GridMessageMetadata m;
  fillGridMessageMetadata(v, time, variable, &m);
  size_t vector_size = sizeof(double) * m.nx * m.ny * m.nz;
  D("Sending Vector %d %d %d at t = %f", m.nx, m.ny, m.nz, *time);
  *result = fca_new_message(module_parflow, sizeof(GridMessageMetadata) + vector_size);
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
// TODO: implement swap: do not do the memcpy but have two buffers one for read and one for write. Change the buffers after one simulation step! (here a simulation step consists of multiple timesteps!


void CreateAndSendMessage(SimulationSnapshot const * const snapshot, const char * portname, Variable var)
{
  // Prepare the port
  fca_port port = fca_get_port(module_parflow, portname);
  // Performance: maybe save all the ports in an array for faster access times

  // Prepare the Message
  fca_message msg;

  vectorToMessage(var, snapshot->time, &msg, &port);


  // Create Stamps...
  const fca_stamp stamp_time = fca_get_stamp(port, "stampTime");
  float time = (float)*(snapshot->time);
  fca_write_stamp(msg, stamp_time, (void*)&time);

  fca_stamp stamp_file_name;
  char filename[128];
  int user_spec_steps = GetIntDefault("FlowVR.NumStepsPerFile", 1);
  int number = 1 + (*snapshot->file_number - 1) / user_spec_steps;

  // Handle special case
  if (*snapshot->file_number == 0)
  {
    number = 0;
  }

  sprintf(filename, "%s.%05d", GlobalsOutFileName, number);

  stamp_file_name = fca_get_stamp(port, "stampFileName");
  fca_write_stamp(msg, stamp_file_name, (void*)filename);


  // Finally send message!
  if (!fca_put(port, msg))
  {
    PARFLOW_ERROR("Could not send FlowVR-Message!");
  }
  fca_free(msg);
  D("put message!%.8f\n", *(snapshot->time));
}

// REM: we are better than the nodelevel netcdf feature because during file write the other nodes are already calculating ;)
// REM: structure of nodelevel netcdf: one process per node gathers everything that has to be written and does the filesystem i/o

void SendSnapshot(SimulationSnapshot const * const snapshot, Variable var)
{
  D("SendSnapshot");
  CreateAndSendMessage(snapshot, "snap", var);
}

void serveFinalState(SimulationSnapshot *snapshot)
{
  D("now serving final state.");
  while (FlowVRInteract(snapshot))
    usleep(100000);
}

/**
 * Dumps data if it has to since a contract.
 */
int FlowVRFulFillContracts(int timestep, SimulationSnapshot const * const snapshot)
{
  int res = 0;

  for (size_t i = 0; i < n_contracts; ++i)
  {
    if (timestep >= contracts[i].offset &&
        ((timestep - contracts[i].offset) % contracts[i].periodicity) == 0)
    {
      res = 1;
      CreateAndSendMessage(snapshot, contracts[i].port_name, contracts[i].variable);
    }
  }
  return res;
}

void sendEmpties()
{
  for (size_t i = 0; i < n_contracts; ++i)
  {
    fca_port port = fca_get_port(module_parflow, contracts[i].port_name);
    SendEmptyMessage(module_parflow, port);
  }
}

void FlowVREnd(SimulationSnapshot *snapshot)
{
  NameArray switch_na = NA_NewNameArray("Abort SendEmpty ServeFinalState");
  const char *key = "FlowVR.OnEnd";

  char *switch_name = GetStringDefault(key, "Abort");

  int on_end = NA_NameToIndex(switch_na, switch_name);

  NA_FreeNameArray(switch_na);

  switch (on_end)
  {
    case 0:
      fca_abort(module_parflow);
      break;

    case 1:
      sendEmpties();
      break;

    case 2:
      serveFinalState(snapshot);
      break;

    default:
      InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
                 key);
  }
}

#endif
