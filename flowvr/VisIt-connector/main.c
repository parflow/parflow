#include <fca/fca.h>
#include "parflow_config.h"
#include <messages.h>
#include <VisItControlInterface_V2.h>
#include <VisItDataInterface_V2.h>
#include <stdio.h>
#include <string.h>
#ifdef __DEBUG
#include <stdlib.h>
#endif
#include <math.h>

#include <assert.h>


#ifdef __DEBUG
#define D(x ...) printf("oooooooo "); printf(x); printf(" %s:%d\n", __FILE__, __LINE__)
#else
#define D(...)
#endif




/* Data Access Function prototypes */
visit_handle SimGetMetaData(void *);
visit_handle SimGetMesh(int, const char *, void *);
visit_handle SimGetVariable(int, const char *, void *);

/******************************************************************************
* Simulation data and functions
******************************************************************************/

typedef struct {
  int cycle;
  double time;
  double* snapshot;
  int nX;
  int nY;
  int nZ;
  float *mesh_x;
  float *mesh_y;
  float *mesh_z;
  int inited;
} SimulationData;

const char *cmd_names[] = { "trigger snap" };

static fca_module flowvr;
static fca_stamp stampTime;
static fca_port portPressureIn;
static fca_port portTriggerSnap;


void FreeSim(SimulationData *sim)
{
  if (sim->inited)
  {
    free(sim->snapshot);
    free(sim->mesh_x);
    free(sim->mesh_y);
    free(sim->mesh_z);
    //sim->inited = 0;  unused atm
  }
}

void CreateMesh(SimulationData *sim)
{
  sim->mesh_x = (float*)malloc(sim->nX * sizeof(float));
  sim->mesh_y = (float*)malloc(sim->nY * sizeof(float));
  sim->mesh_z = (float*)malloc(sim->nZ * sizeof(float));

  // populate mesh:
  for (int i = 0; i < sim->nX || i < sim->nY || i < sim->nZ; ++i)
  {
    if (i < sim->nX)
      sim->mesh_x[i] = (float)i;
    if (i < sim->nY)
      sim->mesh_y[i] = (float)i;
    if (i < sim->nZ)
      sim->mesh_z[i] = (float)i;
  }
}

void initSim(SimulationData *sim, GridDefinition const *const grid)
{
  D("init sim on grid: %d %d %d", grid->nX, grid->nY, grid->nZ);

  FreeSim(sim);  // try to free sim

  sim->snapshot = (double*)malloc(sizeof(double) * grid->nX * grid->nY * grid->nZ);

  sim->nX = grid->nX;
  sim->nY = grid->nY;
  sim->nZ = grid->nZ;

  // recreate mesh
  CreateMesh(sim);

  sim->inited = 1;
}

MergeMessageParser(setSnapshot)
{
  GridMessageMetadata* m = (GridMessageMetadata*)buffer;
  SimulationData *sim = (SimulationData*)cbdata;

  assert(m->variable == VARIABLE_PRESSURE);  // low: atm we can only show pressures ;)

  sim->time = m->time;

  if (!sim->inited || sim->nX != m->grid.nX || sim->nY != m->grid.nY || sim->nZ != m->grid.nZ)
  {
    initSim(sim, &(m->grid));
  }


  // populate snapshot:
  buffer += sizeof(GridMessageMetadata);
  const double * data = (double*)buffer;
  for (int z = 0; z < m->nz; ++z)
  {
    for (int y = 0; y < m->ny; ++y)
    {
      int snapindex = m->ix + (y + m->iy) * sim->nX + (z + m->iz) * sim->nX * sim->nY;
      memcpy(sim->snapshot + snapindex, data, m->nx * sizeof(double));
      data += m->nx;
    }
  }

  D("copied buffers!");
  return m->nx * m->ny * m->nz * sizeof(double) + sizeof(GridMessageMetadata);
}

void triggerSnap(SimulationData *sim)
{
  D("triggerSnap");
  // TODO: allow to access more than just pressure
  SendActionMessage(flowvr, portTriggerSnap, ACTION_TRIGGER_SNAPSHOT,
                    VARIABLE_PRESSURE, NULL, 0);

  D("waiting...");
  fca_wait(flowvr);
  D("got an answer");

  // low: we could play with lambdas here when we were using cpp ;)
  ParseMergedMessage(portPressureIn, setSnapshot, (void*)sim);
}


MergeMessageParser(setGrid)
{
  GridDefinition *grid = (GridDefinition*)buffer;
  SimulationData *sim = (SimulationData*)cbdata;

  initSim(sim, grid);
  return sizeof(GridDefinition);
}

void wait_for_init(SimulationData *sim)
{
  if (sim->inited)
  {
    return;
  }
  else
  {
    SendActionMessage(flowvr, portTriggerSnap, ACTION_GET_GRID_DEFINITION, -1, NULL, 0);

    fca_wait(flowvr);

    // low: we could play with lambdas here when we were using cpp ;)
    ParseMergedMessage(portPressureIn, setGrid, (void*)sim);
  }
}

void ControlCommandCallback(const char *cmd, const char *args, void *cbdata)
{
  D("Executing %s", cmd);
  SimulationData *sim = (SimulationData*)cbdata;
  if (strcmp(cmd, "trigger snap") == 0)
  {
    D("trigger snap command");
    triggerSnap(sim);

#ifdef __DEBUG
    // generate some random data to see the change for now...
    for (size_t i = 0; i < sim->nX * sim->nY * sim->nZ; ++i)
      sim->snapshot[i] = (rand() * 10. / RAND_MAX) - 5.;
#endif

    // Redraw!
    VisItUpdatePlots();
  }
}

void mainloop(void)
{
  int visitstate, err = 0;

  SimulationData sim;

  sim.inited = 0;
  sim.cycle = 0;

  /* main loop */
  do
  {
    /* Get input from VisIt or timeout so the simulation can run. */
    /*visitstate = VisItDetectInput(1, fileno(stdin)); // blocking, eats way to much cpu...*/
    usleep(100000);
    visitstate = VisItDetectInput(0, fileno(stdin)); // nonblocking

    /* Do different things depending on the output from VisItDetectInput. */
    if (visitstate >= -5 && visitstate <= -1)
    {
      D("Can't recover from error!");
      err = 1;
    }
    else if (visitstate == 0)
    {
      /* There was no input from VisIt, do nothing. */
    }
    else if (visitstate == 1)
    {
      /* VisIt is trying to connect to sim. */
      if (VisItAttemptToCompleteConnection() == VISIT_OKAY)
      {
        D("VisIt connected");
        VisItSetCommandCallback(ControlCommandCallback, (void*)&sim);

        VisItSetGetMetaData(SimGetMetaData, (void*)&sim);
        VisItSetGetMesh(SimGetMesh, (void*)&sim);
        VisItSetGetVariable(SimGetVariable, (void*)&sim);
      }
      else
        D("VisIt did not connect");
    }
    else if (visitstate == 2)
    {
      /* VisIt wants to tell the engine something. */
      if (VisItProcessEngineCommand() == VISIT_ERROR)
      {
        /* Disconnect on an error or closed connection. */
        VisItDisconnect();
      }
    }
    else if (visitstate == 3)
    {
      /* VisItDetectInput detected console input - do something with it.
       * NOTE: you can't get here unless you pass a file descriptor to
       * VisItDetectInput instead of -1.
       */
    }
  }
  while (err == 0);

  FreeSim(&sim);
}

int main(int argc, char **argv)
{
  D("Startup VisIt Connector");

#ifdef __DEBUG
  VisItOpenTraceFile("visitconnector.trace");
#endif

  /* Initialize environment variables. */
  VisItSetupEnvironment();

  /* Write out .sim2 file that VisIt uses to connect. */
  VisItInitializeSocketAndDumpSimFile("parflow-visit-connector",
                                      "Attach to the running parflow simulation to do online observation",
                                      ".",  // Path to where sim was started...
                                      NULL, NULL, NULL);


  /***********************
   * init FlowVR Module
   */
  flowvr = fca_new_empty_module();
  portPressureIn = fca_new_port("pressureIn", fca_IN, 0, NULL);
  fca_append_port(flowvr, portPressureIn);

  portTriggerSnap = fca_new_port("triggerSnap", fca_OUT, 0, NULL);
  fca_append_port(flowvr, portTriggerSnap);

  stampTime = fca_register_stamp(portPressureIn, "stampTime", fca_FLOAT); // TODO good idea to use float? or should we put the double in the messages payload??
  if (!fca_init_module(flowvr))
  {
    D("ERROR : fca_init_module failed!");
  }

  /* Call the main loop. */
  mainloop();

  fca_free(flowvr);
  return 0;
}

/* DATA ACCESS FUNCTIONS */

visit_handle
SimGetMetaData(void *cbdata)
{
  D("SimGetMetaData");
  SimulationData *sim = (SimulationData*)cbdata;
  wait_for_init(sim);
  visit_handle md = VISIT_INVALID_HANDLE;

  /* Create metadata. */
  if (VisIt_SimulationMetaData_alloc(&md) == VISIT_OKAY)
  {
    int i;
    visit_handle m = VISIT_INVALID_HANDLE;
    visit_handle vmd = VISIT_INVALID_HANDLE;
    visit_handle cmd = VISIT_INVALID_HANDLE;

    /* Set the simulation state. */
    /*VisIt_SimulationMetaData_setMode(md, (sim->runMode == SIM_STOPPED) ?*/
    /*VISIT_SIMMODE_STOPPED : VISIT_SIMMODE_RUNNING);*/
    VisIt_SimulationMetaData_setCycleTime(md, sim->cycle, sim->time);

    char meshname[100];
    sprintf(meshname, "mesh%dx%dx%d", sim->nX, sim->nY, sim->nZ);
    if (VisIt_MeshMetaData_alloc(&m) == VISIT_OKAY)
    {
      /* Set the mesh's properties.*/
      VisIt_MeshMetaData_setName(m, meshname);
      VisIt_MeshMetaData_setMeshType(m, VISIT_MESHTYPE_RECTILINEAR);
      VisIt_MeshMetaData_setTopologicalDimension(m, 3);
      VisIt_MeshMetaData_setSpatialDimension(m, 3);
      VisIt_MeshMetaData_setXLabel(m, "x");
      VisIt_MeshMetaData_setYLabel(m, "y");
      VisIt_MeshMetaData_setZLabel(m, "z");

      VisIt_SimulationMetaData_addMesh(md, m);
    }

    /* Add a nodal scalar variable without mesh */
    if (VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
    {
      VisIt_VariableMetaData_setName(vmd, "pressure");
      VisIt_VariableMetaData_setMeshName(vmd, meshname);

      VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
      VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_NODE);

      VisIt_SimulationMetaData_addVariable(md, vmd);
    }


    /* Add some custom commands. */
    for (i = 0; i < sizeof(cmd_names) / sizeof(const char *); ++i)
    {
      visit_handle cmd = VISIT_INVALID_HANDLE;
      if (VisIt_CommandMetaData_alloc(&cmd) == VISIT_OKAY)
      {
        VisIt_CommandMetaData_setName(cmd, cmd_names[i]);
        VisIt_SimulationMetaData_addGenericCommand(md, cmd);
      }
    }
  }

  return md;
}


visit_handle
SimGetMesh(int domain, const char *name, void *cbdata)
{
  D("SimGetMesh");
  wait_for_init((SimulationData*)cbdata);
  visit_handle h = VISIT_INVALID_HANDLE;

  SimulationData const * const sim = (SimulationData*)cbdata;
  // nX, nY, nZ mesh...
  if (VisIt_RectilinearMesh_alloc(&h) != VISIT_ERROR)
  {
    visit_handle hxc, hyc, hzc;
    VisIt_VariableData_alloc(&hxc);
    VisIt_VariableData_alloc(&hyc);
    VisIt_VariableData_alloc(&hzc);
    VisIt_VariableData_setDataF(hxc, VISIT_OWNER_SIM, 1, sim->nX, sim->mesh_x);
    VisIt_VariableData_setDataF(hyc, VISIT_OWNER_SIM, 1, sim->nY, sim->mesh_y);
    VisIt_VariableData_setDataF(hzc, VISIT_OWNER_SIM, 1, sim->nZ, sim->mesh_z);
    VisIt_RectilinearMesh_setCoordsXYZ(h, hxc, hyc, hzc);
  }

  return h;
}

visit_handle
SimGetVariable(int domain, const char *name, void *cbdata)
{
  SimulationData const * const sim = (SimulationData*)cbdata;

  D("SimGetVariable");
  wait_for_init(sim);
  triggerSnap(sim);
  visit_handle h = VISIT_INVALID_HANDLE;
  int nComponents = 1, nTuples = 0;

  if (VisIt_VariableData_alloc(&h) == VISIT_OKAY)
  {
    if (strcmp(name, "pressure") == 0)
    {
      nTuples = sim->nX * sim->nY * sim->nZ;
      VisIt_VariableData_setDataD(h, VISIT_OWNER_SIM, nComponents,
                                  nTuples, sim->snapshot);
    }
  }
  return h;
}
