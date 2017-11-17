/*****************************************************************************
*
* Copyright (c) 2000 - 2010, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory
* LLNL-CODE-400142
* All rights reserved.
*
* This file is  part of VisIt. For  details, see https://visit.llnl.gov/.  The
* full copyright notice is contained in the file COPYRIGHT located at the root
* of the VisIt distribution or at http://www.llnl.gov/visit/copyright.html.
*
* Redistribution  and  use  in  source  and  binary  forms,  with  or  without
* modification, are permitted provided that the following conditions are met:
*
*  - Redistributions of  source code must  retain the above  copyright notice,
*    this list of conditions and the disclaimer below.
*  - Redistributions in binary form must reproduce the above copyright notice,
*    this  list of  conditions  and  the  disclaimer (as noted below)  in  the
*    documentation and/or other materials provided with the distribution.
*  - Neither the name of  the LLNS/LLNL nor the names of  its contributors may
*    be used to endorse or promote products derived from this software without
*    specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT  HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR  IMPLIED WARRANTIES, INCLUDING,  BUT NOT  LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND  FITNESS FOR A PARTICULAR  PURPOSE
* ARE  DISCLAIMED. IN  NO EVENT  SHALL LAWRENCE  LIVERMORE NATIONAL  SECURITY,
* LLC, THE  U.S.  DEPARTMENT OF  ENERGY  OR  CONTRIBUTORS BE  LIABLE  FOR  ANY
* DIRECT,  INDIRECT,   INCIDENTAL,   SPECIAL,   EXEMPLARY,  OR   CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT  LIMITED TO, PROCUREMENT OF  SUBSTITUTE GOODS OR
* SERVICES; LOSS OF  USE, DATA, OR PROFITS; OR  BUSINESS INTERRUPTION) HOWEVER
* CAUSED  AND  ON  ANY  THEORY  OF  LIABILITY,  WHETHER  IN  CONTRACT,  STRICT
* LIABILITY, OR TORT  (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY  WAY
* OUT OF THE  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
* DAMAGE.
*
*****************************************************************************/

#include <fca/fca.h>
#include "parflow_config.h"
#include "../../pfsimulator/parflow_lib/GridMessage.h"
#include <VisItControlInterface_V2.h>
#include <VisItDataInterface_V2.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


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

#define SIM_STOPPED       0
#define SIM_RUNNING       1

typedef struct {
  int cycle;
  double time;
  int runMode;
  int done;
} simulation_data;

const char *cmd_names[] = { "trigger snap" };

static double lastTime;
static fca_module flowvr;
static fca_stamp stampTime;
static fca_port portPressureIn;
static fca_port portTriggerSnap;
static double* snapshot=NULL;
static size_t nX = 0;
static size_t nY = 0;
static size_t nZ = 0;
static float *mesh_x;
static float *mesh_y;
static float *mesh_z;
static int inited = 0;

void triggerSnap(void)  // TODO: execute reload here in visit automatically
{
  fca_message msgOut = fca_new_message(flowvr, 1);
  // TODO: really the only way to transmit messages of size !=0 that are nonblocking?
  fca_put(portTriggerSnap, msgOut);
  fca_free(msgOut);
  D("waiting...");
  fca_wait(flowvr);
  D("got an answer");

  // TODO: might happen that we are getting different timesteps?!! I think so as sometimes one mpi process is behind and one is before the wait for the event module...
  fca_message msgIn = fca_get(portPressureIn);
  double lastTime = (double)*((float*)fca_read_stamp(msgIn, stampTime));
  const void* buffer = fca_get_read_access(msgIn, 0);
  void* end = buffer + fca_get_segment_size(msgIn, 0);

  GridMessageMetadata* m = (GridMessageMetadata*)buffer;
  nX = m->nX;
  nY = m->nY;
  nZ = m->nZ;

  if (snapshot != NULL) {
    free(snapshot);
  }

  snapshot = (double*)malloc(sizeof(double)*m->nX*m->nY*m->nZ);

  // populate snapshot:
  while (buffer < end)
  {
    buffer += sizeof(GridMessageMetadata);
    const double * data = (double*)buffer;
    for (int z = 0; z < m->nz; ++z) {  // TODO: unroll?
      for (int y = 0; y < m->ny; ++y) {
        for (int x = 0; x < m->nx; ++x) {
          int snapindex = x+m->ix + (y+m->iy) * nX + (z+m->iz) * nX * nY;
          snapshot[snapindex] = *data;
          ++data;
        }
      }
    }

    // get next...
    buffer += sizeof(double) * m->nx * m->ny * m->nz;
    m = (GridMessageMetadata*)buffer;
    D("copied buffers!");
  }
  fca_free(msgIn);
}

void wait_for_init(void)
{
  if (inited)
  {
    return;
  }
  else
  {
    triggerSnap();

    // populate mesh:
    mesh_x = (float*) malloc(nX*sizeof(float));
    mesh_y = (float*) malloc(nY*sizeof(float));
    mesh_z = (float*) malloc(nZ*sizeof(float));
    for (size_t i=0; i<nX || i<nY || i<nZ; ++i)
    {
      if (i<nX) mesh_x[i] = (float)i;
      if (i<nY) mesh_y[i] = (float)i;
      if (i<nZ) mesh_z[i] = (float)i;
    }

    inited = 1;
  }
}

/******************************************************************************
 *
 * Purpose: Callback function for control commands.
 *
 * Programmer: Brad Whitlock
 * Date:       Fri Feb  6 14:29:36 PST 2009
 *
 * Input Arguments:
 *   cmd    : The command string that we want the sim to execute.
 *   args   : String argument for the command.
 *   cbdata : User-provided callback data.
 *
 * Modifications:
 *
 *****************************************************************************/
void ControlCommandCallback(const char *cmd, const char *args, void *cbdata)
{
  simulation_data *sim = (simulation_data*)cbdata;
  if (strcmp(cmd, "trigger snap") == 0) {
    triggerSnap();
  }
}

/******************************************************************************
 *
 * Purpose: This is the main event loop function.
 *
 * Programmer: Brad Whitlock
 * Date:       Fri Feb  6 14:29:36 PST 2009
 *
 * Modifications:
 *
 *****************************************************************************/

void mainloop(void)
{
  int visitstate, err = 0;

  simulation_data sim;

  /* main loop */
  do
  {
    /* Get input from VisIt or timeout so the simulation can run. */
    /*visitstate = VisItDetectInput(1, fileno(stdin)); // blocking*/
    usleep(100000);
    visitstate = VisItDetectInput(0, fileno(stdin)); // nonblocking

    /* Do different things depending on the output from VisItDetectInput. */
    if (visitstate >= -5 && visitstate <= -1)
    {
      fprintf(stderr, "Can't recover from error!\n");
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
        sim.runMode = SIM_STOPPED;
        fprintf(stderr, "VisIt connected\n");
        VisItSetCommandCallback(ControlCommandCallback, (void*)&sim);

        VisItSetGetMetaData(SimGetMetaData, (void*)&sim);
        VisItSetGetMesh(SimGetMesh, (void*)&sim);
        VisItSetGetVariable(SimGetVariable, (void*)&sim);
      }
      else
        fprintf(stderr, "VisIt did not connect\n");
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

  if (snapshot != NULL) {
    free(snapshot);
  }
}

/******************************************************************************
 *
 * Purpose: This is the main function for the program.
 *
 * Programmer: Brad Whitlock
 * Date:       Fri Feb  6 14:29:36 PST 2009
 *
 * Input Arguments:
 *   argc : The number of command line arguments.
 *   argv : The command line arguments.
 *
 * Modifications:
 *
 *****************************************************************************/

int main(int argc, char **argv)
{
  D("Startup VisIt Connector");

#ifdef __DEBUG
  VisItOpenTraceFile("visitconnector.trace");
#endif

  /* Initialize environment variables. */
  VisItSetupEnvironment();

  /* Write out .sim2 file that VisIt uses to connect. */
  VisItInitializeSocketAndDumpSimFile("scalar",
                                      "Demonstrates scalar data access function",
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
// TODO: does it work without stampFileName?
  if (!fca_init_module(flowvr))
  {
    printf("ERROR : fca_init_module failed!\n");
  }

  /* Call the main loop. */
  mainloop();

  fca_free(flowvr);
  if (inited) {
    free(mesh_x);
    free(mesh_y);
    free(mesh_z);
    free(snapshot);
  }

  return 0;
}

/* DATA ACCESS FUNCTIONS */

/******************************************************************************
 *
 * Purpose: This callback function returns simulation metadata.
 *
 * Programmer: Brad Whitlock
 * Date:       Fri Feb  6 14:29:36 PST 2009
 *
 * Modifications:
 *
 *****************************************************************************/

visit_handle
SimGetMetaData(void *cbdata)
{
  D("SimGetMetaData");
  wait_for_init();
  visit_handle md = VISIT_INVALID_HANDLE;
  simulation_data *sim = (simulation_data*)cbdata;

  /* Create metadata. */
  if (VisIt_SimulationMetaData_alloc(&md) == VISIT_OKAY)
  {
    int i;
    visit_handle m1 = VISIT_INVALID_HANDLE, m2 = VISIT_INVALID_HANDLE,
                 m3 = VISIT_INVALID_HANDLE;
    visit_handle vmd = VISIT_INVALID_HANDLE;
    visit_handle cmd = VISIT_INVALID_HANDLE;

    /* Set the simulation state. */
    VisIt_SimulationMetaData_setMode(md, (sim->runMode == SIM_STOPPED) ?
                                     VISIT_SIMMODE_STOPPED : VISIT_SIMMODE_RUNNING);
    VisIt_SimulationMetaData_setCycleTime(md, sim->cycle, sim->time);

    /* Set the first mesh's properties.*/
    if (VisIt_MeshMetaData_alloc(&m1) == VISIT_OKAY)
    {
      /* Set the mesh's properties.*/
      VisIt_MeshMetaData_setName(m1, "mesh2d");
      VisIt_MeshMetaData_setMeshType(m1, VISIT_MESHTYPE_RECTILINEAR);
      VisIt_MeshMetaData_setTopologicalDimension(m1, 2);
      VisIt_MeshMetaData_setSpatialDimension(m1, 2);
      VisIt_MeshMetaData_setXUnits(m1, "cm");
      VisIt_MeshMetaData_setYUnits(m1, "cm");
      VisIt_MeshMetaData_setXLabel(m1, "Width");
      VisIt_MeshMetaData_setYLabel(m1, "Height");

      VisIt_SimulationMetaData_addMesh(md, m1);
    }

    /* Set the second mesh's properties.*/
    if (VisIt_MeshMetaData_alloc(&m2) == VISIT_OKAY)
    {
      /* Set the mesh's properties.*/
      VisIt_MeshMetaData_setName(m2, "mesh3d");
      VisIt_MeshMetaData_setMeshType(m2, VISIT_MESHTYPE_CURVILINEAR);
      VisIt_MeshMetaData_setTopologicalDimension(m2, 3);
      VisIt_MeshMetaData_setSpatialDimension(m2, 3);
      VisIt_MeshMetaData_setXUnits(m2, "cm");
      VisIt_MeshMetaData_setYUnits(m2, "cm");
      VisIt_MeshMetaData_setZUnits(m2, "cm");
      VisIt_MeshMetaData_setXLabel(m2, "Width");
      VisIt_MeshMetaData_setYLabel(m2, "Height");
      VisIt_MeshMetaData_setZLabel(m2, "Depth");

      VisIt_SimulationMetaData_addMesh(md, m2);
    }


    char meshname3[100];
    sprintf(meshname3, "mesh%dx%dx%d", nX, nY, nZ);
    if (VisIt_MeshMetaData_alloc(&m3) == VISIT_OKAY)
    {
      /* Set the mesh's properties.*/
      VisIt_MeshMetaData_setName(m3, meshname3);
      VisIt_MeshMetaData_setMeshType(m3, VISIT_MESHTYPE_RECTILINEAR);
      VisIt_MeshMetaData_setTopologicalDimension(m3, 3);
      VisIt_MeshMetaData_setSpatialDimension(m3, 3);
      VisIt_MeshMetaData_setXLabel(m3, "x");
      VisIt_MeshMetaData_setYLabel(m3, "y");
      VisIt_MeshMetaData_setZLabel(m3, "z");

      VisIt_SimulationMetaData_addMesh(md, m3);
    }

    /* Add a zonal scalar variable on mesh2d. */
    if (VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
    {
      VisIt_VariableMetaData_setName(vmd, "zonal");
      VisIt_VariableMetaData_setMeshName(vmd, "mesh2d");
      VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
      VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);

      VisIt_SimulationMetaData_addVariable(md, vmd);
    }

    /* Add a nodal scalar variable on mesh3d. */
    if (VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
    {
      VisIt_VariableMetaData_setName(vmd, "nodal");
      VisIt_VariableMetaData_setMeshName(vmd, "mesh3d");
      VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
      VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_NODE);

      VisIt_SimulationMetaData_addVariable(md, vmd);
    }

    /* Add a nodal scalar variable without mesh */
    if (VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
    {
      VisIt_VariableMetaData_setName(vmd, "pressure");
      VisIt_VariableMetaData_setMeshName(vmd, meshname3);

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

/* Rectilinear mesh */
float rmesh_x[] = { 0., 1., 2.5, 5. };
float rmesh_y[] = { 0., 2., 2.25, 2.55, 5. };
int rmesh_dims[] = { 4, 5, 1 };
int rmesh_ndims = 2;
float zonal[] = { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. };

/*float rmesh_x[] = { 0., 1., 2., 3., 4., 5., 6., 7., 8., 9. };*/
/*float rmesh_y[] = { 0., 1., 2., 3., 4., 5., 6., 7., 8., 9. };*/


/*int rmesh_dims[] = { 10, 10, 1 };*/
/*int rmesh_ndims = 2;*/

/*float zonal[] = { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. };*/
float zonal_vector[][2] = {
  { 1., 2. }, { 3., 4. }, { 5., 6. }, { 7., 8. }, { 9., 10. }, { 11., 12. },
  { 13., 14. }, { 15., 16. }, { 17., 18. }, { 19., 20. }, { 21., 22. }, { 23., 24. }
};
const char *zonal_labels = "zone1\0\0zone2\0\0zone3\0\0zone4\0\0zone5\0\0zone6\0\0zone7\0\0zone8\0\0zone9\0\0zone10\0zone11\0zone12";

/* Curvilinear mesh */
float cmesh_x[2][3][4] = {
  { { 0., 1., 2., 3. }, { 0., 1., 2., 3. }, { 0., 1., 2., 3. } },
  { { 0., 1., 2., 3. }, { 0., 1., 2., 3. }, { 0., 1., 2., 3. } }
};
float cmesh_y[2][3][4] = {
  { { 0.5, 0., 0., 0.5 }, { 1., 1., 1., 1. }, { 1.5, 2., 2., 1.5 } },
  { { 0.5, 0., 0., 0.5 }, { 1., 1., 1., 1. }, { 1.5, 2., 2., 1.5 } }
};
float cmesh_z[2][3][4] = {
  { { 0., 0., 0., 0. }, { 0., 0., 0., 0. }, { 0., 0., 0., 0. } },
  { { 1., 1., 1., 1. }, { 1., 1., 1., 1. }, { 1., 1., 1., 1. } }
};
int cmesh_dims[] = { 4, 3, 2 };
int cmesh_ndims = 3;
double nodal[2][3][4] = {
  { { 1., 2., 3., 4. }, { 5., 6., 7., 8. }, { 9., 10., 11., 12 } },
  { { 13., 14., 15., 16. }, { 17., 18., 19., 20. }, { 21., 22., 23., 24. } }
};
double nodal_vector[2][3][4][3] = {
  { { { 0., 1., 2. }, { 3., 4., 5. }, { 6., 7., 8. }, { 9., 10., 11. } },
    { { 12., 13., 14. }, { 15., 16., 17. }, { 18., 19., 20. }, { 21., 22., 23. } },
    { { 24., 25., 26. }, { 27., 28., 29. }, { 30., 31., 32. }, { 33., 34., 35. } } },

  { { { 36., 37., 38. }, { 39., 40., 41. }, { 42., 43., 44. }, { 45., 46., 47. } },
    { { 48., 49., 50. }, { 51., 52., 53. }, { 54., 55., 56. }, { 57., 58., 59. } },
    { { 60., 61., 62. }, { 63., 64., 65. }, { 66., 67., 68. }, { 69., 70., 71 } } }
};

/******************************************************************************
 *
 * Purpose: This callback function returns meshes.
 *
 * Programmer: Brad Whitlock
 * Date:       Fri Feb  6 14:29:36 PST 2009
 *
 * Modifications:
 *
 *****************************************************************************/

visit_handle
SimGetMesh(int domain, const char *name, void *cbdata)
{
  D("SimGetMesh");
  wait_for_init();
  visit_handle h = VISIT_INVALID_HANDLE;

  if (strcmp(name, "mesh2d") == 0)
  {
    if (VisIt_RectilinearMesh_alloc(&h) != VISIT_ERROR)
    {
      visit_handle hxc, hyc;
      VisIt_VariableData_alloc(&hxc);
      VisIt_VariableData_alloc(&hyc);
      VisIt_VariableData_setDataF(hxc, VISIT_OWNER_SIM, 1, rmesh_dims[0], rmesh_x);
      VisIt_VariableData_setDataF(hyc, VISIT_OWNER_SIM, 1, rmesh_dims[1], rmesh_y);
      VisIt_RectilinearMesh_setCoordsXY(h, hxc, hyc);
    }
  }
  else if (strcmp(name, "mesh3d") == 0)
  {
    if (VisIt_CurvilinearMesh_alloc(&h) != VISIT_ERROR)
    {
      int nn;
      visit_handle hxc, hyc, hzc;
      nn = cmesh_dims[0] * cmesh_dims[1] * cmesh_dims[2];
      VisIt_VariableData_alloc(&hxc);
      VisIt_VariableData_alloc(&hyc);
      VisIt_VariableData_alloc(&hzc);
      VisIt_VariableData_setDataF(hxc, VISIT_OWNER_SIM, 1, nn, (float*)cmesh_x);
      VisIt_VariableData_setDataF(hyc, VISIT_OWNER_SIM, 1, nn, (float*)cmesh_y);
      VisIt_VariableData_setDataF(hzc, VISIT_OWNER_SIM, 1, nn, (float*)cmesh_z);
      VisIt_CurvilinearMesh_setCoordsXYZ(h, cmesh_dims, hxc, hyc, hzc);
    }
  }
  else
  {
    // nX, nY, nZ mesh...
    if (VisIt_RectilinearMesh_alloc(&h) != VISIT_ERROR)
    {
      visit_handle hxc, hyc, hzc;
      VisIt_VariableData_alloc(&hxc);
      VisIt_VariableData_alloc(&hyc);
      VisIt_VariableData_alloc(&hzc);
      VisIt_VariableData_setDataF(hxc, VISIT_OWNER_SIM, 1, nX, mesh_x);
      VisIt_VariableData_setDataF(hyc, VISIT_OWNER_SIM, 1, nY, mesh_y);
      VisIt_VariableData_setDataF(hzc, VISIT_OWNER_SIM, 1, nZ, mesh_z);
      D("%dx%dx%d", nX, nY, nZ);
      VisIt_RectilinearMesh_setCoordsXYZ(h, hxc, hyc, hzc);
    }
  }

  return h;
}

/******************************************************************************
 *
 * Purpose: This callback function returns meshes.
 *
 * Programmer: Brad Whitlock
 * Date:       Fri Feb  6 14:29:36 PST 2009
 *
 * Modifications:
 *
 *****************************************************************************/

visit_handle
SimGetVariable(int domain, const char *name, void *cbdata)
{
  D("SimGetVariable");
  wait_for_init();
  visit_handle h = VISIT_INVALID_HANDLE;
  int nComponents = 1, nTuples = 0;

  if (VisIt_VariableData_alloc(&h) == VISIT_OKAY)
  {
    if (strcmp(name, "zonal") == 0)
    {
      nTuples = (rmesh_dims[0] - 1) * (rmesh_dims[1] - 1);
      VisIt_VariableData_setDataF(h, VISIT_OWNER_SIM, nComponents,
                                  nTuples, zonal);
    }
    else if (strcmp(name, "nodal") == 0)
    {
      nTuples = cmesh_dims[0] * cmesh_dims[1] *
                cmesh_dims[2];
      VisIt_VariableData_setDataD(h, VISIT_OWNER_SIM, nComponents,
                                  nTuples, (double*)nodal);
    }
    else if (strcmp(name, "pressure") == 0)
    {
      nTuples = nX * nY * nZ;
      VisIt_VariableData_setDataD(h, VISIT_OWNER_SIM, nComponents,
                                  nTuples, snapshot);
    }
  }
  return h;
}
