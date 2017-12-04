#include <fca/fca.h>
#include "parflow_config.h"
#include <messages.h>
#include <stdio.h>
#include <stdlib.h>  // for rand()
#include <assert.h>
#include <unistd.h>


#ifdef __DEBUG
#define D(x ...) printf("aaaaaaaa "); printf(x); printf(" %s:%d\n", __FILE__, __LINE__)
#else
#define D(...)
#endif

// TODO: there must be a very similar code in the pfsimulator code. use it?!
void setInBox(double *data, SteerMessageMetadata const * const s, double value, double lower_x, double lower_y, double lower_z,
              double upper_x, double upper_y, double upper_z)
{
  const int dx = 10;
  const int dy = 10;
  const int dz = 1;

  // TODO:  better information transmission..., too much hard coded
  for (int x = (int)lower_x / dx; x < (int)upper_x / dx; ++x)
  {
    for (int y = (int)lower_y / dy; y < (int)upper_y / dy; ++y)
    {
      for (int z = (int)lower_z / dz; z < (int)upper_z / dz; ++z)  // TODO: probably we can use a classic boxloop here ...
      {
        if (x >= s->ix && x < s->ix + s->nx &&
            y >= s->iy && y < s->iy + s->ny &&
            z >= s->iz && z < s->iz + s->nz)
        {
          size_t index = x - s->ix + (y - s->iy) * s->nx + (z - s->iz) * s->nx * s->ny;
          data[index] = value;
        }
      }
    }
  }
}

int main(int argc, char *argv [])
{
  /***********************
   * init FlowVR Module
   */
  fca_module flowvr = fca_new_empty_module();
  fca_port portPressureIn = fca_new_port("pressureIn", fca_IN, 0, NULL);

  fca_append_port(flowvr, portPressureIn);

  fca_port portSteerOut = fca_new_port("steerOut", fca_OUT, 0, NULL);
  fca_append_port(flowvr, portSteerOut);

  if (!fca_init_module(flowvr))
  {
    printf("ERROR : init_module failed!\n");
  }

  int currentFileID;
  int xID, yID, zID, timeID;
  D("now waiting\n");
  while (fca_wait(flowvr))
  {
    fca_message msg = fca_get(portPressureIn);
// do something like this instead:
//ParseMergedMessage(portPressureIn, setSnapshot, (void*)sim);
// see content TODO
// do steering.
    GridMessageMetadata *m;
    m = (GridMessageMetadata*)fca_get_read_access(msg, 0);

    SteerMessageMetadata s;
    s.ix = 0;
    s.iy = 0;
    s.iz = 0;
    s.nx = m->grid.nX;
    s.ny = m->grid.nY;
    s.nz = m->grid.nZ;
    fca_free(msg);

    size_t size = sizeof(SteerMessageMetadata) + sizeof(double) * s.nx * s.ny * s.nz;
    void *parameter = malloc(size);
    memcpy(parameter, (void*)&s, sizeof(SteerMessageMetadata));
    double *operand = (double*)(parameter + sizeof(SteerMessageMetadata));

    double H1_Lower_X = 0;
    double H1_Lower_Y = 0;
    double H1_Lower_Z = 20;
    double H1_Upper_X = 10;
    double H1_Upper_Y = 500;
    double H1_Upper_Z = 24;
    double H2_Lower_X = 0;
    double H2_Lower_Y = 0;
    double H2_Lower_Z = 15;
    double H2_Upper_X = 10;
    double H2_Upper_Y = 500;
    double H2_Upper_Z = 20;
    double H3_Lower_X = 0;
    double H3_Lower_Y = 0;
    double H3_Lower_Z = 10;
    double H3_Upper_X = 10;
    double H3_Upper_Y = 500;
    double H3_Upper_Z = 15;
    double H4_Lower_X = 0;
    double H4_Lower_Y = 0;
    double H4_Lower_Z = 5;
    double H4_Upper_X = 10;
    double H4_Upper_Y = 500;
    double H4_Upper_Z = 10;
    double H5_Lower_X = 0;
    double H5_Lower_Y = 0;
    double H5_Lower_Z = 0;
    double H5_Upper_X = 10;
    double H5_Upper_Y = 500;
    double H5_Upper_Z = 5;

    // set standard perm
    // TODO: later this must be all parametrized!
    double K = 1.;
    double K1 = K * 0.0000001667;
    double K2 = K * 0.0042;
    double K3 = K * 0.003;
    double K4 = K * 0.0042;
    double K5 = K * 0.0042;

    setInBox(operand, &s, K1, H1_Lower_X, H1_Lower_Y, H1_Lower_Z,
             H1_Upper_X, H1_Upper_Y, H1_Upper_Z);
    setInBox(operand, &s, K2, H2_Lower_X, H2_Lower_Y, H2_Lower_Z,
             H2_Upper_X, H2_Upper_Y, H2_Upper_Z);
    setInBox(operand, &s, K3, H3_Lower_X, H3_Lower_Y, H3_Lower_Z,
             H3_Upper_X, H3_Upper_Y, H3_Upper_Z);
    setInBox(operand, &s, K4, H4_Lower_X, H4_Lower_Y, H4_Lower_Z,
             H4_Upper_X, H4_Upper_Y, H4_Upper_Z);
    setInBox(operand, &s, K5, H5_Lower_X, H5_Lower_Y, H5_Lower_Z,
             H5_Upper_X, H5_Upper_Y, H5_Upper_Z);

    D("Send steer!");
    // Should not change anything for now.
    // TODO: does this work as we are sending multiple actions here...
    SendActionMessage(flowvr, portSteerOut, ACTION_SET,
                      VARIABLE_PERMEABILITY_X, parameter, size);
    SendActionMessage(flowvr, portSteerOut, ACTION_SET,
                      VARIABLE_PERMEABILITY_Y, parameter, size);
    SendActionMessage(flowvr, portSteerOut, ACTION_SET,
                      VARIABLE_PERMEABILITY_Z, parameter, size);

    free(parameter);
    // TODO: SendActionsMessage for multiple actions
    // sth like add action and then send action...

    // TODO: rename into action port (triggerSnap and steeout?) or at least show type in port name??
  }

  fca_free(flowvr);
}
