#include <fca.h>
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

int main(int argc, char *argv [])
{
  /***********************
   * init FlowVR Module
   */
  fca_module flowvr = fca_new_empty_module();
  fca_port port_pressure_in = fca_new_port("pressureIn", fca_IN, 0, NULL);

  fca_append_port(flowvr, port_pressure_in);

  fca_port port_steer_out = fca_new_port("steerOut", fca_OUT, 0, NULL);
  fca_append_port(flowvr, port_steer_out);

  if (!fca_init_module(flowvr))
  {
    printf("ERROR : init_module failed!\n");
  }

  D("now waiting\n");
  while (fca_wait(flowvr))
  {
    fca_message msg = fca_get(port_pressure_in);
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
    for (size_t i = 0; i < s.nx * s.ny * s.nz; ++i)
      operand[i] = (rand() * 1. / RAND_MAX + .5);
    /*operand[i] = 1.;*/


    D("Send steer!");
    SendActionMessage(flowvr, port_steer_out, ACTION_MULTIPLY,
                      VARIABLE_PRESSURE, parameter, size);

    free(parameter);
  }

  fca_free(flowvr);
}
