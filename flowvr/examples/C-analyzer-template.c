#include <fca.h>
#include <messages.h>
#include <signal.h>

// compile with:
// gcc -L$PARFLOW_DIR/lib -I../pfanalyzer -I/usr/local/include/fca C-analyzer-template.c -lfca -lpfanalyzer

fca_module flowvr;
fca_port port_in, port_out;

MergeMessageParser(onGridMessage)
{
  GridMessage gm = ReadGridMessage(buffer);

  // Allocate the steering operand.
  // We perform the Steering on the same part of the domain that we received a grid
  // message from.
  double operand[gm.m->nx * gm.m->ny * gm.m->nz];

  // The array index corresponding to the given coordinates
  size_t index = 0;

  for (int k = 0; k < gm.m->nz; ++k)
  {
    for (int j = 0; j < gm.m->ny; ++j)
    {
      for (int i = 0; i < gm.m->nx; ++i)
      {
        // The coordinates x, y and z are grid coordinates in the problem domain
        int x = i + gm.m->ix;
        int y = j + gm.m->iy;
        int z = k + gm.m->iz;

        // Doing sample analysis and calculate the operand
        operand[index] = 42. + x + y + z;


        ++index;
      }
    }
  }

  // Steer the simulation with the operand.
  // Depending on the FlowVR graph it is sometimes not guaranteed that the Steer
  // is performed for the next Simulation step already.
  SendSteerMessage(flowvr, port_out,
                   ACTION_SET, VARIABLE_PRESSURE, gm.m->ix, gm.m->iy, gm.m->iz,
                   operand, gm.m->nx, gm.m->ny, gm.m->nz);

  // now we need to return how much we read out from the merged message!
  return sizeof(GridMessageMetadata) + sizeof(double) * gm.m->nx * gm.m->ny * gm.m->nz;
}

int main(int argc, char *argv [])
{
  flowvr = fca_new_empty_module();

  // Add ports to the FlowVR module
  port_in = fca_new_port("in", fca_IN, 0, NULL);
  fca_append_port(flowvr, port_in);
  port_out = fca_new_port("out", fca_OUT, 0, NULL);
  fca_append_port(flowvr, port_out);

  // Register FlowVR module
  fca_init_module(flowvr);

  // Start message loop
  while (fca_wait(flowvr))
  {
    ParseMergedMessage(port_in, onGridMessage, NULL);
  }
  fca_free(flowvr);
}
