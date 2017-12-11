#include "messages.h"
#include <fca/fca.h>
#include <assert.h>


MergeMessageParser(preparser);

// Python Wrapper with NumpyDataTypes:

#ifdef __DEBUG
#define D(x ...) printf("pypypypy "); printf(x); printf(" %s:%d\n", __FILE__, __LINE__)
#else
#define D(...)
#endif

static fca_module flowvr;
static fca_port in;
static fca_port out;

void SendSteerMessage(const Action action, const Variable variable,
                      int ix, int iy, int iz,
                      double *IN_ARRAY3, int DIM1, int DIM2, int DIM3)
{
  fca_message msg = fca_new_message(flowvr, sizeof(ActionMessageMetadata) + sizeof(SteerMessageMetadata) + sizeof(double) * DIM1 * DIM2 * DIM3);
  ActionMessageMetadata *amm = (ActionMessageMetadata*)fca_get_write_access(msg, 0);

  amm->action = action;
  amm->variable = variable;
  SteerMessageMetadata *m = (SteerMessageMetadata*)(amm + 1);
  m->ix = ix;
  m->iy = iy;
  m->iz = iz;
  m->nx = DIM1;
  m->ny = DIM2;
  m->nz = DIM3;

  // low: would be cooler if we could work directly in the message from python...
  memcpy((void*)(m + 1), IN_ARRAY3, sizeof(double) * DIM1 * DIM2 * DIM3);

  fca_put(out, msg);
  fca_free(msg);
}

// TODO: add documentation!
void _run()
{
  /***********************
   * init FlowVR Module
   */
  flowvr = fca_new_empty_module();
  in = fca_new_port("in", fca_IN, 0, NULL);

  fca_append_port(flowvr, in);

  out = fca_new_port("out", fca_OUT, 0, NULL);
  fca_append_port(flowvr, out);

  if (!fca_init_module(flowvr))
  {
    printf("ERROR : init_module failed!\n");
  }

  D("now waiting\n");
  while (fca_wait(flowvr))
  {
    ParseMergedMessage(in, &preparser, NULL);
  }

  fca_free(flowvr);
}
