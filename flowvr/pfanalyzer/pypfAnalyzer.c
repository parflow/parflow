#include "messages.h"
#include <fca/fca.h>
#include <assert.h>
#include "parflow_config.h"

MergeMessageParser(preparser);
void callOnInit();

// Python Wrapper with NumpyDataTypes:

#ifdef __DEBUG
#define D(x ...) printf("pypypypy "); printf(x); printf(" %s:%d\n", __FILE__, __LINE__)
#else
#define D(...)
#endif

static fca_module flowvr;
static fca_port in;
static fca_port out;
static fca_port log;

void SendSteer(const Action action, const Variable variable,
               int ix, int iy, int iz,
               double *IN_ARRAY3, int DIM1, int DIM2, int DIM3)
{
  SendSteerMessage(flowvr, out, action, variable, ix, iy, iz, IN_ARRAY3, DIM3, DIM2, DIM1);
}

void SendLog(StampLog slog[], size_t n)
{
  SendLogMessage(flowvr, log, slog, n);
}


// TODO: add documentation!
void _run(char *logstamps[], size_t logstampsc)
{
  /***********************
   * init FlowVR Module
   */
  flowvr = fca_new_empty_module();
  in = fca_new_port("in", fca_IN, 0, NULL);

  fca_append_port(flowvr, in);

  out = fca_new_port("out", fca_OUT, 0, NULL);
  fca_append_port(flowvr, out);

  log = fca_new_port("log", fca_OUT, 0, NULL);
  while (logstampsc--)
  {
    D("Register stamp %s for logging", logstamps[logstampsc]);
    fca_register_stamp(log, logstamps[logstampsc], fca_FLOAT);
  }
  fca_append_port(flowvr, log);

  if (!fca_init_module(flowvr))
  {
    printf("ERROR : init_module failed!\n");
  }

  D("calling onInit now!");
  callOnInit();

  D("now waiting\n");
  while (fca_wait(flowvr))
  {
    ParseMergedMessage(in, &preparser, NULL);
  }

  fca_free(flowvr);
}

void SendEmpty()
{
  SendEmptyMessage(flowvr, out);
}


