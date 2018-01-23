#include <fca/fca.h>
#include <string.h>

#include "messages.h"

#include <assert.h>

const char *VARIABLE_TO_NAME[VARIABLE_LAST] = {
  "pressure",
  "saturation",
  "porosity",
  "manning",
  "permeability_x",
  "permeability_y",
  "permeability_z"
};

void SendActionMessage(fca_module mod, fca_port port, Action action, Variable variable,
                       void *parameter, size_t parameter_size)
{
  fca_message msg = fca_new_message(mod, sizeof(ActionMessageMetadata) + parameter_size);
  ActionMessageMetadata *amm = (ActionMessageMetadata*)fca_get_write_access(msg, 0);

  amm->action = action;
  amm->variable = variable;
  memcpy((void*)(++amm), parameter, parameter_size);

  fca_put(port, msg);
  fca_free(msg);
}

void ParseMergedMessage(fca_port port, size_t (*cb)(const void *buffer, size_t size, void *cbdata), void *cbdata)
{
  fca_message msg = fca_get(port);
  size_t s = fca_get_segment_size(msg, 0);

  if (s > 0)
  {
    const void* buffer = fca_get_read_access(msg, 0);
    void* end = buffer + s;
    while (buffer < end)
    {
      buffer += cb(buffer, s, cbdata);
//      printf("--buffer: %#010x/%#010x\n", buffer, end);
    }
    if (buffer > end)
    {
      printf("Probably a message parsing error occured!");
    }
  }
  fca_free(msg);
}

void SendLogMessage(fca_module mod, fca_port port, StampLog log[], size_t n)
{
  fca_message msg = fca_new_message(mod, 0);

  while (n--)
  {
    fca_stamp stamp = fca_get_stamp(port, log[n].stamp_name);
    fca_write_stamp(msg, stamp, (void*)&(log[n].value));
  }
  fca_put(port, msg);
  fca_free(msg);
}

Variable NameToVariable(const char *name)
{
  for (Variable res = VARIABLE_PRESSURE; res < VARIABLE_LAST; ++res)
  {
    if (strcmp(VARIABLE_TO_NAME[res], name) == 0)
      return res;
  }
  assert(0 && "Could not convert to a Variable!");
  return VARIABLE_LAST;
}

GenerateMessageReaderC(Grid);
GenerateMessageReaderC(Steer);
GenerateMessageReaderC(Action);
