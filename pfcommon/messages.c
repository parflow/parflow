#include <fca/fca.h>
#include <string.h>

#include "messages.h"

void SendActionMessage(fca_module mod, fca_port port, Action action, Variable variable,
                       void *parameter, size_t parameterSize)
{
  fca_message msg = fca_new_message(mod, sizeof(ActionMessageMetadata) + parameterSize);
  ActionMessageMetadata *amm = (ActionMessageMetadata*)fca_get_write_access(msg, 0);

  amm->action = action;
  amm->variable = variable;
  memcpy((void*)(++amm), parameter, parameterSize);

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

