#include "amps.h"

amps_Handle amps_NewHandle(comm, id, invoice)
amps_Comm comm;
int id;
amps_Invoice invoice;
{
   amps_Handle handle;

   handle = (amps_Handle)malloc(sizeof(amps_HandleObject));

   handle -> comm     = comm;
   handle -> id       = id;
   handle -> invoice  = invoice;

   return handle;
}
