#include "amps.h"

int amps_Wait(handle)
amps_Handle handle;
{
   if(handle)
   {
      amps_Recv(handle -> comm, handle -> id, handle -> invoice);

      amps_FreeHandle(handle);
      handle = NULL;
   }
   
   return 0;
}
