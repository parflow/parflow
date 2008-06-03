/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

int amps_Wait(handle)
amps_Handle handle;
{
   if(handle)
   {
      if(handle -> type)
	 amps_Recv(handle -> comm, handle -> id, handle -> invoice);
      else
	 _amps_wait_exchange(handle);

      amps_FreeHandle(handle);
      handle = NULL;
   }

   return 0;
}
