/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

int amps_Test(handle)
amps_Handle handle;
{
   if(handle)
      if(pvm_nrecv(handle -> id, amps_MsgTag ))
      {
	 /* we have recvd it */
	 amps_ClearInvoice(handle -> invoice);
	 amps_unpack(handle -> comm, handle -> invoice);
	 amps_FreeHandle(handle);
	 handle = NULL;
	 return 1;
      }
      else
	 /* did not recv it */
	 return 0;
   else
      return 1;
}
