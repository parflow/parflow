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
   char *buffer;

   if(handle)
      if((buffer = amps_recv( handle -> id )))
      {
	 /* we have recvd it */
	 amps_ClearInvoice( (handle -> invoice) );
	 amps_unpack(handle -> comm, handle -> invoice, buffer);
	 AMPS_PACK_FREE_LETTER((handle -> comm), (handle -> invoice), buffer);

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
