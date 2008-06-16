/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

amps_Handle amps_IRecv(comm, source, invoice)
amps_Comm comm;
int source;
amps_Invoice invoice;
{
   int buffer;

   if((buffer = pvm_nrecv(amps_gettid(comm, source), amps_MsgTag)))
   {
      /* we have recvd it */
      amps_ClearInvoice( invoice );
      amps_unpack(comm, invoice);
      return NULL;
   }
   else
      /* did not recv it */
      return amps_NewHandle(comm, source, invoice, NULL);
}
