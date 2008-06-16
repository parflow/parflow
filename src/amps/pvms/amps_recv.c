/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

int amps_Recv(comm, source, invoice)
amps_Comm comm;
int source;
amps_Invoice invoice;
{

   amps_ClearInvoice(invoice);

   pvm_recv(amps_gettid(comm, source), amps_MsgTag);
   
   amps_unpack(comm, invoice);

   return 0;
}
