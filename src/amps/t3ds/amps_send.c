/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

int amps_Send(comm, dest, invoice)
amps_Comm comm;
int dest;
amps_Invoice invoice;
{
   amps_pack(comm, invoice);

   pvm_send(dest, amps_MsgTag);
   
   amps_ClearInvoice(invoice);

   return 0;
}
