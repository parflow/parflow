/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

/*
   This should probably be written using Log type alg. since PVM is stupid.
*/

int amps_BCast(comm, source, invoice) 
amps_Comm comm;
int source;
amps_Invoice invoice;
{

   int buffer;

   if(source == amps_Rank(comm))
   {
      amps_pack(comm, invoice);
      pvm_bcast(comm, amps_MsgTag);
   }
   else
   {
      /* in case there is some memory still allocated */
      amps_ClearInvoice(invoice);
      pvm_recv(-1, amps_MsgTag);
      amps_unpack(comm, invoice);
   }
   
   return 0;
}
