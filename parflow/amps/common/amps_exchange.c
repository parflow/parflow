/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/


#include "amps.h"

#ifndef AMPS_EXCHANGE_SPECIALIZED

void _amps_wait_exchange(amps_Handle handle)
{
  int notdone;
  int i;


  notdone = handle -> package -> recv_remaining;
  while(notdone>1)
    for(i = 0; i <  handle -> package -> num_recv; i++)
      if(handle -> package -> recv_handles[i])
	if(amps_Test((amps_Handle)handle -> package -> recv_handles[i]))
	  {
	    handle -> package -> recv_handles[i] = NULL;
	    notdone--;
	  }
  
  for(i = 0; i <  handle -> package -> num_recv; i++)
    if(handle -> package -> recv_handles[i])
      {
	amps_Wait((amps_Handle)handle -> package -> recv_handles[i]);
	handle -> package -> recv_handles[i] = NULL;
      }
}

amps_Handle amps_IExchangePackage(amps_Package package)
{

   int i;

   /*--------------------------------------------------------------------
    * post receives for data to get
    *--------------------------------------------------------------------*/
   package -> recv_remaining = 0;

   for(i = 0; i < package -> num_recv; i++)
     if( (package -> recv_handles[i] =
       (struct amps_HandleObject *)amps_IRecv(amps_CommWorld, 
					       package -> src[i],
					       package -> recv_invoices[i])) )
	 package -> recv_remaining++;

   /*--------------------------------------------------------------------
    * send out the data we have
    *--------------------------------------------------------------------*/
   for(i = 0; i < package -> num_send; i++)
   {
      amps_Send(amps_CommWorld, 
		package -> dest[i],
		package -> send_invoices[i]);
   }

   return( amps_NewHandle(NULL, 0, NULL, package));
}

#endif
