/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "amps.h"


amps_Package amps_NewPackage(amps_Comm comm,
			     int num_send,
			     int *dest,
			     amps_Invoice *send_invoices,
			     int num_recv,
			     int *src,
			     amps_Invoice *recv_invoices)
{
   amps_Package package;


   package = (amps_Package)malloc(sizeof(amps_PackageStruct));
      
   package -> num_send = num_send;
   package -> dest = dest;
   package -> send_invoices = send_invoices;

   package -> num_recv = num_recv;
   package -> src = src;
   package -> recv_invoices = recv_invoices;

   if(num_recv)
      package -> recv_handles = (struct amps_HandleObject **)malloc(sizeof(amps_Handle)*num_recv);

   return package;
}

void amps_FreePackage(amps_Package package)
{
  if(package -> num_recv)
    free(package -> recv_handles);
  free(package);
}
