/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "amps.h"

#ifdef AMPS_MPI_NOT_USE_PERSISTENT

amps_Package amps_NewPackage(amps_Comm comm,
			     int num_send,
			     int *dest,
			     amps_Invoice *send_invoices,
			     int num_recv,
			     int *src,
			     amps_Invoice *recv_invoices)
{
   amps_Package package;


   package = (amps_Package)calloc(1, sizeof(amps_PackageStruct));

   if(num_recv+num_send)
   {
      package -> requests = (MPI_Request *)calloc((num_recv+num_send),
						  sizeof(MPI_Request));
   }
    
   package -> num_send = num_send;
   package -> dest = dest;
   package -> send_invoices = send_invoices;

   package -> num_recv = num_recv;
   package -> src = src;
   package -> recv_invoices = recv_invoices;

   return package;
}

void amps_FreePackage(amps_Package package)
{
  if(package -> num_recv + package -> num_send)
  {
     free(package -> requests);
  }

  free(package);
}

#else

amps_Package amps_NewPackage(amps_Comm comm,
			     int num_send,
			     int *dest,
			     amps_Invoice *send_invoices,
			     int num_recv,
			     int *src,
			     amps_Invoice *recv_invoices)
{
   amps_Package package;

   package = (amps_Package)calloc(1, sizeof(amps_PackageStruct));

   package -> num_recv = num_recv;
   package -> src = src;
   package -> recv_invoices = recv_invoices;

   package -> num_send = num_send;
   package -> dest = dest;
   package -> send_invoices = send_invoices;

   return package;
}

void amps_FreePackage(amps_Package package)
{
   if((package -> num_send + package -> num_recv) &&
      package -> commited)
   {
      free(package -> recv_requests);
      free(package -> status);
   }

   free(package);
}

#endif


