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

void _amps_wait_exchange(amps_Handle handle)
{
  int notdone;
  int i;

  MPI_Status *status;

  if(handle -> package -> num_recv + handle -> package -> num_send)
  {
     status = calloc((handle -> package -> num_recv + 
		      handle -> package -> num_send), sizeof(MPI_Status));

     MPI_Waitall(handle -> package -> num_recv + handle -> package -> num_send,
		 handle -> package -> requests,
		 status);

     free(status);

     for(i = 0; i < handle -> package -> num_recv; i++)
     {
	MPI_Type_free(&(handle -> package -> recv_invoices[i] -> mpi_type));   
     }
     
     for(i = 0; i < handle -> package -> num_send; i++)
     {
	MPI_Type_free(&handle -> package -> send_invoices[i] -> mpi_type);
     }
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
   {
      amps_create_mpi_type(MPI_COMM_WORLD, package -> recv_invoices[i]);

      MPI_Type_commit(&(package -> recv_invoices[i] -> mpi_type));
     
      MPI_Irecv(MPI_BOTTOM, 1, package -> recv_invoices[i] -> mpi_type, 
		package -> src[i], 0, MPI_COMM_WORLD,
		&(package -> requests[i]));
   }

   /*--------------------------------------------------------------------
    * send out the data we have
    *--------------------------------------------------------------------*/
   for(i = 0; i < package -> num_send; i++)
   {
      amps_create_mpi_type(MPI_COMM_WORLD, package -> send_invoices[i]);

      MPI_Type_commit(&(package -> send_invoices[i] -> mpi_type));
      
      MPI_Isend(MPI_BOTTOM, 1, package -> send_invoices[i] -> mpi_type, 
		package -> dest[i], 0, MPI_COMM_WORLD,
		&(package -> requests[package -> num_recv +i]));
   }
   
   return( amps_NewHandle(NULL, 0, NULL, package));
}

#else

void _amps_wait_exchange(amps_Handle handle)
{
  int i;
  int num;

  num = handle -> package -> num_send + handle -> package -> num_recv;

  if(num)
  {
     if(handle -> package -> num_recv) 
     {
	for(i = 0; i <  handle -> package -> num_recv; i++)
	{
	   AMPS_CLEAR_INVOICE(handle -> package -> recv_invoices[i]);
	}
     }
	
     MPI_Waitall(num, handle -> package -> recv_requests, 
		 handle -> package -> status);
  }

#ifdef AMPS_MPI_PACKAGE_LOWSTORAGE
  /* Needed by the DEC's; need better memory allocation strategy */
  /* Need to uncommit packages when not in use */
  /* amps_Commit followed by amps_UnCommit ????? */
  if(handle -> package -> commited) 
  {
     for(i = 0; i < handle -> package -> num_recv; i++)
     {
	MPI_Type_free(&handle -> package -> recv_invoices[i] 
		      -> mpi_type);
    
	MPI_Request_free(&handle -> package -> recv_requests[i]);
     }
    
     for(i = 0; i < handle -> package -> num_send; i++)
     {
	MPI_Type_free(&handle -> package -> send_invoices[i] 
		      -> mpi_type);
	MPI_Request_free(&handle -> package -> send_requests[i]);
     }
    
     if(handle -> package -> recv_requests)
	free(handle -> package -> recv_requests);
     if(handle -> package -> status)
	free(handle -> package -> status);
    
     handle -> package -> commited = FALSE;
  }
#endif
}

/*===========================================================================*/
/**

The \Ref{amps_IExchangePackage} initiates the communication of the
invoices found in the {\bf package} structure that is passed in.  Once a
\Ref{amps_IExchangePackage} is issued it is illegal to access the
variables that are being communicated.  An \Ref{amps_IExchangePackage}
is always followed by an \Ref{amps_Wait} on the {\bf handle} that is
returned. 

{\large Example:}
\begin{verbatim}
// Initialize exchange of boundary points 
handle = amps_IExchangePackage(package);
 
// Compute on the "interior points"

// Wait for the exchange to complete 
amps_Wait(handle);
\end{verbatim}

{\large Notes:}

This routine can be optimized on some architectures so if your
communication can be formulated using it there might be
some performance advantages.

@memo Initiate package communication
@param package the collection of invoices to communicate
@return Handle for the asynchronous communication
*/
amps_Handle amps_IExchangePackage(amps_Package package)
{

   int i;
   int num;

   num = package -> num_send + package -> num_recv;

   /*-------------------------------------------------------------------
    * Check if we need to allocate the MPI types and requests 
    *------------------------------------------------------------------*/
   if(!package -> commited)
   {
      
      package -> commited = TRUE;
      
      /*--------------------------------------------------------------------
       * Allocate the arrays need for MPI 
       *--------------------------------------------------------------------*/
      if(num)
      {
	 package -> recv_requests = (MPI_Request *)calloc(num,
							  sizeof(MPI_Request));
	 
	 package -> status = (MPI_Status *)calloc(num,
						  sizeof(MPI_Status));
	 
	 package -> send_requests = package -> recv_requests + 
	    package -> num_recv;
      }
      
      /*--------------------------------------------------------------------
       * Set up the receive types and requests 
       *--------------------------------------------------------------------*/
      if( package -> num_recv)
      {
	 for(i = 0; i < package -> num_recv; i++)
	 {
	    
	    amps_create_mpi_type(MPI_COMM_WORLD, package -> recv_invoices[i]);
	    MPI_Type_commit(&(package -> recv_invoices[i] -> mpi_type));
	    MPI_Recv_init(MPI_BOTTOM, 1, 
			  package -> recv_invoices[i] -> mpi_type, 
			  package -> src[i], 0, MPI_COMM_WORLD,
			  &(package -> recv_requests[i]));
	 }
      }
      
      /*--------------------------------------------------------------------
       * Set up the send types and requests 
       *--------------------------------------------------------------------*/
      if(package -> num_send)
      {
	 for(i = 0; i < package -> num_send; i++)
	 {
	    amps_create_mpi_type(MPI_COMM_WORLD, 
				 package -> send_invoices[i]);
	    
	    MPI_Type_commit(&(package -> send_invoices[i] -> mpi_type));
	    
	    MPI_Ssend_init(MPI_BOTTOM, 1, 
			   package -> send_invoices[i] -> mpi_type, 
			   package -> dest[i], 0, MPI_COMM_WORLD,
			   &(package -> send_requests[i]));
	 }
      }
   }

   if(num)
   {
      /*--------------------------------------------------------------------
       * post send and receives 
       *--------------------------------------------------------------------*/
      MPI_Startall(num, package -> recv_requests);
   }

   return( amps_NewHandle(NULL, 0, NULL, package));
}

#endif

