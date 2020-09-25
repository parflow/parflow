/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
 *  LLC. Produced at the Lawrence Livermore National Laboratory. Written
 *  by the Parflow Team (see the CONTRIBUTORS file)
 *  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
 *
 *  This file is part of Parflow. For details, see
 *  http://www.llnl.gov/casc/parflow
 *
 *  Please read the COPYRIGHT file or Our Notice and the LICENSE file
 *  for the GNU Lesser General Public License.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License (as published
 *  by the Free Software Foundation) version 2.1 dated February 1999.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
 *  and conditions of the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 *  USA
 **********************************************************************EHEADER*/

#include "amps.h"
#include <assert.h>

#ifdef AMPS_MPI_NOT_USE_PERSISTENT

void _amps_wait_exchange(amps_Handle handle)
{
  int notdone;
  int i;

  MPI_Status *status;

  if (handle->package->num_recv + handle->package->num_send)
  {
    status = (MPI_Status*)calloc((handle->package->num_recv +
                                  handle->package->num_send), sizeof(MPI_Status));

    MPI_Waitall(handle->package->num_recv + handle->package->num_send,
                handle->package->requests,
                status);

    free(status);

    for (i = 0; i < handle->package->num_recv; i++)
    {
      if (handle->package->recv_invoices[i]->mpi_type != MPI_DATATYPE_NULL)
      {
        MPI_Type_free(&(handle->package->recv_invoices[i]->mpi_type));
      }

      MPI_Request_free(&handle->package->requests[i]);
    }

    for (i = 0; i < handle->package->num_send; i++)
    {
      if (handle->package->send_invoices[i]->mpi_type != MPI_DATATYPE_NULL)
      {
        MPI_Type_free(&handle->package->send_invoices[i]->mpi_type);
      }

      MPI_Request_free(&handle->package->requests[handle->package->num_recv + i]);
    }
  }
}

amps_Handle amps_IExchangePackage(amps_Package package)
{
  int i;

  /*--------------------------------------------------------------------
   * post receives for data to get
   *--------------------------------------------------------------------*/
  package->recv_remaining = 0;

  for (i = 0; i < package->num_recv; i++)
  {
    amps_create_mpi_type(MPI_COMM_WORLD, package->recv_invoices[i]);

    MPI_Type_commit(&(package->recv_invoices[i]->mpi_type));

    MPI_Irecv(MPI_BOTTOM, 1, package->recv_invoices[i]->mpi_type,
              package->src[i], 0, MPI_COMM_WORLD,
              &(package->requests[i]));
  }

  /*--------------------------------------------------------------------
   * send out the data we have
   *--------------------------------------------------------------------*/
  for (i = 0; i < package->num_send; i++)
  {
    amps_create_mpi_type(MPI_COMM_WORLD, package->send_invoices[i]);

    MPI_Type_commit(&(package->send_invoices[i]->mpi_type));

    MPI_Isend(MPI_BOTTOM, 1, package->send_invoices[i]->mpi_type,
              package->dest[i], 0, MPI_COMM_WORLD,
              &(package->requests[package->num_recv + i]));
  }

  return(amps_NewHandle(NULL, 0, NULL, package));
}

#else

void _amps_wait_exchange(amps_Handle handle)
{
  int i;
  int num;

  num = handle->package->num_send + handle->package->num_recv;

  if (num)
  {
    if (handle->package->num_recv)
    {
          MPI_Waitall(num, handle->package->recv_requests,
               handle->package->status);
      // int pos = 0;
      char *combuf = amps_gpu_recvbuf_mpi(handle->package);
      for (i = 0; i < handle->package->num_recv; i++)
      {
        // int size = amps_sizeof_invoice(amps_CommWorld, handle->package->recv_invoices[i]);
        // char *combuf = amps_gpu_recvbuf_packing(pos, size);
        // pos += size;
        int errchk = amps_gpupacking(handle->package->recv_invoices[i], &combuf, 1);       
        if(errchk)printf("GPU unpacking failed at line: %d\n", errchk);
        AMPS_CLEAR_INVOICE(handle->package->recv_invoices[i]);
      }
    }
  }

#ifdef AMPS_MPI_PACKAGE_LOWSTORAGE
  /* Needed by the DEC's; need better memory allocation strategy */
  /* Need to uncommit packages when not in use */
  /* amps_Commit followed by amps_UnCommit ????? */
  if (handle->package->commited)
  {
    for (i = 0; i < handle->package->num_recv; i++)
    {
      if (handle->package->recv_invoices[i]->mpi_type != MPI_DATATYPE_NULL)
      {
        MPI_Type_free(&(handle->package->recv_invoices[i]->mpi_type));
      }

      MPI_Request_free(&(handle->package->recv_requests[i]));
    }

    for (i = 0; i < handle->package->num_send; i++)
    {
      if (handle->package->send_invoices[i]->mpi_type != MPI_DATATYPE_NULL)
      {
        MPI_Type_free(&handle->package->send_invoices[i]->mpi_type);
      }

      MPI_Request_free(&(handle->package->send_requests[i]));
    }

    if (handle->package->recv_requests)
    {
      free(handle->package->recv_requests);
      handle->package->recv_requests = NULL;
    }
    if (handle->package->status)
    {
      free(handle->package->status);
      handle->package->status = NULL;
    }

    handle->package->commited = FALSE;
  }
#endif
}

/*===========================================================================*/
/**
 *
 * The \Ref{amps_IExchangePackage} initiates the communication of the
 * invoices found in the {\bf package} structure that is passed in.  Once a
 * \Ref{amps_IExchangePackage} is issued it is illegal to access the
 * variables that are being communicated.  An \Ref{amps_IExchangePackage}
 * is always followed by an \Ref{amps_Wait} on the {\bf handle} that is
 * returned.
 *
 * {\large Example:}
 * \begin{verbatim}
 * // Initialize exchange of boundary points
 * handle = amps_IExchangePackage(package);
 *
 * // Compute on the "interior points"
 *
 * // Wait for the exchange to complete
 * amps_Wait(handle);
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * This routine can be optimized on some architectures so if your
 * communication can be formulated using it there might be
 * some performance advantages.
 *
 * @memo Initiate package communication
 * @param package the collection of invoices to communicate
 * @return Handle for the asynchronous communication
 */
amps_Handle amps_IExchangePackage(amps_Package package)
{
  int num = package->num_send + package->num_recv;

  /*-------------------------------------------------------------------
  * Check if we need to allocate the MPI types and requests
  *------------------------------------------------------------------*/
  if (!package->commited)
  {
    package->commited = TRUE;

    /*--------------------------------------------------------------------
     * Allocate the arrays need for MPI
     *--------------------------------------------------------------------*/
    if (num)
    {
      package->recv_requests = (MPI_Request*)calloc((size_t)(num),
                                                    sizeof(MPI_Request));

      package->status = (MPI_Status*)calloc((size_t)(num),
                                            sizeof(MPI_Status));

      package->send_requests = package->recv_requests +
                               package->num_recv;
    }

    /*--------------------------------------------------------------------
     * Set up the receive types and requests
     *--------------------------------------------------------------------*/
    if (package->num_recv)
    {
      char *combuf;
      char *gpubuf = amps_gpu_recvbuf_mpi(package);
      // int pos = 0;
      int size;
      for (int i = 0; i < package->num_recv; i++)
      {
        // int size = amps_sizeof_invoice(amps_CommWorld, package->recv_invoices[i]);
        // char *combuf = amps_gpu_recvbuf_mpi(pos, size);

        if(gpubuf != NULL){
          // size = amps_sizeof_invoice(amps_CommWorld, package->recv_invoices[i]);
          combuf = gpubuf;
          int errchk = amps_gpupacking(package->recv_invoices[i], &gpubuf, 2);
          if(errchk) printf("GPU check failed at line: %d\n", errchk);
          size = gpubuf - combuf;
          package->recv_invoices[i]->mpi_type = MPI_BYTE;
          MPI_Type_commit(&(package->recv_invoices[i]->mpi_type));
        }
        else{ 
          printf("RECVBUF FAILED failed\n");
          exit(1);
          amps_create_mpi_type(MPI_COMM_WORLD, package->recv_invoices[i]);
          MPI_Type_commit(&(package->recv_invoices[i]->mpi_type));
          combuf = NULL;
          size = 1;
        }
        // printf("Rank: %d, Recv size: %d, Msg: %d\n",amps_rank,size, i);
        MPI_Request *request_ptr = &(package->recv_requests[i]);
        MPI_Recv_init(combuf, size, package->recv_invoices[i]->mpi_type,
                      package->src[i], 0, MPI_COMM_WORLD, request_ptr);
      }
    }

    /*--------------------------------------------------------------------
     * Set up the send types and requests
     *--------------------------------------------------------------------*/
    if (package->num_send)
    {
      char *combuf;
      char *gpubuf = amps_gpu_sendbuf_packing(package);
      // int pos = 0;
      for (int i = 0; i < package->num_send; i++)
      {
        int errchk = -1;
        int size; 
	      if(gpubuf != NULL){
          combuf = gpubuf;
          errchk = amps_gpupacking(package->send_invoices[i], &gpubuf, 0);
        }
    
        if(gpubuf != NULL && errchk == 0){
          // size = amps_sizeof_invoice(amps_CommWorld, package->send_invoices[i]);
          size = gpubuf - combuf;
          // combuf = amps_gpu_sendbuf_mpi(pos, size);
          package->send_invoices[i]->mpi_type = MPI_BYTE;
          MPI_Type_commit(&(package->send_invoices[i]->mpi_type));
          // pos += size;
          // assert(gpubuf == combuf + size);
        }
        else{
          if(errchk) printf("GPU packing failed at line: %d\n", errchk);
          amps_create_mpi_type(MPI_COMM_WORLD,
                               package->send_invoices[i]);
  
          MPI_Type_commit(&(package->send_invoices[i]->mpi_type));
          combuf = NULL;
          size = 1;
        }
        // printf("Rank: %d, Send size: %d, Msg: %d\n",amps_rank,size, i);
        MPI_Request* request_ptr = &(package->send_requests[i]);
        MPI_Ssend_init(combuf, size, package->send_invoices[i]->mpi_type,
                       package->dest[i], 0, MPI_COMM_WORLD, request_ptr);
      }
    }
    if (num)
    {
      /*--------------------------------------------------------------------
       * post send and receives
       *--------------------------------------------------------------------*/
      MPI_Startall(num, package->recv_requests);
    }
  }

  return(amps_NewHandle(amps_CommWorld, 0, NULL, package));
}

#endif

