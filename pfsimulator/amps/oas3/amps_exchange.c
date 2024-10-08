/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
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

/* This CUDA stuff could be combined with AMPS_MPI_NOT_USE_PERSISTENT case */
#if defined(PARFLOW_HAVE_CUDA) || defined(PARFLOW_HAVE_KOKKOS)

void _amps_wait_exchange(amps_Handle handle)
{
  char *combuf;
  int i;
  int size;

  if (handle->package->num_recv + handle->package->num_send)
  {
    MPI_Waitall(handle->package->num_recv + handle->package->num_send,
                handle->package->recv_requests, handle->package->status);
    for (i = 0; i < handle->package->num_recv; i++)
    {
      amps_gpupacking(AMPS_UNPACK,
                      handle->package->recv_invoices[i],
                      i, &combuf, &size);
    }
    for (i = 0; i < handle->package->num_recv; i++)
    {
      amps_gpu_sync_streams(i);
      AMPS_CLEAR_INVOICE(handle->package->recv_invoices[i]);
    }
  }

  for (i = 0; i < handle->package->num_recv; i++)
  {
    MPI_Datatype type = handle->package->recv_invoices[i]->mpi_type;
    if (type != MPI_DATATYPE_NULL && type != MPI_BYTE)
      MPI_Type_free(&(handle->package->recv_invoices[i]->mpi_type));
    if (handle->package->recv_requests[i] != MPI_REQUEST_NULL)
      MPI_Request_free(&(handle->package->recv_requests[i]));
  }
  for (i = 0; i < handle->package->num_send; i++)
  {
    MPI_Datatype type = handle->package->send_invoices[i]->mpi_type;
    if (type != MPI_DATATYPE_NULL && type != MPI_BYTE)
      MPI_Type_free(&handle->package->send_invoices[i]->mpi_type);
    if (handle->package->send_requests[i] != MPI_REQUEST_NULL)
      MPI_Request_free(&(handle->package->send_requests[i]));
  }
}

amps_Handle amps_IExchangePackage(amps_Package package)
{
  char **combuf;
  int *size;
  int errchk;
  int i;

  if (package->num_send > 0)
  {
    combuf = (char**)malloc(package->num_send * sizeof(char*));
    size = (int*)malloc(package->num_send * sizeof(int));
  }
  else
  {
    combuf = (char**)malloc(sizeof(char*));
    size = (int*)malloc(sizeof(int));
  }

  /*--------------------------------------------------------------------
   * post receives for data to get
   *--------------------------------------------------------------------*/
  for (i = 0; i < package->num_recv; i++)
  {
    errchk = amps_gpupacking(AMPS_GETRBUF, package->recv_invoices[i],
                             i, &combuf[0], &size[0]);
    if (errchk == 0)
    {
      package->recv_invoices[i]->mpi_type = MPI_BYTE;
    }
    else
    {
      combuf[0] = NULL;
      size[0] = 1;
      amps_create_mpi_type(oas3Comm, package->recv_invoices[i]);
      MPI_Type_commit(&(package->recv_invoices[i]->mpi_type));
    }

    MPI_Irecv(combuf[0], size[0], package->recv_invoices[i]->mpi_type,
              package->src[i], 0, oas3Comm,
              &(package->recv_requests[i]));
  }

  /*--------------------------------------------------------------------
   * send out the data we have
   *--------------------------------------------------------------------*/
  for (i = 0; i < package->num_send; i++)
  {
    errchk = amps_gpupacking(AMPS_PACK, package->send_invoices[i],
                             i, &combuf[i], &size[i]);
    if (errchk == 0)
    {
      package->send_invoices[i]->mpi_type = MPI_BYTE;
    }
    else
    {
      combuf[i] = NULL;
      size[i] = 1;
      amps_create_mpi_type(oas3Comm, package->send_invoices[i]);
      MPI_Type_commit(&(package->send_invoices[i]->mpi_type));
    }
  }
  for (i = 0; i < package->num_send; i++)
  {
    amps_gpu_sync_streams(i);
    MPI_Isend(combuf[i], size[i], package->send_invoices[i]->mpi_type,
              package->dest[i], 0, oas3Comm,
              &(package->send_requests[i]));
  }
  free(combuf);
  free(size);

  return(amps_NewHandle(amps_CommWorld, 0, NULL, package));
}

#elif defined(AMPS_MPI_NOT_USE_PERSISTENT)

void _amps_wait_exchange(amps_Handle handle)
{
  int i;

  if (handle->package->num_recv + handle->package->num_send)
  {
    MPI_Waitall(handle->package->num_recv + handle->package->num_send,
                handle->package->recv_requests, handle->package->status);

    for (i = 0; i < handle->package->num_recv; i++)
    {
      if (handle->package->recv_invoices[i]->mpi_type != MPI_DATATYPE_NULL)
        MPI_Type_free(&(handle->package->recv_invoices[i]->mpi_type));
      if (handle->package->recv_requests[i] != MPI_REQUEST_NULL)
        MPI_Request_free(&handle->package->recv_requests[i]);
    }

    for (i = 0; i < handle->package->num_send; i++)
    {
      if (handle->package->send_invoices[i]->mpi_type != MPI_DATATYPE_NULL)
        MPI_Type_free(&handle->package->send_invoices[i]->mpi_type);
      if (handle->package->send_requests[i] != MPI_REQUEST_NULL)
        MPI_Request_free(&handle->package->send_requests[i]);
    }
  }
}

amps_Handle amps_IExchangePackage(amps_Package package)
{
  int i;

  /*--------------------------------------------------------------------
   * post receives for data to get
   *--------------------------------------------------------------------*/
  for (i = 0; i < package->num_recv; i++)
  {
    amps_create_mpi_type(oas3Comm, package->recv_invoices[i]);

    MPI_Type_commit(&(package->recv_invoices[i]->mpi_type));

    MPI_Irecv(MPI_BOTTOM, 1, package->recv_invoices[i]->mpi_type,
              package->src[i], 0, oas3Comm,
              &(package->recv_requests[i]));
  }

  /*--------------------------------------------------------------------
   * send out the data we have
   *--------------------------------------------------------------------*/
  for (i = 0; i < package->num_send; i++)
  {
    amps_create_mpi_type(oas3Comm, package->send_invoices[i]);

    MPI_Type_commit(&(package->send_invoices[i]->mpi_type));

    MPI_Isend(MPI_BOTTOM, 1, package->send_invoices[i]->mpi_type,
              package->dest[i], 0, oas3Comm,
              &(package->send_requests[i]));
  }

  return(amps_NewHandle(amps_CommWorld, 0, NULL, package));
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
      for (i = 0; i < handle->package->num_recv; i++)
      {
        AMPS_CLEAR_INVOICE(handle->package->recv_invoices[i]);
      }
    }

    MPI_Waitall(num, handle->package->recv_requests,
                handle->package->status);
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
  int i;
  int num;

  num = package->num_send + package->num_recv;

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
      for (i = 0; i < package->num_recv; i++)
      {
        amps_create_mpi_type(oas3Comm, package->recv_invoices[i]);
        MPI_Type_commit(&(package->recv_invoices[i]->mpi_type));

        // Temporaries needed by insure++
        MPI_Datatype type = package->recv_invoices[i]->mpi_type;
        MPI_Request *request_ptr = &(package->recv_requests[i]);
        MPI_Recv_init(MPI_BOTTOM, 1,
                      type,
                      package->src[i], 0, oas3Comm,
                      request_ptr);
      }
    }

    /*--------------------------------------------------------------------
     * Set up the send types and requests
     *--------------------------------------------------------------------*/
    if (package->num_send)
    {
      for (i = 0; i < package->num_send; i++)
      {
        amps_create_mpi_type(oas3Comm,
                             package->send_invoices[i]);

        MPI_Type_commit(&(package->send_invoices[i]->mpi_type));

        // Temporaries needed by insure++
        MPI_Datatype type = package->send_invoices[i]->mpi_type;
        MPI_Request* request_ptr = &(package->send_requests[i]);
        MPI_Ssend_init(MPI_BOTTOM, 1,
                       type,
                       package->dest[i], 0, oas3Comm,
                       request_ptr);
      }
    }
  }

  if (num)
  {
    /*--------------------------------------------------------------------
     * post send and receives
     *--------------------------------------------------------------------*/
    MPI_Startall(num, package->recv_requests);
  }


  return(amps_NewHandle(amps_CommWorld, 0, NULL, package));
}

#endif

