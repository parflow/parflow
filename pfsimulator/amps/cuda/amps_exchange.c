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

int _amps_send_sizes(amps_Package package, int **sizes)
{
  int size_acc;
  int streams_hired = 0;

  *sizes = (int*)calloc(package->num_send, sizeof(int));

  size_acc = 0;
  for (int i = 0; i < package->num_send; i++)
  {
    (*sizes)[i] = amps_sizeof_invoice(amps_CommWorld, package->send_invoices[i]);
    size_acc += (*sizes)[i];
  }

  /* check to see if enough space is already allocated            */
  if (amps_device_globals.combuf_send_size < size_acc)
  {
    if (amps_device_globals.combuf_send_size != 0)
      CUDA_ERRCHK(cudaFree(amps_device_globals.combuf_send));

    CUDA_ERRCHK(cudaMalloc((void**)&amps_device_globals.combuf_send, size_acc));
    amps_device_globals.combuf_send_size = size_acc;
  }

  // CUDA_ERRCHK(cudaStreamSynchronize(0));

  size_acc = 0;
  for (int i = 0; i < package->num_send; i++)
  {
    package->send_invoices[i]->combuf = &amps_device_globals.combuf_send[size_acc];
    amps_pack(amps_CommWorld, package->send_invoices[i], package->send_invoices[i]->combuf, &streams_hired);

    MPI_Isend(&((*sizes)[i]), 1, MPI_INT, package->dest[i],
              0, amps_CommWorld,
              &(package->send_requests[i]));
    size_acc += (*sizes)[i];
  }

  for (int i = 0; i < streams_hired; i++)
  {
    if (i < amps_device_max_streams)
      CUDA_ERRCHK(cudaStreamSynchronize(amps_device_globals.stream[i]));
  }

  return(0);
}
int _amps_recv_sizes(amps_Package package)
{
  int *sizes;
  int size_acc;

  MPI_Status status;

  sizes = (int*)calloc(package->num_recv, sizeof(int));

  size_acc = 0;
  for (int i = 0; i < package->num_recv; i++)
  {
    MPI_Recv(&sizes[i], 1, MPI_INT, package->src[i], 0,
             amps_CommWorld, &status);
    size_acc += sizes[i];
  }

  /* check to see if enough space is already allocated            */
  if (amps_device_globals.combuf_recv_size < size_acc)
  {
    if (amps_device_globals.combuf_recv_size != 0)
      CUDA_ERRCHK(cudaFree(amps_device_globals.combuf_recv));

    CUDA_ERRCHK(cudaMalloc((void**)&(amps_device_globals.combuf_recv), size_acc));
    CUDA_ERRCHK(cudaMemset(amps_device_globals.combuf_recv, 0, size_acc));
    amps_device_globals.combuf_recv_size = size_acc;
  }

  size_acc = 0;
  for (int i = 0; i < package->num_recv; i++)
  {
    package->recv_invoices[i]->combuf = &amps_device_globals.combuf_recv[size_acc];
    MPI_Recv_init(package->recv_invoices[i]->combuf, sizes[i],
                  MPI_BYTE, package->src[i], 1, amps_CommWorld,
                  &(package->recv_requests[i]));
    size_acc += sizes[i];
  }
  free(sizes);

  return(0);
}

void _amps_wait_exchange(amps_Handle handle)
{
  int i;
  int num;
  int streams_hired = 0;

  num = handle->package->num_send + handle->package->num_recv;

  if (num)
  {
    MPI_Waitall(num, handle->package->recv_requests,
                handle->package->status);
    if (handle->package->num_recv)
    {
      for (i = 0; i < handle->package->num_recv; i++)
      {
        amps_unpack(amps_CommWorld, handle->package->recv_invoices[i],
                    (char *)handle->package->recv_invoices[i]->combuf, &streams_hired);
      }
      for (int i = 0; i < streams_hired; i++)
      {
        if (i < amps_device_max_streams)
          CUDA_ERRCHK(cudaStreamSynchronize(amps_device_globals.stream[i]));
      }
    }
    for (i = 0; i < handle->package->num_recv; i++)
    {
      if (handle->package->recv_requests[i] != MPI_REQUEST_NULL)
        MPI_Request_free(&(handle->package->recv_requests[i]));
    }
    for (i = 0; i < handle->package->num_send; i++)
    {
      if (handle->package->send_requests[i] != MPI_REQUEST_NULL)
        MPI_Request_free(&(handle->package->send_requests[i]));
    }
  }
}

amps_Handle amps_IExchangePackage(amps_Package package)
{
  int i;
  int *send_sizes;

  MPI_Status *status_array;

  if (package->num_send)
    _amps_send_sizes(package, &send_sizes);

  if (package->num_recv)
    _amps_recv_sizes(package);

  /*--------------------------------------------------------------------
   * end the sending of sizes
   *--------------------------------------------------------------------*/

  if (package->num_send)
  {
    status_array = (MPI_Status*)calloc(package->num_send,
                                       sizeof(MPI_Status));
    MPI_Waitall(package->num_send, package->send_requests, status_array);
    free(status_array);
  }

  if (package->num_send)
  {
    for (i = 0; i < package->num_send; i++)
    {
      MPI_Send_init(package->send_invoices[i]->combuf,
                    send_sizes[i], MPI_BYTE, package->dest[i], 1,
                    amps_CommWorld, &(package->send_requests[i]));
    }

    free(send_sizes);
  }

  /*--------------------------------------------------------------------
   * post receives for data to get
   *--------------------------------------------------------------------*/

  if (package->num_recv)
    MPI_Startall(package->num_recv, package->recv_requests);

  /*--------------------------------------------------------------------
   * send out the data we have
   *--------------------------------------------------------------------*/
  if (package->num_send)
    MPI_Startall(package->num_send, package->send_requests);

  return(amps_NewHandle(amps_CommWorld, 0, NULL, package));
}
