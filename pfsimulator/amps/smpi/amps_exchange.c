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

#include <assert.h>
#include <sys/times.h>
#include "amps.h"

int _amps_send_sizes(amps_Package package, int **sizes, int **tags)
{
  char *gpubuf = amps_gpu_sendbuf(package);
  char *gpubuf_assert = gpubuf;

  *sizes = (int*)calloc(package->num_send, sizeof(int));
  *tags = (int*)malloc(package->num_send * sizeof(int));

  for (int i = 0; i < package->num_send; i++)
  {
    (*tags)[i] = 1;

	  if(gpubuf != NULL)
      (*tags)[i] = amps_gpupacking(package->send_invoices[i], &gpubuf, 0);

    if(gpubuf == NULL || (*tags)[i]){
      // if((*tags)[i] > 0) printf("GPU packing failed at line: %d\n", (*tags)[i]);
      (*sizes)[i] = amps_pack(amps_CommWorld, package->send_invoices[i], 
          (char**)&package->send_invoices[i]->combuf);
    }
    else{
      (*sizes)[i] = amps_sizeof_invoice(amps_CommWorld, package->send_invoices[i]);
      gpubuf_assert += (*sizes)[i];
    }
    
    MPI_Isend(&((*sizes)[i]), 1, MPI_INT, package->dest[i],
              (*tags)[i], amps_CommWorld,
              &(package->send_requests[i]));
  }
  assert(gpubuf == gpubuf_assert);

  return(0);
}

#ifdef AMPS_MPI_NOT_USE_PERSISTENT

int _amps_recv_sizes(amps_Package package)
{
  int i;
  int size;

  MPI_Status status;

  for (i = 0; i < package->num_recv; i++)
  {
    MPI_Recv(&size, 1, MPI_INT, package->src[i], 0,
             amps_CommWorld, &status);

    package->recv_invoices[i]->combuf =
      (char*)calloc(size, sizeof(char *));

    /* Post receives for incoming byte buffers */

    MPI_Irecv(package->recv_invoices[i]->combuf, size,
              MPI_BYTE, package->src[i], 1,
              amps_CommWorld, &(package->recv_requests[i]));
  }

  return(0);
}

void _amps_wait_exchange(amps_Handle handle)
{
  int i;
  int size;
  int *flags;

  MPI_Status *status;

/*
 * if (handle -> package -> num_recv)
 *   _amps_probes(handle -> package);
 */

  if (handle->package->num_recv + handle->package->num_send)
  {
    status = calloc((handle->package->num_recv +
                     handle->package->num_send), sizeof(MPI_Status));

    MPI_Waitall(handle->package->num_recv + handle->package->num_send,
                handle->package->recv_requests, status);

    fflush(NULL);

    free(status);

    for (i = 0; i < handle->package->num_recv; i++)
    {
      amps_unpack(amps_CommWorld, handle->package->recv_invoices[i],
                  handle->package->recv_invoices[i]->combuf);
      AMPS_PACK_FREE_LETTER(amps_CommWorld,
                            handle->package->recv_invoices[i],
                            handle->package->recv_invoices[i]->combuf);
    }
/*
 *
 *   for(i = 0; i < handle -> package -> num_send; i++)
 *   {
 *      MPI_Type_free(&handle -> package -> send_invoices[i] -> mpi_type);
 *   }
 */
  }
}

amps_Handle amps_IExchangePackage(amps_Package package)
{
  int i;  assert(gpubuf == gpubuf_assert);
  int done;
  int *flags;
  char *buffer;
  int size;
  struct timeval tm;

  MPI_Status status;

  /*--------------------------------------------------------------------
   * send out the data we have
   *--------------------------------------------------------------------*/
  for (i = 0; i < package->num_send; i++)
  {
    size = amps_pack(amps_CommWorld, package->send_invoices[i],
                     &buffer);
    MPI_Isend(package->send_invoices[i]->combuf, size,
              MPI_BYTE, package->dest[i], 1,
              amps_CommWorld, &(package->send_requests[i]));
  }

/*
 * fflush(NULL);
 * gettimeofday(&tm, 0);
 * amps_Printf("I'm at the Barrier %ld %ld\n", tm.tv_usec, tm.tv_sec);
 * fflush(NULL);
 * MPI_Barrier(amps_CommWorld);
 * fflush(NULL);
 * gettimeofday(&tm, 0);
 * amps_Printf("I'm beyond the Barrier %ld %ld\n", tm.tv_usec, tm.tv_sec);
 * fflush(NULL);
 */

  if (package->num_recv)
  {
    flags = (int*)calloc(package->num_recv, sizeof(int));

    for (i = 0; i < package->num_recv; i++)
    {
      flags[i] = 0;
    }

    done = 0;
    while (!done)
    {
      done = 1;
      for (i = 0; i < package->num_recv; i++)
      {
        if (!flags[i])
        {
          MPI_Iprobe(package->src[i], 1, amps_CommWorld, &(flags[i]), &status);

          if (flags[i])
          {
            MPI_Get_count(&status, MPI_BYTE, &size);
            package->recv_invoices[i]->combuf =
              (char*)calloc(size, sizeof(char *));

            MPI_Irecv(package->recv_invoices[i]->combuf, size,
                      MPI_BYTE, package->src[i], 1,
                      amps_CommWorld, &(package->recv_requests[i]));
          }
          else
            done = 0;
        }
      }
    }

    free(flags);
  }

/*
 * fflush(NULL);
 * gettimeofday(&tm, 0);
 * amps_Printf("IEX done %ld\n", tm.tv_sec);
 * fflush(NULL);
 */
  return(amps_NewHandle(NULL, 0, NULL, package));
}

#else

int _amps_recv_sizes(amps_Package package)
{
  char *combuf;
  char *gpubuf;
  int *sizes;
  int *tags;
  int size_total = 0;

  sizes = (int*)calloc(package->num_recv, sizeof(int));
  tags = (int*)calloc(package->num_recv, sizeof(int));

  for (int i = 0; i < package->num_recv; i++)
  {
    MPI_Status status;
    MPI_Recv(&sizes[i], 1, MPI_INT, package->src[i],
          MPI_ANY_TAG, amps_CommWorld, &status);
    
    tags[i] = status.MPI_TAG;
    if(tags[i] == 0)
      size_total += sizes[i];
  }

  gpubuf = amps_gpu_recvbuf(size_total);

  for (int i = 0; i < package->num_recv; i++)
  {
    if(tags[i] == 0){
      combuf = gpubuf;
      gpubuf += sizes[i];
    }
    else{ 
      combuf = package->recv_invoices[i]->combuf 
             = (char*)calloc(sizes[i], sizeof(char *));
    }
 
    MPI_Recv_init(combuf, sizes[i], MPI_BYTE, package->src[i], 
        1, amps_CommWorld, &(package->recv_requests[i]));
  }
  free(sizes);
  free(tags);

  return(0);
}

void _amps_wait_exchange(amps_Handle handle)
{
  int i;
  int num;

  num = handle->package->num_send + handle->package->num_recv;

  MPI_Waitall(num, handle->package->recv_requests,
              handle->package->status);

  if (num)
  {
    if (handle->package->num_recv)
    {
      char *gpubuf = amps_gpu_recvbuf(0);
      for (i = 0; i < handle->package->num_recv; i++)
      {
        //if GPU unpacking fails, switch to original method
        if(amps_gpupacking(handle->package->recv_invoices[i], &gpubuf, 1)){
          // printf("GPU unpacking failed at line: %d\n", gpufail);
          amps_unpack(amps_CommWorld, handle->package->recv_invoices[i],
                    handle->package->recv_invoices[i]->combuf);
          AMPS_PACK_FREE_LETTER(amps_CommWorld,
                             handle->package->recv_invoices[i],
                             handle->package->recv_invoices[i]->combuf); 
        }
        // AMPS_CLEAR_INVOICE(handle -> package -> recv_invoices[i]);
      }
    }
  }
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
  int *send_sizes;
  int *send_tags;

  MPI_Status *status_array;

  if (package->num_send)
    _amps_send_sizes(package, &send_sizes, &send_tags);

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
    char *combuf;
    char *gpubuf = amps_gpu_sendbuf(package);
    for (i = 0; i < package->num_send; i++)
    {
      if(send_tags[i] == 0){
        combuf = gpubuf;
        gpubuf += send_sizes[i];
      }
      else{ 
        combuf = package->send_invoices[i]->combuf;
      }
      MPI_Send_init(combuf, send_sizes[i], MPI_BYTE, package->dest[i], 
                    1, amps_CommWorld, &(package->send_requests[i]));
    }

    free(send_sizes);
    free(send_tags);
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

#endif
