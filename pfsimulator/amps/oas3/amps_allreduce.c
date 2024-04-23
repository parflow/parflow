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

#include <strings.h>

/*===========================================================================*/
/**
 * The collective operation \Ref{amps_AllReduce} is used to take information
 * from each node of a context, perform an operation on the data, and return
 * the combined result to the all the nodes.  This operation is also called
 * a combine in some message passing systems.  The supported operation are
 * \Ref{amps_Max}, \Ref{amps_Min} and \Ref{amps_Add}.
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_Invoice invoice;
 * double       d;
 * int          i;
 *
 * invoice = amps_NewInvoice("%i%d", &i, &d);
 *
 * // find maximum of i and d on all nodes
 * amps_AllReduce(amps_CommWorld, invoice, amps_Max);
 *
 * // find sum of i and d on all nodes
 * amps_AllReduce(amps_CommWorld, invoice, amps_Add);
 *
 * amps_FreeInvoice(invoice);
 *
 * \end{verbatim}
 *
 * @memo Reduction Operation
 * @param comm communication context for the reduction [IN]
 * @param invoice invoice to reduce [IN/OUT]
 * @param operation reduction operation to perform [IN]
 * @return Error code
 */
int amps_AllReduce(amps_Comm comm, amps_Invoice invoice, MPI_Op operation)
{
  amps_InvoiceEntry *ptr;

  int len;
  int stride;

  char *data;
  char *in_buffer;
  char *out_buffer;

  char *ptr_src;
  char *ptr_dest;

  MPI_Datatype mpi_type = MPI_CHAR;
  int element_size = 0;

  ptr = invoice->list;

  while (ptr != NULL)
  {
    if (ptr->len_type == AMPS_INVOICE_POINTER)
      len = *(ptr->ptr_len);
    else
      len = ptr->len;

    if (ptr->stride_type == AMPS_INVOICE_POINTER)
      stride = *(ptr->ptr_stride);
    else
      stride = ptr->stride;

    if (ptr->data_type == AMPS_INVOICE_POINTER)
      data = *((char**)(ptr->data));
    else
      data = (char*)ptr->data;


    switch (ptr->type)
    {
      case AMPS_INVOICE_BYTE_CTYPE:
	      mpi_type = MPI_BYTE;
        element_size = sizeof(char);
        break;

      case AMPS_INVOICE_CHAR_CTYPE:
        mpi_type = MPI_CHAR;
        element_size = sizeof(char);
        break;

      case AMPS_INVOICE_SHORT_CTYPE:
        mpi_type = MPI_SHORT;
        element_size = sizeof(short);
        break;

      case AMPS_INVOICE_INT_CTYPE:
        mpi_type = MPI_INT;
        element_size = sizeof(int);
        break;

      case AMPS_INVOICE_LONG_CTYPE:
        mpi_type = MPI_LONG;
        element_size = sizeof(long);
        break;

      case AMPS_INVOICE_FLOAT_CTYPE:
        mpi_type = MPI_FLOAT;
        element_size = sizeof(float);
        break;

      case AMPS_INVOICE_DOUBLE_CTYPE:
        mpi_type = MPI_DOUBLE;
        element_size = sizeof(double);
        break;

      default:
        printf("AMPS Operation not supported\n");
    }

    in_buffer = (char*)malloc((size_t)(element_size * len));
    out_buffer = (char*)malloc((size_t)(element_size * len));

#ifdef PARFLOW_HAVE_CUDA
    /* Prefetch device data into host memory */
    struct cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, (void *)data);

    if(cudaGetLastError() == cudaSuccess && attributes.type > 1){
      if (stride == 1)
        CUDA_ERRCHK(cudaMemPrefetchAsync(data, (size_t)len * element_size, cudaCpuDeviceId, 0));
      else
        for (ptr_src = data;
             ptr_src < data + len * stride * element_size;
             ptr_src += stride * element_size)
          CUDA_ERRCHK(cudaMemPrefetchAsync(ptr_src, (size_t)element_size, cudaCpuDeviceId, 0));

      CUDA_ERRCHK(cudaStreamSynchronize(0)); 
    }
#endif

    /* Copy into a contigous buffer */
    if (stride == 1)
      bcopy(data, in_buffer, (size_t)(len * element_size));
    else
      for (ptr_src = data, ptr_dest = in_buffer;
           ptr_src < data + len * stride * element_size;
           ptr_src += stride * element_size, ptr_dest += element_size)
        bcopy(ptr_src, ptr_dest, (size_t)(element_size));

    MPI_Allreduce(in_buffer, out_buffer, len, mpi_type, operation, comm);

    /* Copy back into user variables */
    if (stride == 1)
      bcopy(out_buffer, data, (size_t)(len * element_size));
    else
      for (ptr_src = out_buffer, ptr_dest = data;
           ptr_src < out_buffer + len * element_size;
           ptr_src += element_size, ptr_dest += stride * element_size)
        bcopy(ptr_src, ptr_dest, (size_t)(element_size));

    free(in_buffer);
    free(out_buffer);

    ptr = ptr->next;
  }
  return 0;
}
