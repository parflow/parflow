/**********************************************************************
 *
 *  Please read the LICENSE file for the GNU Lesser General Public License.
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
 ***********************************************************************/

extern "C"{

#include <string.h>
#include "amps.h"

// #define DISABLE_GPU_PACKING
// #define ENFORCE_HOST_STAGING

#ifdef DISABLE_GPU_PACKING
#include "amps_gpupacking.c"
#else

void amps_gpu_freebufs(){
#ifdef ENFORCE_HOST_STAGING
  if (amps_device_globals.combuf_recv_size != 0) free(amps_device_globals.combuf_recv_host);
  if (amps_device_globals.combuf_send_size != 0) free(amps_device_globals.combuf_send_host);
#endif
  if (amps_device_globals.combuf_recv_size != 0) CUDA_ERRCHK(cudaFree(amps_device_globals.combuf_recv));
  if (amps_device_globals.combuf_send_size != 0) CUDA_ERRCHK(cudaFree(amps_device_globals.combuf_send));
}

char* amps_gpu_recvbuf_mpi(int size_total){
  /* check to see if enough space is already allocated */
  if (amps_device_globals.combuf_recv_size < size_total)
  {
#ifdef ENFORCE_HOST_STAGING
    if (amps_device_globals.combuf_recv_size != 0) free(amps_device_globals.combuf_recv_host);
    amps_device_globals.combuf_recv_host = (char*)malloc(size_total);
#endif
    if (amps_device_globals.combuf_recv_size != 0) CUDA_ERRCHK(cudaFree(amps_device_globals.combuf_recv));
    CUDA_ERRCHK(cudaMalloc((void**)&amps_device_globals.combuf_recv, size_total));
    amps_device_globals.combuf_recv_size = size_total;
  }
#ifdef ENFORCE_HOST_STAGING
  return amps_device_globals.combuf_recv_host;
#else
  return amps_device_globals.combuf_recv;
#endif
}

char* amps_gpu_recvbuf_packing(){
#ifdef ENFORCE_HOST_STAGING
  //copy host buffer to device
  CUDA_ERRCHK(cudaMemcpy(amps_device_globals.combuf_recv,
              amps_device_globals.combuf_recv_host,
                amps_device_globals.combuf_recv_size,
                  cudaMemcpyHostToDevice));
#endif
  return amps_device_globals.combuf_recv;
}

char* amps_gpu_sendbuf_mpi(){
#ifdef ENFORCE_HOST_STAGING
  //copy device buffer to host
  CUDA_ERRCHK(cudaMemcpy(amps_device_globals.combuf_send_host,
              amps_device_globals.combuf_send,
                amps_device_globals.combuf_send_size,
                  cudaMemcpyDeviceToHost));

  return amps_device_globals.combuf_send_host;
#else
  return amps_device_globals.combuf_send;
#endif
}

char* amps_gpu_sendbuf_packing(amps_Package package){
  int size_total = 0;
  for (int i = 0; i < package->num_send; i++){
    size_total += amps_sizeof_invoice(amps_CommWorld, package->send_invoices[i]);
  }
  /* check to see if enough space is already allocated            */
  if (amps_device_globals.combuf_send_size < size_total)
  {
#ifdef ENFORCE_HOST_STAGING
    if (amps_device_globals.combuf_send_size != 0) free(amps_device_globals.combuf_send_host);
    amps_device_globals.combuf_send_host = (char*)malloc(size_total);
#endif
    if (amps_device_globals.combuf_send_size != 0) CUDA_ERRCHK(cudaFree(amps_device_globals.combuf_send));
    CUDA_ERRCHK(cudaMalloc((void**)&amps_device_globals.combuf_send, size_total));
    amps_device_globals.combuf_send_size = size_total;
  }
  return amps_device_globals.combuf_send;
}

int _amps_gpupack_invcheck_vector(int dim, int *len, int type)
{
  if (dim == 0){
    switch (type){
      case AMPS_INVOICE_DOUBLE_CTYPE:
          return 0; //This case is supported
        break;
      default:
        // printf("ERROR at %s:%d: Only \"AMPS_INVOICE_DOUBLE_CTYPE\" is supported.", __FILE__, __LINE__);
        return __LINE__;
    }
  }
  else{
    for (int i = 0; i < len[dim]; i++){
      if(int error = _amps_gpupack_invcheck_vector(dim - 1, len, type))
        return error;
    }
    return 0;
  }
}

int _amps_gpupack_invcheck(amps_InvoiceEntry *ptr){
    switch (ptr->type)
    {
      case AMPS_INVOICE_CHAR_CTYPE:
        // printf("ERROR at %s:%d: This case is not supported.", __FILE__, __LINE__);
        return __LINE__;
        break;

      case AMPS_INVOICE_SHORT_CTYPE:
        // printf("ERROR at %s:%d: This case is not supported.", __FILE__, __LINE__);
        return __LINE__;
        break;

      case AMPS_INVOICE_INT_CTYPE:
        // printf("ERROR at %s:%d: This case is not supported.", __FILE__, __LINE__);
        return __LINE__;
        break;

      case AMPS_INVOICE_LONG_CTYPE:
        // printf("ERROR at %s:%d: This case is not supported.", __FILE__, __LINE__);
        return __LINE__;
        break;

      case AMPS_INVOICE_FLOAT_CTYPE:
        // printf("ERROR at %s:%d: This case is not supported.", __FILE__, __LINE__);
        return __LINE__;
        break;

      case AMPS_INVOICE_DOUBLE_CTYPE:
        // printf("ERROR at %s:%d: This case is not supported.", __FILE__, __LINE__);
        return __LINE__;
        break;

      default:
        int dim = (ptr->dim_type == AMPS_INVOICE_POINTER) ? *(ptr->ptr_dim) : ptr->dim;
        return _amps_gpupack_invcheck_vector(dim - 1, ptr->ptr_len, ptr->type - AMPS_INVOICE_LAST_CTYPE);
    }
}

int amps_gpupacking(amps_Invoice inv, char **buffer, int unpack){
  /* Get buffer location and its properties*/
  struct cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, (void *)*buffer);  

  /* Check that the data location (not MPI staging buffer) is accessible by the GPU */
  if(cudaGetLastError() != cudaSuccess || attributes.type < 2){
    // printf("ERROR at %s:%d: The MPI staging buffer location  is not accessible by the GPU(s).", __FILE__, __LINE__);
    return __LINE__;
  }
  amps_InvoiceEntry *ptr = inv->list;
  while (ptr != NULL)
  {
    if (ptr->len_type != AMPS_INVOICE_POINTER){
      // printf("ERROR at %s:%d: ptr->len_type must be a pointer.", __FILE__, __LINE__);
      return __LINE__;
    }
  
    if (ptr->stride_type != AMPS_INVOICE_POINTER){
      // printf("ERROR at %s:%d: ptr->stride_type must be a pointer.", __FILE__, __LINE__);
      return __LINE__;
    }
  
    if (ptr->data_type == AMPS_INVOICE_POINTER){
      // printf("ERROR at %s:%d: Overlayed invoices not supported.", __FILE__, __LINE__);
      return __LINE__;
    }
  
    if(int error = _amps_gpupack_invcheck(ptr)){
      return error;
    }
  
    // Preparations for the kernel launch
    int blocksize_x = 1;
    int blocksize_y = 1;
    int blocksize_z = 1;  
    int len_x = 1;
    int len_y = 1;
    int len_z = 1;  
    int stride_x = 0;
    int stride_y = 0;
    int stride_z = 0;
  
    int dim = (ptr->dim_type == AMPS_INVOICE_POINTER) ? *(ptr->ptr_dim) : ptr->dim;
    switch (dim)
    {
      case 3:
        blocksize_z = 2;
        len_z = ptr->ptr_len[2];
        stride_z = ptr->ptr_stride[2];
      case 2:
        blocksize_y = 8;
        len_y = ptr->ptr_len[1];
        stride_y = ptr->ptr_stride[1];
      case 1:
        blocksize_x = 8;
        len_x = ptr->ptr_len[0];
        stride_x = ptr->ptr_stride[0];
        break;
      default:
        // printf("ERROR at %s:%d: Only dimensions 1 - 3 are supported.", __FILE__, __LINE__);
        return __LINE__;
    }  
    
    /* Get data location and its properties*/
    char *data = (char*)ptr->data;
    struct cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, (void *)data);  
  
    /* Check that the data location (not MPI staging buffer) is accessible by the GPU */
    if(cudaGetLastError() == cudaSuccess && attributes.type > 1){
      dim3 grid = dim3(((len_x - 1) + blocksize_x) / blocksize_x, ((len_y - 1) + blocksize_y) / blocksize_y, ((len_z - 1) + blocksize_z) / blocksize_z);
      dim3 block = dim3(blocksize_x, blocksize_y, blocksize_z);
      if(unpack){
        UnpackingKernel<<<grid, block, 0, 0>>>((double*)*buffer, (double*)data, len_x, len_y, len_z, stride_x, stride_y, stride_z);
        inv->flags &= ~AMPS_PACKED;
      }
      else{
        PackingKernel<<<grid, block, 0, 0>>>((double*)*buffer, (double*)data, len_x, len_y, len_z, stride_x, stride_y, stride_z);
        inv->flags |= AMPS_PACKED;
      }
      CUDA_ERRCHK(cudaPeekAtLastError()); 
      CUDA_ERRCHK(cudaStreamSynchronize(0)); 
    }
    else
    {
      // printf("ERROR at %s:%d: The data location (not MPI staging buffer) is not accessible by the GPU(s).", __FILE__, __LINE__);
      return __LINE__;
    }
    *buffer = (char*)((double*)*buffer + len_x * len_y * len_z);  
    ptr = ptr->next;
  }
  return 0;
} 
#endif // DISABLE_GPU_PACKING
}   
    