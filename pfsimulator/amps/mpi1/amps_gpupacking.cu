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

amps_GpuBuffer amps_gpu_recvbuf = {.buf = NULL, 
                                   .buf_host = NULL,
                                   .buf_size = NULL,
                                   .num_bufs = 0};

amps_GpuBuffer amps_gpu_sendbuf = {.buf = NULL, 
                                   .buf_host = NULL,
                                   .buf_size = NULL,
                                   .num_bufs = 0};

// #define DISABLE_GPU_PACKING
// #define ENFORCE_HOST_STAGING

#ifdef DISABLE_GPU_PACKING
#include "amps_gpupacking.c"
#else

static void _amps_gpubuf_free(amps_GpuBuffer *gpubuf){
  if(gpubuf->num_bufs > 0){
    for(int i = 0; i < gpubuf->num_bufs; i++)
      if(gpubuf->buf_size[i] > 0)
        CUDA_ERRCHK(cudaFree(gpubuf->buf[i]));
    free(gpubuf->buf);
    free(gpubuf->buf_size);
  }
}

void amps_gpu_freebufs(){
  _amps_gpubuf_free(&amps_gpu_recvbuf);
  _amps_gpubuf_free(&amps_gpu_sendbuf);
}

static char* _amps_gpubuf_realloc(amps_GpuBuffer *gpubuf, int inv_num, int pos, int size){
  //check if arrays are big enough for inv_num
  if(inv_num >= gpubuf->num_bufs){
    char **buf_array = (char**)malloc((inv_num + 1) * sizeof(char*));
    int *size_array = (int*)malloc((inv_num + 1) * sizeof(int));
    if(gpubuf->num_bufs != 0){
      memcpy(buf_array, 
             gpubuf->buf, 
               gpubuf->num_bufs * sizeof(char*));
      memcpy(size_array, 
             gpubuf->buf_size, 
               gpubuf->num_bufs * sizeof(int));
      free(gpubuf->buf);
      free(gpubuf->buf_size);
    }
    gpubuf->buf = buf_array;
    gpubuf->buf_size = size_array;
    gpubuf->buf_size[inv_num] = 0;
#ifdef ENFORCE_HOST_STAGING
    char **buf_host_array = (char**)malloc((inv_num + 1) * sizeof(char*));
    if(gpubuf->num_bufs != 0){
      memcpy(buf_host_array, 
             gpubuf->buf_host, 
               gpubuf->num_bufs * sizeof(char*));
      free(gpubuf->buf_host);
    }
    gpubuf->buf_host = buf_host_array;
#endif
    gpubuf->num_bufs++;
  }
  //check to see if enough space is allocated for inv_num
  int size_total = pos + size;
  if (gpubuf->buf_size[inv_num] < size_total)
  {
    char *newbuf;
    CUDA_ERRCHK(cudaMallocManaged((void**)&newbuf, size_total, cudaMemAttachGlobal));
    if(pos != 0){
      CUDA_ERRCHK(cudaMemcpy(newbuf,
        gpubuf->buf[inv_num], 
          pos, cudaMemcpyDeviceToDevice));
      CUDA_ERRCHK(cudaFree(gpubuf->buf[inv_num]));
    }
    gpubuf->buf[inv_num] = newbuf;
#ifdef ENFORCE_HOST_STAGING
    char *newbuf_host;
    newbuf_host = (char*)malloc(size_total);
    if(pos != 0){
      memcpy(newbuf_host, gpubuf->buf_host[inv_num], pos);
      free(gpubuf->buf_host[inv_num]);
    }
    gpubuf->buf_host[inv_num] = newbuf_host;
#endif
    gpubuf->buf_size[inv_num] = size_total;
  }
  return gpubuf->buf[inv_num] + pos;
}

static char* _amps_gpu_recvbuf(int inv_num, int size){
  if(inv_num >= amps_gpu_recvbuf.num_bufs){
    printf("ERROR at %s:%d: Not enough space is allocated for recvbufs\n", __FILE__, __LINE__);
    exit(1);
  }
#ifdef ENFORCE_HOST_STAGING
  //copy device buffer to host
  CUDA_ERRCHK(cudaMemcpy(amps_gpu_recvbuf.buf_host[inv_num],
    amps_gpu_recvbuf.buf[inv_num],
      size, cudaMemcpyDeviceToHost));
  return amps_gpu_recvbuf.buf_host[inv_num];
#else
  (void) size;
  return amps_gpu_recvbuf.buf[inv_num];
#endif
}

static char* _amps_gpu_recvbuf_realloc(int inv_num, int pos, int size){
  return _amps_gpubuf_realloc(&amps_gpu_recvbuf, inv_num, pos, size);
}

#ifdef ENFORCE_HOST_STAGING
static void _amps_gpu_recvbuf_sync_device(int inv_num, int pos, int size){
  //copy host buffer to device
  CUDA_ERRCHK(cudaMemcpy(amps_gpu_recvbuf.buf[inv_num] + pos,
    amps_gpu_recvbuf.buf_host[inv_num] + pos,
      size, cudaMemcpyHostToDevice));
}
#endif

static char* _amps_gpu_sendbuf(int inv_num, int size){
  if(inv_num >= amps_gpu_sendbuf.num_bufs){
    printf("ERROR at %s:%d: Not enough space is allocated for sendbufs\n", __FILE__, __LINE__);
    exit(1);
  }
#ifdef ENFORCE_HOST_STAGING
  //copy device buffer to host
  CUDA_ERRCHK(cudaMemcpy(amps_gpu_sendbuf.buf_host[inv_num],
    amps_gpu_sendbuf.buf[inv_num],
      size, cudaMemcpyDeviceToHost));
  return amps_gpu_sendbuf.buf_host[inv_num];
#else
  (void) size;
  return amps_gpu_sendbuf.buf[inv_num];
#endif
}

static char* _amps_gpu_sendbuf_realloc(int inv_num, int pos, int size){
  return _amps_gpubuf_realloc(&amps_gpu_sendbuf, inv_num, pos, size);
}

static int _amps_gpupack_invcheck_vector(int dim, int *len, int type)
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

static int _amps_gpupack_invcheck(amps_InvoiceEntry *ptr){
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

int amps_gpupacking(int action, amps_Invoice inv, int inv_num, char **buffer_out, int *size_out){
  amps_InvoiceEntry *ptr = inv->list;
  char *buffer;
  int pos = 0;

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
    if(cudaGetLastError() != cudaSuccess || attributes.type < 2){
      // printf("ERROR at %s:%d: The data location (not MPI staging buffer) is not accessible by the GPU(s).", __FILE__, __LINE__);
      return __LINE__;
    }

    int size = len_x * len_y * len_z * sizeof(double);
#ifdef ENFORCE_HOST_STAGING
    if(action == AMPS_UNPACK){
      //copy host staging buffer to device
      _amps_gpu_recvbuf_sync_device(inv_num, pos, size);
    }
#endif

    if(action == AMPS_PACK)
      buffer = _amps_gpu_sendbuf_realloc(inv_num, pos, size);
    else
      buffer = _amps_gpu_recvbuf_realloc(inv_num, pos, size);

    /* Get buffer location and its properties*/
    cudaPointerGetAttributes(&attributes, (void *)buffer);  

    /* Check that the staging buffer is accessible by the GPU */
    if(cudaGetLastError() != cudaSuccess || attributes.type < 2){
      // printf("ERROR at %s:%d: The MPI staging buffer location is not accessible by the GPU(s).", __FILE__, __LINE__);
      return __LINE__;
    }

    dim3 grid = dim3(((len_x - 1) + blocksize_x) / blocksize_x, ((len_y - 1) + blocksize_y) / blocksize_y, ((len_z - 1) + blocksize_z) / blocksize_z);
    dim3 block = dim3(blocksize_x, blocksize_y, blocksize_z);
    if(action == AMPS_PACK){
      PackingKernel<<<grid, block, 0, 0>>>((double*)buffer, (double*)data, len_x, len_y, len_z, stride_x, stride_y, stride_z);
      inv->flags |= AMPS_PACKED;
    }
    else if(action == AMPS_UNPACK){
      UnpackingKernel<<<grid, block, 0, 0>>>((double*)buffer, (double*)data, len_x, len_y, len_z, stride_x, stride_y, stride_z);
      inv->flags &= ~AMPS_PACKED;
    }
    CUDA_ERRCHK(cudaPeekAtLastError()); 
    CUDA_ERRCHK(cudaStreamSynchronize(0));

    pos += size;  
    ptr = ptr->next;
  }

  /* Check that the size is calculated right (somewhat costly) */
  // if(pos != amps_sizeof_invoice(amps_CommWorld, inv)){
    // printf("ERROR at %s:%d: The size does not match the invoice size.", __FILE__, __LINE__);
    // return __LINE__;
  // }

  //set out values here if everything went fine
  if(action == AMPS_PACK)
    *buffer_out = _amps_gpu_sendbuf(inv_num, pos);
  else
    *buffer_out = _amps_gpu_recvbuf(inv_num, pos);

  *size_out = pos;

  return 0;
} 
#endif // DISABLE_GPU_PACKING
}   
    