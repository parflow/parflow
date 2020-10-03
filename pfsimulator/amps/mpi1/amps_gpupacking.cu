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

 /* TODO 
  * -make gpu bufs contiguous across invoices?
  */

extern "C"{

#include <string.h>
#include "amps.h"

// #define DISABLE_GPU_PACKING
// #define ENFORCE_HOST_STAGING

#ifdef DISABLE_GPU_PACKING
//Dummy definitions if no GPU packing
void amps_gpu_free_bufs(){}

void amps_gpu_destroy_streams(){}

void amps_gpu_sync_streams(int id){
  (void)id;
}

int amps_gpupacking(int action, amps_Invoice inv, int inv_num, char **buffer_out, int *size_out){
  (void)action;
  (void)inv;
  (void)inv_num;
  (void)buffer_out;
  (void)size_out;
  return 1;
}    
#else

amps_GpuBuffer amps_gpu_recvbuf = 
{
  .buf = NULL, 
  .buf_host = NULL,
  .buf_size = NULL,
  .num_bufs = 0
};

amps_GpuBuffer amps_gpu_sendbuf = 
{
  .buf = NULL, 
  .buf_host = NULL,
  .buf_size = NULL,
  .num_bufs = 0
};

amps_GpuStreams amps_gpu_streams = 
{
  .stream = NULL,
  .stream_id = NULL,
  .num_streams = 0,
  .reqs_since_sync = 0
};

/* Destroys all GPU streams */
void amps_gpu_destroy_streams(){
  for(int i = 0; i < amps_gpu_streams.num_streams; i++){
    CUDA_ERRCHK(cudaStreamDestroy(amps_gpu_streams.stream[i]));
  }
  free(amps_gpu_streams.stream);
  free(amps_gpu_streams.stream_id);
}

/* Marks a stream with the id and returns this stream (new streams is allocated if necessary) */
cudaStream_t amps_gpu_get_stream(int id){
  if(amps_gpu_streams.reqs_since_sync > amps_gpu_streams.num_streams - 1)
  {
    if(amps_gpu_streams.num_streams < AMPS_GPU_MAX_STREAMS)
    {
      int new_num_streams = amps_gpu_streams.reqs_since_sync + 1;
      if(amps_gpu_streams.reqs_since_sync >= AMPS_GPU_MAX_STREAMS)
        new_num_streams = AMPS_GPU_MAX_STREAMS;

      cudaStream_t *newstream = (cudaStream_t*)malloc(new_num_streams * sizeof(cudaStream_t));
      int *id_array = (int*)malloc(new_num_streams * sizeof(int));
      if(amps_gpu_streams.num_streams != 0){
        memcpy(newstream, 
          amps_gpu_streams.stream, 
            amps_gpu_streams.num_streams * sizeof(cudaStream_t));
        memcpy(id_array, 
          amps_gpu_streams.stream_id, 
            amps_gpu_streams.num_streams * sizeof(int));
        free(amps_gpu_streams.stream);
        free(amps_gpu_streams.stream_id);
      }
      amps_gpu_streams.stream = newstream;
      amps_gpu_streams.stream_id = id_array;
      for(int i = amps_gpu_streams.num_streams; i < new_num_streams; i++){
        CUDA_ERRCHK(cudaStreamCreate(&(amps_gpu_streams.stream[i])));
        amps_gpu_streams.stream_id[i] = INT_MAX; 
      }
      amps_gpu_streams.num_streams = new_num_streams;
    }
  }
  int stream_index = amps_gpu_streams.reqs_since_sync % AMPS_GPU_MAX_STREAMS;
  amps_gpu_streams.reqs_since_sync++;
  if(id < amps_gpu_streams.stream_id[stream_index])
    amps_gpu_streams.stream_id[stream_index] = id;
  return amps_gpu_streams.stream[stream_index];
}

/* Synchronizes GPU streams associated with the id */
void amps_gpu_sync_streams(int id){
  for(int i = 0; i < amps_gpu_streams.num_streams; i++)
  {
    if(amps_gpu_streams.stream_id[i] < id){
      printf("ERROR at %s:%d: The streams must be synced in ascending id order \n", __FILE__, __LINE__);
      exit(1);
    }
  }
  for(int i = 0; i < amps_gpu_streams.num_streams; i++)
  {
    if(amps_gpu_streams.stream_id[i] == id){
      CUDA_ERRCHK(cudaStreamSynchronize(amps_gpu_streams.stream[i])); 
      amps_gpu_streams.stream_id[i] = INT_MAX;
    }
  }
  amps_gpu_streams.reqs_since_sync = 0;
}
  
static void _amps_gpubuf_free(amps_GpuBuffer *gpubuf){
  if(gpubuf->num_bufs > 0){
    for(int i = 0; i < gpubuf->num_bufs; i++){
      if(gpubuf->buf_size[i] > 0){
        CUDA_ERRCHK(cudaFree(gpubuf->buf[i]));
#ifdef ENFORCE_HOST_STAGING
        CUDA_ERRCHK(cudaFreeHost(gpubuf->buf_host[i]));
#endif
      }
    }
    free(gpubuf->buf);
#ifdef ENFORCE_HOST_STAGING
    free(gpubuf->buf_host);
#endif
    free(gpubuf->buf_size);
  }
}

void amps_gpu_free_bufs(){
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
    CUDA_ERRCHK(cudaMalloc((void**)&newbuf, size_total));
    if(pos != 0){
      CUDA_ERRCHK(cudaMemcpy(newbuf,
        gpubuf->buf[inv_num], 
          pos, cudaMemcpyDeviceToDevice));
      CUDA_ERRCHK(cudaFree(gpubuf->buf[inv_num]));
    }
    gpubuf->buf[inv_num] = newbuf;
#ifdef ENFORCE_HOST_STAGING
    char *newbuf_host;
    CUDA_ERRCHK(cudaMallocHost((void**)&newbuf_host, size_total));
    if(pos != 0){
      CUDA_ERRCHK(cudaMemcpy(newbuf_host,
        gpubuf->buf_host[inv_num], 
          pos, cudaMemcpyHostToHost));
      CUDA_ERRCHK(cudaFreeHost(gpubuf->buf_host[inv_num]));
    }
    gpubuf->buf_host[inv_num] = newbuf_host;
#endif
    gpubuf->buf_size[inv_num] = size_total;
  }
  return gpubuf->buf[inv_num] + pos;
}

static char* _amps_gpu_recvbuf(int inv_num){
  if(inv_num >= amps_gpu_recvbuf.num_bufs){
    printf("ERROR at %s:%d: Not enough space is allocated for recvbufs\n", __FILE__, __LINE__);
    exit(1);
  }
#ifdef ENFORCE_HOST_STAGING
  return amps_gpu_recvbuf.buf_host[inv_num];
#else
  return amps_gpu_recvbuf.buf[inv_num];
#endif
}

static char* _amps_gpu_recvbuf_realloc(int inv_num, int pos, int size){
  return _amps_gpubuf_realloc(&amps_gpu_recvbuf, inv_num, pos, size);
}

static char* _amps_gpu_sendbuf(int inv_num){
  if(inv_num >= amps_gpu_sendbuf.num_bufs){
    printf("ERROR at %s:%d: Not enough space is allocated for sendbufs\n", __FILE__, __LINE__);
    exit(1);
  }
#ifdef ENFORCE_HOST_STAGING
  return amps_gpu_sendbuf.buf_host[inv_num];
#else
  return amps_gpu_sendbuf.buf[inv_num];
#endif
}

static char* _amps_gpu_sendbuf_realloc(int inv_num, int pos, int size){
  return _amps_gpubuf_realloc(&amps_gpu_sendbuf, inv_num, pos, size);
}

static int _amps_gpupack_check_inv_vector(int dim, int *len, int type)
{
  if (dim == 0){
    switch (type){
      case AMPS_INVOICE_DOUBLE_CTYPE:
        return 0; //This case is supported
      default:
        printf("ERROR at %s:%d: Only \"AMPS_INVOICE_DOUBLE_CTYPE\" is supported\n", __FILE__, __LINE__);
        return __LINE__;
    }
  }
  else{
    for (int i = 0; i < len[dim]; i++){
      if(int error = _amps_gpupack_check_inv_vector(dim - 1, len, type))
        return error;
    }
    return 0;
  }
}

/**
*
* The \Ref{amps_gpupacking} operates in 3 different modes:
*
* Mode 1: AMPS_GETRBUF / AMPS_GETSBUF
* -determines the size of the invoice message in bytes
* -allocates/reallocates the recv/send staging buffer if necessary
* -*buffer_out is set to point to the recv/send staging buffer
* -*size_out is set to the size of the invoice message in bytes
*
* Mode 2: AMPS_PACK (requires calling amps_gpu_sync_streams() before the packed data is accessed)
* -determines the size of the invoice message in bytes
* -allocates/reallocates the send staging buffer if necessary
* -packs data to the send staging buffer parallel using a GPU
* -*buffer_out is set to point to the send staging buffer
* -*size_out is set to the size of invoice message in bytes (size of the packed data)
*
* Mode 3: AMPS_UNPACK (requires calling amps_gpu_sync_streams() before the unpacked data is accessed)
* -determines the size of the invoice message in bytes
* -allocates/reallocates the recv staging buffer if necessary
* -unpacks data from the recv staging buffer parallel using a GPU
* -*buffer_out is set to point to the recv staging buffer
* -*size_out is set to the size of invoice message in bytes (size of the unpacked data)
*
* @param action either AMPS_GETRBUF, AMPS_GETSBUF, AMPS_PACK or AMPS_UNPACK [in]
* @param inv amps invoice [in]
* @param inv_num amps invoice order number [in]
* @param buffer_out pointer to the pointer of the staging buffer [out]
* @param size_out pointer to the invoice message size in bytes [out]
* @return error code (line number), 0 if succesful
*/
int amps_gpupacking(int action, amps_Invoice inv, int inv_num, char **buffer_out, int *size_out){
  
  char *buffer;
  int pos = 0;

  amps_InvoiceEntry *ptr = inv->list;
  while (ptr != NULL)
  {
    if (ptr->len_type != AMPS_INVOICE_POINTER){
      printf("ERROR at %s:%d: ptr->len_type must be a pointer\n", __FILE__, __LINE__);
      return __LINE__;
    }
  
    if (ptr->stride_type != AMPS_INVOICE_POINTER){
      printf("ERROR at %s:%d: ptr->stride_type must be a pointer\n", __FILE__, __LINE__);
      return __LINE__;
    }
  
    if (ptr->data_type == AMPS_INVOICE_POINTER){
      printf("ERROR at %s:%d: Overlayed invoices not supported\n", __FILE__, __LINE__);
      return __LINE__;
    }
  
    /* Make sure all needed invoice vector cases are supported */
    int dim = (ptr->dim_type == AMPS_INVOICE_POINTER) ? *(ptr->ptr_dim) : ptr->dim;
    int error = _amps_gpupack_check_inv_vector(dim - 1, ptr->ptr_len, ptr->type - AMPS_INVOICE_LAST_CTYPE);
    if(error) return error;
  
    /* Preparations for the kernel launch */
    int blocksize_x = 1;
    int blocksize_y = 1;
    int blocksize_z = 1;  
    int len_x = 1;
    int len_y = 1;
    int len_z = 1;  
    int stride_x = 0;
    int stride_y = 0;
    int stride_z = 0;
  
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
        printf("ERROR at %s:%d: Only dimensions 1 - 3 are supported\n", __FILE__, __LINE__);
        return __LINE__;
    }  
    
    /* Get data location and its properties */
    char *data = (char*)ptr->data;
    struct cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, (void *)data);  
  
    /* Check that the data location (not MPI staging buffer) is accessible by the GPU */
    if(cudaGetLastError() != cudaSuccess || attributes.type < 2){
      printf("ERROR at %s:%d: The data location (not MPI staging buffer) is not accessible by the GPU(s)\n", __FILE__, __LINE__);
      return __LINE__;
    }

    int size = len_x * len_y * len_z * sizeof(double);
    if((action == AMPS_GETSBUF) || (action == AMPS_PACK)){
      buffer = _amps_gpu_sendbuf_realloc(inv_num, pos, size);
    }
    else if((action == AMPS_GETRBUF) || (action == AMPS_UNPACK)){
      buffer = _amps_gpu_recvbuf_realloc(inv_num, pos, size);
    }
    else{
      printf("ERROR at %s:%d: Unknown action argument (val = %d)\n", __FILE__, __LINE__, action);
      return __LINE__;
    }

    /* Get buffer location and its properties */
    cudaPointerGetAttributes(&attributes, (void *)buffer);  

    /* Check that the staging buffer is accessible by the GPU */
    if(cudaGetLastError() != cudaSuccess || attributes.type < 2){
      printf("ERROR at %s:%d: The MPI staging buffer location is not accessible by the GPU(s)\n", __FILE__, __LINE__);
      return __LINE__;
    }

    /* Run packing or unpacking kernel */
    dim3 grid = dim3(((len_x - 1) + blocksize_x) / blocksize_x, 
                      ((len_y - 1) + blocksize_y) / blocksize_y, 
                       ((len_z - 1) + blocksize_z) / blocksize_z);
    dim3 block = dim3(blocksize_x, blocksize_y, blocksize_z);

    if(action == AMPS_PACK){
      cudaStream_t new_stream = amps_gpu_get_stream(inv_num);
      PackingKernel<<<grid, block, 0, new_stream>>>(
        (double*)buffer, (double*)data, len_x, len_y, len_z, stride_x, stride_y, stride_z);
#ifdef ENFORCE_HOST_STAGING
      //copy device buffer to host after packing
      CUDA_ERRCHK(cudaMemcpyAsync(amps_gpu_sendbuf.buf_host[inv_num] + pos,
                                    amps_gpu_sendbuf.buf[inv_num] + pos,
                                      size, cudaMemcpyDeviceToHost, new_stream));
#endif
      inv->flags |= AMPS_PACKED;
    }
    else if(action == AMPS_UNPACK){
      cudaStream_t new_stream = amps_gpu_get_stream(inv_num);
#ifdef ENFORCE_HOST_STAGING
      //copy host buffer to device before unpacking
      CUDA_ERRCHK(cudaMemcpyAsync(amps_gpu_recvbuf.buf[inv_num] + pos,
                                    amps_gpu_recvbuf.buf_host[inv_num] + pos,
                                      size, cudaMemcpyHostToDevice, new_stream));
#endif
      UnpackingKernel<<<grid, block, 0, new_stream>>>(
        (double*)buffer, (double*)data, len_x, len_y, len_z, stride_x, stride_y, stride_z);
      inv->flags &= ~AMPS_PACKED;
    }
    // CUDA_ERRCHK(cudaPeekAtLastError()); 
    // CUDA_ERRCHK(cudaStreamSynchronize(0));

    pos += size;  
    ptr = ptr->next;
  }

  /* Check that the size is calculated right */
  // if(pos != amps_sizeof_invoice(amps_CommWorld, inv)){
    // printf("ERROR at %s:%d: The size does not match the invoice size\n", __FILE__, __LINE__);
    // return __LINE__;
  // }

  //set the out values here if everything went fine
  if((action == AMPS_GETSBUF) || (action == AMPS_PACK)){
    *buffer_out = _amps_gpu_sendbuf(inv_num);
  }
  else if((action == AMPS_GETRBUF) || (action == AMPS_UNPACK)){
    *buffer_out = _amps_gpu_recvbuf(inv_num);
  }
  else{
    printf("ERROR at %s:%d: Unknown action argument (val = %d)\n", __FILE__, __LINE__, action);
    return __LINE__;
  }

  *size_out = pos;

  return 0;
} 
#endif // DISABLE_GPU_PACKING
}   
