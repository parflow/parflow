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

#include <stdlib.h>
#include <string.h>
#include "amps.h"

/**
 * @brief The maximum number of GPU streams
 *
 * Here 1024 is an arbitrary number that should be larger than the sum 
 * of amps_InvoiceEntries in all recv or send invoices of the current 
 * communication package (ie, all packing/unpacking kernels should be 
 * allowed to run in a separate stream for best performance).
 * 
 * If the number is smaller than the required packing/unpacking kernel launches,
 * the kernels are queued in the available streams and run at least partially 
 * sequentially (useful for debugging purposes).
 */
#define AMPS_GPU_MAX_STREAMS 1024

/**
 * @brief The maximum block size for a GPU kernel
 *
 * The largest blocksize ParFlow is using, but also the largest blocksize 
 * supported by any currently available NVIDIA GPU architecture. This can 
 * also differ between different architectures. It is used for informing 
 * the compiler about how many registers should be available for the GPU 
 * kernel during the compilation. Another option is to use 
 * --maxrregcount 64 compiler flag, but NVIDIA recommends specifying 
 * this kernel-by-kernel basis by __launch_bounds__() identifier.
 */
#define BLOCKSIZE_MAX 1024

/* Disable GPU packing (useful for debugging) */
// #define DISABLE_GPU_PACKING

/**
 * @brief A global GPU staging buffer for amps_exchange
 * 
 * A separate instance is allocated for recv and send operations
 * which are shared between all amps packages. The space is allocated
 * according to the largest invoice size.
 */
 typedef struct _amps_GpuBuffer {
  /** 
  * A list of GPU pointers for the packed data for each recv/send invoice. 
  */
  char **buf;
  /** 
  * A list of CPU pointers for the packed data for each recv/send invoice. 
  * Only allocated if host staging is used.
  */
  char **buf_host;
  /** 
  * A list of buf sizes for each invoice (applies to *buf and *host_buf).
  */
  int *buf_size;
  /** 
  * The number of buffers allocated (applies to *buf and *host_buf).
  */
  int num_bufs;
} amps_GpuBuffer;

/**
 * @brief Information about the global GPU packing/unpacking streams
 */
typedef struct _amps_GpuStreams {
  /** 
  * A list of GPU streams.
  */
  cudaStream_t *stream;
  /** 
  * The stream id can be used to associate a stream with 
  * an id number (typically the invoice index).
  */
  int *stream_id;
  /** 
  * The number of streams created.
  */
  int num_streams;
  /** 
  * The number of requested streams since a synchronization (can be 
  * larger than num_streams).
  */
  int reqs_since_sync;
} amps_GpuStreams;

#ifdef DISABLE_GPU_PACKING
/* Dummy definitions if no GPU packing */
void amps_gpu_finalize(){}

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

/**
 * @brief Defines whether host staging is used
 *
 * 0: Use page-locked device staging buffer and pass GPU pointer
 * to MPI library
 *
 * 1: Use page-locked host staging buffer and pass CPU pointer
 * to MPI library (default)
 */
static int ENFORCE_HOST_STAGING = 1;

/**
 * @brief Check for GPUDirect environment variable
 * 
 * If PARFLOW_USE_GPUDIRECT=1 environment variable is found,
 * disable host staging.
 *
 */
void amps_gpu_check_env(){
  if(getenv("PARFLOW_USE_GPUDIRECT") != NULL){
    if(atoi(getenv("PARFLOW_USE_GPUDIRECT")) == 1){
      if(amps_rank == 0 && ENFORCE_HOST_STAGING != 0)
        printf("Node %d: Using GPUDirect (CUDA-Aware MPI required)\n", amps_rank);
      ENFORCE_HOST_STAGING = 0;
    }
  }
}

/**
 * @brief Destroy all GPU streams and free allocations
 */
void amps_gpu_destroy_streams(){
  for(int i = 0; i < amps_gpu_streams.num_streams; i++){
    CUDA_ERRCHK(cudaStreamDestroy(amps_gpu_streams.stream[i]));
  }
  free(amps_gpu_streams.stream);
  free(amps_gpu_streams.stream_id);
}

/**
 * @brief Get a new GPU stream
 *
 * Marks a stream with the id and returns this stream (new stream is allocated if necessary).
 *
 * @param id The integer id associated with the returned stream [IN]
 * @return A GPU stream
 */
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

/**
 * @brief Synchronizes GPU streams associated with the id
 *
 * The streams must be synchronized in ascending id order.
 *
 * @param id The integer id associated with the synchronized streams [IN]
 */
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
  
/**
 * @brief Free the staging buffers associated with gpubuf
 *
 * @param gpubuf A pointer to the amps_GpuBuffer [IN]
 */
static void _amps_gpubuf_free(amps_GpuBuffer *gpubuf){
  if(gpubuf->num_bufs > 0){
    for(int i = 0; i < gpubuf->num_bufs; i++){
      if(gpubuf->buf_size[i] > 0){
        CUDA_ERRCHK(cudaFree(gpubuf->buf[i]));
        if(ENFORCE_HOST_STAGING)
          CUDA_ERRCHK(cudaFreeHost(gpubuf->buf_host[i]));
      }
    }
    free(gpubuf->buf);
    free(gpubuf->buf_size);
    if(ENFORCE_HOST_STAGING)
      free(gpubuf->buf_host);
  }
}

/**
 * @brief Free the global recv and send staging buffers
 */
void amps_gpu_free_bufs(){
  _amps_gpubuf_free(&amps_gpu_recvbuf);
  _amps_gpubuf_free(&amps_gpu_sendbuf);
}

/**
 * @brief Finalize GPU resource usage
 */
 void amps_gpu_finalize(){
  amps_gpu_free_bufs();
  amps_gpu_destroy_streams();
}

/**
 * @brief Allocate/reallocate page-locked GPU/CPU staging buffers.
 *
 * Page-locked host memory allocation mirrors the GPU allocation
 * if host staging is enabled.
 *
 * @param gpubuf A pointer to the amps_GpuBuffer [IN]
 * @param id The id associated with the buffer (typically amps invoice index) [IN]
 * @param pos The position in bytes up to which the buffer is already filled [IN]
 * @param size The total required buffer size subtracted by pos [IN]
 * @return A pointer to the allocated/reallocated GPU buffer
 */
static char* _amps_gpubuf_realloc(amps_GpuBuffer *gpubuf, int id, int pos, int size){
  /* Check if arrays are big enough for id */
  if(id >= gpubuf->num_bufs){
    if(gpubuf->num_bufs == 0){
      amps_gpu_check_env();
    }
    if(id != gpubuf->num_bufs){
      printf("ERROR at %s:%d: Unexpected id\n", __FILE__, __LINE__);
      return NULL;
    }

    char **buf_array = (char**)malloc((id + 1) * sizeof(char*));
    int *size_array = (int*)malloc((id + 1) * sizeof(int));
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
    gpubuf->buf_size[id] = 0;

    if(ENFORCE_HOST_STAGING){
      char **buf_host_array = (char**)malloc((id + 1) * sizeof(char*));
      if(gpubuf->num_bufs != 0){
        memcpy(buf_host_array, 
               gpubuf->buf_host, 
                 gpubuf->num_bufs * sizeof(char*));
        free(gpubuf->buf_host);
      }
      gpubuf->buf_host = buf_host_array;
    }
    gpubuf->num_bufs++;
  }

  /* Check to see if enough space is allocated for id */
  int size_total = pos + size;
  if (gpubuf->buf_size[id] < size_total)
  {
    char *newbuf;
    CUDA_ERRCHK(cudaMalloc((void**)&newbuf, size_total));
    if(pos != 0){
      CUDA_ERRCHK(cudaMemcpy(newbuf,
        gpubuf->buf[id], 
          pos, cudaMemcpyDeviceToDevice));
      CUDA_ERRCHK(cudaFree(gpubuf->buf[id]));
    }
    gpubuf->buf[id] = newbuf;
    
    if(ENFORCE_HOST_STAGING){
      char *newbuf_host;
      CUDA_ERRCHK(cudaMallocHost((void**)&newbuf_host, size_total));
      if(pos != 0){
        CUDA_ERRCHK(cudaMemcpy(newbuf_host,
          gpubuf->buf_host[id], 
            pos, cudaMemcpyHostToHost));
        CUDA_ERRCHK(cudaFreeHost(gpubuf->buf_host[id]));
      }
      gpubuf->buf_host[id] = newbuf_host;
    }
    gpubuf->buf_size[id] = size_total;
  }
  return gpubuf->buf[id] + pos;
}

/**
 * @brief Get recv buffer associated with id to be passed for MPI
 *
 * @param id The id of the recv buffer [IN]
 * @return A pointer to the recv staging buffer
 */
static char* _amps_gpu_recvbuf(int id){
  if(id >= amps_gpu_recvbuf.num_bufs){
    printf("ERROR at %s:%d: Not enough space is allocated for recvbufs\n", __FILE__, __LINE__);
    exit(1);
  }
  if(ENFORCE_HOST_STAGING)
    return amps_gpu_recvbuf.buf_host[id];
  else
    return amps_gpu_recvbuf.buf[id];
}

/**
 * @brief Allocate/reallocate recv GPU/CPU staging buffers
 *
 * @param id The id associated with the buffer (typically amps invoice index) [IN]
 * @param pos The position in bytes up to which the buffer is already filled [IN]
 * @param size The total required buffer size subtracted by pos [IN]
 * @return A pointer to the allocated/reallocated GPU recv buffer
 */
static char* _amps_gpu_recvbuf_realloc(int id, int pos, int size){
  return _amps_gpubuf_realloc(&amps_gpu_recvbuf, id, pos, size);
}

/**
 * @brief Get send buffer associated with id to be passed for MPI
 *
 * @param id The id of the send buffer [IN]
 * @return A pointer to the send staging buffer
 */
static char* _amps_gpu_sendbuf(int id){
  if(id >= amps_gpu_sendbuf.num_bufs){
    printf("ERROR at %s:%d: Not enough space is allocated for sendbufs\n", __FILE__, __LINE__);
    exit(1);
  }
  if(ENFORCE_HOST_STAGING)
    return amps_gpu_sendbuf.buf_host[id];
  else
    return amps_gpu_sendbuf.buf[id];
}

/**
 * @brief Allocate/reallocate send GPU/CPU staging buffers
 *
 * @param id The id associated with the buffer (typically amps invoice index) [IN]
 * @param pos The position in bytes up to which the buffer is already filled [IN]
 * @param size The total required buffer size subtracted by pos [IN]
 * @return A pointer to the allocated/reallocated GPU send buffer
 */
static char* _amps_gpu_sendbuf_realloc(int id, int pos, int size){
  return _amps_gpubuf_realloc(&amps_gpu_sendbuf, id, pos, size);
}

 extern "C++"{
/**
 * @brief GPU packing kernel
 *
 * @param ptr_buf A pointer to the packing destination (GPU staging buffer) [IN/OUT]
 * @param ptr_data A pointer to the source to be packed (Unified Memory) [IN]
 * @param len_x The data length along x dimension [IN]
 * @param len_y The data length along y dimension [IN]
 * @param len_z The data length along z dimension [IN]
 * @param stride_x The stride along x dimension [IN]
 * @param stride_y The stride along y dimension [IN]
 * @param stride_z The stride along z dimension [IN]
 */
 template <typename T>
 __global__ static void 
 __launch_bounds__(BLOCKSIZE_MAX)
 _amps_packing_kernel(T * __restrict__ ptr_buf, const T * __restrict__ ptr_data, 
     const int len_x, const int len_y, const int len_z, const int stride_x, const int stride_y, const int stride_z) 
 {
   const int k = ((blockIdx.z*blockDim.z)+threadIdx.z);   
   if(k < len_z)
   {
     const int j = ((blockIdx.y*blockDim.y)+threadIdx.y);   
     if(j < len_y)
     {
       const int i = ((blockIdx.x*blockDim.x)+threadIdx.x);   
       if(i < len_x)
       {
         *(ptr_buf + k * len_y * len_x + j * len_x + i) = 
           *(ptr_data + k * (stride_z + (len_y - 1) * stride_y + len_y * (len_x - 1) * stride_x) + 
             j * (stride_y + (len_x - 1) * stride_x) + i * stride_x);
       }
     }
   }
 }
 
 /**
 * @brief GPU unpacking kernel
 *
 * @param ptr_buf A pointer to the source to be unpacked (GPU staging buffer) [IN]
 * @param ptr_data A pointer to the unpacking destination (Unified Memory) [IN/OUT]
 * @param len_x The data length along x dimension [IN]
 * @param len_y The data length along y dimension [IN]
 * @param len_z The data length along z dimension [IN]
 * @param stride_x The stride along x dimension [IN]
 * @param stride_y The stride along y dimension [IN]
 * @param stride_z The stride along z dimension [IN]
 */
 template <typename T>
 __global__ static void 
 __launch_bounds__(BLOCKSIZE_MAX)
 _amps_unpacking_kernel(const T * __restrict__ ptr_buf, T * __restrict__  ptr_data, 
     const int len_x, const int len_y, const int len_z, const int stride_x, const int stride_y, const int stride_z) 
 {
   const int k = ((blockIdx.z*blockDim.z)+threadIdx.z);   
   if(k < len_z)
   {
     const int j = ((blockIdx.y*blockDim.y)+threadIdx.y);   
     if(j < len_y)
     {
       const int i = ((blockIdx.x*blockDim.x)+threadIdx.x);   
       if(i < len_x)
       {
         *(ptr_data + k * (stride_z + (len_y - 1) * stride_y + len_y * (len_x - 1) * stride_x) + 
           j * (stride_y + (len_x - 1) * stride_x) + i * stride_x) = 
             *(ptr_buf + k * len_y * len_x + j * len_x + i);
       }
     }
   }
 }
 }

/**
* @brief The main amps GPU packing driver function
*
* Operates in 3 different modes:
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
* @param action either AMPS_GETRBUF, AMPS_GETSBUF, AMPS_PACK or AMPS_UNPACK [IN]
* @param inv amps invoice [IN]
* @param inv_num amps invoice order number [IN]
* @param buffer_out pointer to the pointer of the staging buffer [OUT]
* @param size_out pointer to the invoice message size in bytes [OUT]
* @return error code (line number), 0 if successful
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
  
    /* Make sure amps invoice vector type is supported */
    if(ptr->type - AMPS_INVOICE_LAST_CTYPE != AMPS_INVOICE_DOUBLE_CTYPE){
      printf("ERROR at %s:%d: Only \"AMPS_INVOICE_DOUBLE_CTYPE\" is supported\n", __FILE__, __LINE__);
      return __LINE__;
    }
  
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
      _amps_packing_kernel<<<grid, block, 0, new_stream>>>(
        (double*)buffer, (double*)data, len_x, len_y, len_z, stride_x, stride_y, stride_z);
      if(ENFORCE_HOST_STAGING){
        /* Copy device buffer to host after packing */
        CUDA_ERRCHK(cudaMemcpyAsync(amps_gpu_sendbuf.buf_host[inv_num] + pos,
                                      amps_gpu_sendbuf.buf[inv_num] + pos,
                                        size, cudaMemcpyDeviceToHost, new_stream));
      }
      inv->flags |= AMPS_PACKED;
    }
    else if(action == AMPS_UNPACK){
      cudaStream_t new_stream = amps_gpu_get_stream(inv_num);
      if(ENFORCE_HOST_STAGING){
        /* Copy host buffer to device before unpacking */
        CUDA_ERRCHK(cudaMemcpyAsync(amps_gpu_recvbuf.buf[inv_num] + pos,
                                      amps_gpu_recvbuf.buf_host[inv_num] + pos,
                                        size, cudaMemcpyHostToDevice, new_stream));
      }
      _amps_unpacking_kernel<<<grid, block, 0, new_stream>>>(
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

  /* Set the out values here if everything went fine */
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
