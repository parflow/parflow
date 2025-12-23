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

#include <Kokkos_Core.hpp>

extern "C"{

#include <stdlib.h>
#include <string.h>
#include "amps.h"

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
        printf("Node %d: Using GPUDirect (GPU-Aware MPI required)\n", amps_rank);
      ENFORCE_HOST_STAGING = 0;
    }
  }
}

/**
 * @brief Destroy all GPU streams and free allocations
 */
void amps_gpu_destroy_streams(){
}

/**
 * @brief Synchronizes GPU streams associated with the id
 *
 * Kokkos does not support multiple streams (only CUDA backend does).
 *
 * @param id The integer id associated with the synchronized streams [IN]
 */
void amps_gpu_sync_streams(int id){
  if(id == 0) Kokkos::fence();
}

/**
 * @brief Allocate device memory with Kokkos
 *
 * @param size The size of the allocation in bytes
 */
void* kokkosDeviceAlloc(size_t size){
#ifdef PARFLOW_HAVE_CUDA
  return Kokkos::kokkos_malloc<Kokkos::CudaSpace>(size);
#else
  return Kokkos::kokkos_malloc(size);
#endif
}

/**
 * @brief Free device memory with Kokkos
 *
 * @param ptr Freed pointer
 */
void kokkosDeviceFree(void *ptr){
#ifdef PARFLOW_HAVE_CUDA
  Kokkos::kokkos_free<Kokkos::CudaSpace>(ptr);
#else
  Kokkos::kokkos_free(ptr);
#endif
}

/**
 * @brief Allocate pinned host memory with Kokkos
 *
 * @param size The size of the allocation in bytes
 */
void* kokkosHostAlloc(size_t size){
#ifdef PARFLOW_HAVE_CUDA
  return Kokkos::kokkos_malloc<Kokkos::CudaHostPinnedSpace>(size);
#else
  return Kokkos::kokkos_malloc<Kokkos::HostSpace>(size);
#endif
}

/**
 * @brief Free pinned host memory with Kokkos
 *
 * @param ptr Freed pointer
 */
void kokkosHostFree(void *ptr){
#ifdef PARFLOW_HAVE_CUDA
  Kokkos::kokkos_free<Kokkos::CudaHostPinnedSpace>(ptr);
#else
  Kokkos::kokkos_free<Kokkos::HostSpace>(ptr);
#endif
}

/**
 * @brief Allocate Unified Memory with Kokkos
 *
 * @param size The size of the allocation in bytes
 */
void* kokkosUVMAlloc(size_t size){
  return Kokkos::kokkos_malloc(size);
}

/**
 * @brief Free Unified Memory with Kokkos
 *
 * @param ptr Freed pointer
 */
void kokkosUVMFree(void *ptr){
  Kokkos::kokkos_free(ptr);
}

/**
 * @brief Device-device memcopy
 *
 * @param dest Destination pointer
 * @param src Source pointer
 * @param size Bytes to be copied
 */
void kokkosMemCpyDeviceToDevice(char *dest, char *src, size_t size){
#ifdef PARFLOW_HAVE_CUDA
  Kokkos::View<char*, Kokkos::CudaSpace> dest_view(dest, size);
  Kokkos::View<char*, Kokkos::CudaSpace> src_view(src, size);
#else
  Kokkos::View<char*> dest_view(dest, size);
  Kokkos::View<char*> src_view(src, size);
#endif
  Kokkos::deep_copy(dest_view, src_view);
}

/**
 * @brief Device-host memcopy
 *
 * @param dest Destination pointer
 * @param src Source pointer
 * @param size Bytes to be copied
 */
void kokkosMemCpyDeviceToHost(char *dest, char *src, size_t size){
#ifdef PARFLOW_HAVE_CUDA
  Kokkos::View<char*, Kokkos::CudaHostPinnedSpace> dest_view(dest, size);
  Kokkos::View<char*, Kokkos::CudaSpace> src_view(src, size);
#else
  Kokkos::View<char*, Kokkos::HostSpace> dest_view(dest, size);
  Kokkos::View<char*> src_view(src, size);
#endif
  Kokkos::deep_copy(dest_view, src_view);
}
/**
 * @brief Host-device memcopy
 *
 * @param dest Destination pointer
 * @param src Source pointer
 * @param size Bytes to be copied
 */
void kokkosMemCpyHostToDevice(char *dest, char *src, size_t size){
#ifdef PARFLOW_HAVE_CUDA
  Kokkos::View<char*, Kokkos::CudaSpace> dest_view(dest, size);
  Kokkos::View<char*, Kokkos::CudaHostPinnedSpace> src_view(src, size);
#else
  Kokkos::View<char*> dest_view(dest, size);
  Kokkos::View<char*, Kokkos::HostSpace> src_view(src, size);
#endif
  Kokkos::deep_copy(dest_view, src_view);
}
/**
 * @brief Host-host memcopy
 *
 * @param dest Destination pointer
 * @param src Source pointer
 * @param size Bytes to be copied
 */
void kokkosMemCpyHostToHost(char *dest, char *src, size_t size){
#ifdef PARFLOW_HAVE_CUDA
  Kokkos::View<char*, Kokkos::CudaHostPinnedSpace> dest_view(dest, size);
  Kokkos::View<char*, Kokkos::CudaHostPinnedSpace> src_view(src, size);
#else
  Kokkos::View<char*, Kokkos::HostSpace> dest_view(dest, size);
  Kokkos::View<char*, Kokkos::HostSpace> src_view(src, size);
#endif
  Kokkos::deep_copy(dest_view, src_view);
}

/**
 * @brief UVM-UVM memcopy
 *
 * @param dest Destination pointer
 * @param src Source pointer
 * @param size Bytes to be copied
 */
void kokkosMemCpyUVMToUVM(char *dest, char *src, size_t size){
  Kokkos::View<char*> dest_view(dest, size);
  Kokkos::View<char*> src_view(src, size);
  Kokkos::deep_copy(dest_view, src_view);
}

/**
 * @brief Kokkos memset
 *
 * @param ptr Pointer for the data to be set to zero
 * @param size Bytes to be zeroed
 */
void kokkosMemSetAmps(char* ptr, size_t size) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  // Create an unmanaged view over the raw memory
  Kokkos::View<char*, Kokkos::MemoryUnmanaged> view(ptr, size);

  // Perform efficient parallel zero-initialization
  Kokkos::deep_copy(ExecSpace(), view, (char)0);

  // Synchronize only the current execution space (e.g., CUDA, HIP, etc.)
  ExecSpace().fence();
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
        kokkosDeviceFree(gpubuf->buf[i]);
        if(ENFORCE_HOST_STAGING)
          kokkosHostFree(gpubuf->buf_host[i]);
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
  if(Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
}

/**
 * @brief Initialize Kokkos if not initialized
 */
void amps_kokkos_initialization(){
  if(!Kokkos::is_initialized()) Kokkos::initialize();
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
    char *newbuf = (char*)kokkosDeviceAlloc(size_total);
    if(pos != 0){
      kokkosMemCpyDeviceToDevice(newbuf,
        gpubuf->buf[id], pos);
      kokkosDeviceFree(gpubuf->buf[id]);
    }
    gpubuf->buf[id] = newbuf;
    
    if(ENFORCE_HOST_STAGING){
      char *newbuf_host = (char*)kokkosHostAlloc(size_total);
      if(pos != 0){
        kokkosMemCpyHostToHost(newbuf_host,
          gpubuf->buf_host[id], pos);
        kokkosHostFree(gpubuf->buf_host[id]);
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

  amps_kokkos_initialization();

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
        len_z = ptr->ptr_len[2];
        stride_z = ptr->ptr_stride[2];
      case 2:
        len_y = ptr->ptr_len[1];
        stride_y = ptr->ptr_stride[1];
      case 1:
        len_x = ptr->ptr_len[0];
        stride_x = ptr->ptr_stride[0];
        break;
      default:
        printf("ERROR at %s:%d: Only dimensions 1 - 3 are supported\n", __FILE__, __LINE__);
        return __LINE__;
    }  
    
    /* Get data location and its properties */
    char *data = (char*)ptr->data;

#ifdef PARFLOW_HAVE_CUDA
    struct cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, (void *)data);  
  
    /* Check that the data location (not MPI staging buffer) is accessible by the GPU */
    if(cudaGetLastError() != cudaSuccess || attributes.type < 2){
      printf("ERROR at %s:%d: The data location (not MPI staging buffer) is not accessible by the GPU(s)\n", __FILE__, __LINE__);
      return __LINE__;
    }
#endif

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

#ifdef PARFLOW_HAVE_CUDA
    /* Get buffer location and its properties */
    cudaPointerGetAttributes(&attributes, (void *)buffer);  

    /* Check that the staging buffer is accessible by the GPU */
    if(cudaGetLastError() != cudaSuccess || attributes.type < 2){
      printf("ERROR at %s:%d: The MPI staging buffer location is not accessible by the GPU(s)\n", __FILE__, __LINE__);
      return __LINE__;
    }
#endif
    /* Run packing or unpacking kernel */
    using MDPolicyType_3D = typename Kokkos::MDRangePolicy<Kokkos::Rank<3> >;
    MDPolicyType_3D mdpolicy_3d({{0, 0, 0}}, {{len_x, len_y, len_z}});
    
    if(action == AMPS_PACK){
      Kokkos::parallel_for(mdpolicy_3d, KOKKOS_LAMBDA(int i, int j, int k)
      {
        double *ptr_buf = (double*)buffer;
        double *ptr_data = (double*)data;
        *(ptr_buf + k * len_y * len_x + j * len_x + i) = 
          *(ptr_data + k * (stride_z + (len_y - 1) * stride_y + len_y * (len_x - 1) * stride_x) + 
            j * (stride_y + (len_x - 1) * stride_x) + i * stride_x);                                          
      });
      if(ENFORCE_HOST_STAGING){
        /* Copy device buffer to host after packing */
        kokkosMemCpyDeviceToHost(amps_gpu_sendbuf.buf_host[inv_num] + pos,
                                    amps_gpu_sendbuf.buf[inv_num] + pos, size);
      }
      inv->flags |= AMPS_PACKED;
    }
    else if(action == AMPS_UNPACK){
      if(ENFORCE_HOST_STAGING){
        /* Copy host buffer to device before unpacking */
        kokkosMemCpyHostToDevice(amps_gpu_recvbuf.buf[inv_num] + pos,
                                    amps_gpu_recvbuf.buf_host[inv_num] + pos, size);
      }
      Kokkos::parallel_for(mdpolicy_3d, KOKKOS_LAMBDA(int i, int j, int k)
      {
        double *ptr_buf = (double*)buffer;
        double *ptr_data = (double*)data;
        *(ptr_data + k * (stride_z + (len_y - 1) * stride_y + len_y * (len_x - 1) * stride_x) + 
          j * (stride_y + (len_x - 1) * stride_x) + i * stride_x) = 
            *(ptr_buf + k * len_y * len_x + j * len_x + i);
      });
      inv->flags &= ~AMPS_PACKED;
    }
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
