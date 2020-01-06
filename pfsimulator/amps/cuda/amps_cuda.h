#ifndef AMPS_CUDA_H
#define AMPS_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdbool.h>
#include <rmm/rmm_api.h>

#define CUDA_ERR( err ) (gpuError( err, __FILE__, __LINE__ ))
static inline void gpuError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

#define CUDA_ERR_ARG( err, arg1, arg2, arg3 ) (gpuErrorArg( err, __FILE__, __LINE__, arg1, arg2, arg3 ))
static inline void gpuErrorArg(cudaError_t err, const char *file, int line, int arg1, int arg2, int arg3) {
	if (err != cudaSuccess) {
		printf("\n\n%s in %s at line %d; arg1: %d, arg2: %d, arg3: %d\n", cudaGetErrorString(err), file, line, arg1, arg2, arg3);
		exit(1);
	}
}

#define RMM_ERR( err ) (rmmError( err, __FILE__, __LINE__ ))
static inline void rmmError(rmmError_t err, const char *file, int line) {
	if (err != RMM_SUCCESS) {
		printf("\n\n%s in %s at line %d\n", rmmGetErrorString(err), file, line);
		exit(1);
	}
}

/*--------------------------------------------------------------------------
 * Define amps GPU kernels
 *--------------------------------------------------------------------------*/
#define BLOCKSIZE_MAX 1024

#ifdef __CUDACC__

extern "C++"{
template <typename T>
__global__ static void 
__launch_bounds__(BLOCKSIZE_MAX)
StridedCopyKernel(T * __restrict__ dest, const int stride_dest, 
                                  T * __restrict__ src, const int stride_src, const int len) 
{
  const int tid = ((blockIdx.x*blockDim.x)+threadIdx.x);
    
  if(tid < len)
  { 
    const int idx_dest = tid * stride_dest;
    const int idx_src = tid * stride_src;

    dest[idx_dest] = src[idx_src];
  }
}
template <typename T>
__global__ static void 
__launch_bounds__(BLOCKSIZE_MAX)
PackingKernel(T * __restrict__ ptr_buf, const T * __restrict__ ptr_data, 
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
template <typename T>
__global__ static void 
__launch_bounds__(BLOCKSIZE_MAX)
UnpackingKernel(const T * __restrict__ ptr_buf, T * __restrict__  ptr_data, 
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
#endif

/*--------------------------------------------------------------------------
 * Define unified memory allocation routines for CUDA
 *--------------------------------------------------------------------------*/

static inline void *tallocCUDA(size_t size)
{
   void *ptr = NULL;

   RMM_ERR(rmmAlloc(&ptr,size,0,__FILE__,__LINE__));
  //  CUDA_ERR(cudaMallocManaged((void**)&ptr, size, cudaMemAttachGlobal));
  //  CUDA_ERR(cudaHostAlloc((void**)&ptr, size, cudaHostAllocMapped));

   return ptr;
}

static inline void *ctallocCUDA(size_t size)
{
   void *ptr = NULL;

   RMM_ERR(rmmAlloc(&ptr,size,0,__FILE__,__LINE__));
  //  CUDA_ERR(cudaMallocManaged((void**)&ptr, size, cudaMemAttachGlobal));
  //  CUDA_ERR(cudaHostAlloc((void**)&ptr, size, cudaHostAllocMapped));
   
  //  memset(ptr, 0, size);
   CUDA_ERR(cudaMemset(ptr, 0, size));

   return ptr;
}
static inline void tfreeCUDA(void *ptr)
{
   RMM_ERR(rmmFree(ptr,0,__FILE__,__LINE__));
  //  CUDA_ERR(cudaFree(ptr));
  //  CUDA_ERR(cudaFreeHost(ptr));
}

/*--------------------------------------------------------------------------
 * Define allocation macros for CUDA
 *--------------------------------------------------------------------------*/

// Redefine general.h definitions
#undef talloc
#define talloc(type, count) \
  ((count) ? (type*)tallocCUDA(sizeof(type) * (unsigned int)(count)) : NULL)

#undef ctalloc
#define ctalloc(type, count) \
  ((count) ? (type*)ctallocCUDA(sizeof(type) * (unsigned int)(count)) : NULL)

#undef tfree
#define tfree(ptr) if (ptr) tfreeCUDA(ptr); else {}

/*--------------------------------------------------------------------------
 * Amps device struct for global amps variables
 *--------------------------------------------------------------------------*/
#define amps_device_max_streams 10
typedef struct amps_devicestruct {

  char *combuf_recv;
  char *combuf_send;
  long combuf_recv_size;
  long combuf_send_size;

  int streams_created;
  cudaStream_t stream[amps_device_max_streams];

} amps_Devicestruct;

extern amps_Devicestruct amps_device_globals;

#endif // AMPS_CUDA_H
