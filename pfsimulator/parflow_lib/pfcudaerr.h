#ifndef PFCUDAERR_H
#define PFCUDAERR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <rmm/rmm_api.h>

/*--------------------------------------------------------------------------
 * CUDA error handling macros
 *--------------------------------------------------------------------------*/

#undef CUDA_ERR
#define CUDA_ERR(expr)                                                                 \
{                                                                                      \
  cudaError_t err = expr;                                                              \
  if (err != cudaSuccess) {                                                            \
    printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);  \
    exit(1);                                                                           \
  }                                                                                    \
}

#undef RMM_ERR
#define RMM_ERR(expr)                                                                  \
{                                                                                      \
  rmmError_t err = expr;                                                               \
  if (err != RMM_SUCCESS) {                                                            \
    printf("\n\n%s in %s at line %d\n", rmmGetErrorString(err), __FILE__, __LINE__);   \
    exit(1);                                                                           \
  }                                                                                    \
}

/*--------------------------------------------------------------------------
 * Define static unified memory allocation routines for CUDA
 *--------------------------------------------------------------------------*/

static inline void *talloc_cuda(size_t size)
{
  void *ptr = NULL;  
  
  RMM_ERR(rmmAlloc(&ptr,size,0,__FILE__,__LINE__));
  // CUDA_ERR(cudaMallocManaged((void**)&ptr, size, cudaMemAttachGlobal));
  // CUDA_ERR(cudaHostAlloc((void**)&ptr, size, cudaHostAllocMapped));  
  
  return ptr;
}

static inline void *ctalloc_cuda(size_t size)
{
  void *ptr = NULL;  
  RMM_ERR(rmmAlloc(&ptr,size,0,__FILE__,__LINE__));
  // CUDA_ERR(cudaMallocManaged((void**)&ptr, size, cudaMemAttachGlobal));
  // CUDA_ERR(cudaHostAlloc((void**)&ptr, size, cudaHostAllocMapped));
  
  // memset(ptr, 0, size);
  CUDA_ERR(cudaMemset(ptr, 0, size));  
  
  return ptr;
}
static inline void tfree_cuda(void *ptr)
{
  RMM_ERR(rmmFree(ptr,0,__FILE__,__LINE__));
  // CUDA_ERR(cudaFree(ptr));
  // CUDA_ERR(cudaFreeHost(ptr));
}

#endif // PFCUDAERR_H
