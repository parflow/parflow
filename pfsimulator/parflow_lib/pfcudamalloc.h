#ifndef PFCUDAMALLOC_H
#define PFCUDAMALLOC_H

/*--------------------------------------------------------------------------
 * Include CUDA error handling header
 *--------------------------------------------------------------------------*/
#include "pfcudaerr.h"

/*--------------------------------------------------------------------------
 * Define unified memory allocation routines for CUDA
 *--------------------------------------------------------------------------*/

static inline void *tallocCUDA(size_t size)
{
   void *ptr = NULL;

  //  RMM_ERR(rmmAlloc(&ptr,size,0,__FILE__,__LINE__));
  //  CUDA_ERR(cudaMallocManaged((void**)&ptr, size, cudaMemAttachGlobal));
   CUDA_ERR(cudaHostAlloc((void**)&ptr, size, cudaHostAllocMapped));

   return ptr;
}

static inline void *ctallocCUDA(size_t size)
{
   void *ptr = NULL;

  //  RMM_ERR(rmmAlloc(&ptr,size,0,__FILE__,__LINE__));
  //  CUDA_ERR(cudaMallocManaged((void**)&ptr, size, cudaMemAttachGlobal));
   CUDA_ERR(cudaHostAlloc((void**)&ptr, size, cudaHostAllocMapped));
   
   CUDA_ERR(cudaMemset(ptr, 0, size));

   return ptr;
}
static inline void tfreeCUDA(void *ptr)
{
  //  RMM_ERR(rmmFree(ptr,0,__FILE__,__LINE__));
  //  CUDA_ERR(cudaFree(ptr));
   CUDA_ERR(cudaFreeHost(ptr));
}

/*--------------------------------------------------------------------------
 * Refefine allocation macros for CUDA
 *--------------------------------------------------------------------------*/

// Redefine amps.h definitions
#undef amps_TAlloc
#define amps_TAlloc(type, count) \
  ((count) ? (type*)tallocCUDA(sizeof(type) * (unsigned int)(count)) : NULL)

#undef amps_CTAlloc
#define amps_CTAlloc(type, count) \
  ((count) ? (type*)ctallocCUDA(sizeof(type) * (unsigned int)(count)) : NULL)

#undef amps_TFree
#define amps_TFree(ptr) if (ptr) tfreeCUDA(ptr); else {}

// Redefine general.h definitions
#undef talloc
#define talloc(type, count) \
  ((count) ? (type*)tallocCUDA(sizeof(type) * (unsigned int)(count)) : NULL)

#undef ctalloc
#define ctalloc(type, count) \
  ((count) ? (type*)ctallocCUDA(sizeof(type) * (unsigned int)(count)) : NULL)

#undef tfree
#define tfree(ptr) if (ptr) tfreeCUDA(ptr); else {}

#endif // PFCUDAMALLOC_H
