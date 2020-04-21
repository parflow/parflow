#ifndef PFCUDAMALLOC_H
#define PFCUDAMALLOC_H

#include <string.h>
#include "pfcudaerr.h"

/*--------------------------------------------------------------------------
 * Redefine macros for CUDA
 *--------------------------------------------------------------------------*/

// Redefine amps.h definitions
#undef amps_TAlloc
#define amps_TAlloc(type, count) \
  ((count) ? (type*)talloc_cuda(sizeof(type) * (unsigned int)(count)) : NULL)

#undef amps_CTAlloc
#define amps_CTAlloc(type, count) \
  ((count) ? (type*)ctalloc_cuda(sizeof(type) * (unsigned int)(count)) : NULL)

#undef amps_TFree
#define amps_TFree(ptr) if (ptr) tfree_cuda(ptr); else {}

// Redefine general.h definitions
#undef talloc
#define talloc(type, count) \
  ((count) ? (type*)talloc_cuda(sizeof(type) * (unsigned int)(count)) : NULL)

#undef ctalloc
#define ctalloc(type, count) \
  ((count) ? (type*)ctalloc_cuda(sizeof(type) * (unsigned int)(count)) : NULL)

#undef tfree
#define tfree(ptr) if (ptr) tfree_cuda(ptr); else {}

#undef MemPrefetchDeviceToHost
#define MemPrefetchDeviceToHost(ptr, size, stream)                   \
{                                                                    \
  CUDA_ERR(cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, stream));\
  CUDA_ERR(cudaStreamSynchronize(stream));                           \
}

#undef MemPrefetchHostToDevice
#define MemPrefetchHostToDevice(ptr, size, stream)                   \
{                                                                    \
  int device;                                                        \
  CUDA_ERR(cudaGetDevice(&device));                                  \
  CUDA_ERR(cudaMemPrefetchAsync(ptr, size, device, stream))          \
}

#endif // PFCUDAMALLOC_H