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

/** @file
 * @brief Contains general CUDA related macros and functions.
 */

#ifndef PF_CUDAMAIN_H
#define PF_CUDAMAIN_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>

/*--------------------------------------------------------------------------
 * CUDA error handling macros
 *--------------------------------------------------------------------------*/

/**
 * @brief CUDA error handling.
 * 
 * If error detected, print error message and exit.
 *
 * @param expr CUDA error (of type cudaError_t) [IN]
 */
#define CUDA_ERR(expr)                                                                 \
{                                                                                      \
  cudaError_t err = expr;                                                              \
  if (err != cudaSuccess) {                                                            \
    printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);  \
    exit(1);                                                                           \
  }                                                                                    \
}

#ifdef PARFLOW_HAVE_RMM
#include <rmm/rmm_api.h>
/**
 * @brief RMM error handling.
 * 
 * If error detected, print error message and exit.
 *
 * @param expr RMM error (of type rmmError_t) [IN]
 */
#define RMM_ERR(expr)                                                                  \
{                                                                                      \
  rmmError_t err = expr;                                                               \
  if (err != RMM_SUCCESS) {                                                            \
    printf("\n\n%s in %s at line %d\n", rmmGetErrorString(err), __FILE__, __LINE__);   \
    exit(1);                                                                           \
  }                                                                                    \
}
#endif

/*--------------------------------------------------------------------------
 * CUDA profiling macros
 *--------------------------------------------------------------------------*/

/** Record an NVTX range for NSYS if accelerator present. */
#include "nvToolsExt.h"
#define PUSH_NVTX_cuda(name,cid)                                                          \
{                                                                                         \
  const uint32_t colors_nvtx[] =                                                          \
    {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff}; \
  const int num_colors_nvtx = sizeof(colors_nvtx)/sizeof(uint32_t);                       \
  int color_id_nvtx = cid;                                                                \
  color_id_nvtx = color_id_nvtx%num_colors_nvtx;                                          \
  nvtxEventAttributes_t eventAttrib = {0};                                                \
  eventAttrib.version = NVTX_VERSION;                                                     \
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                       \
  eventAttrib.colorType = NVTX_COLOR_ARGB;                                                \
  eventAttrib.color = colors_nvtx[color_id_nvtx];                                         \
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                                      \
  eventAttrib.message.ascii = name;                                                       \
  nvtxRangePushEx(&eventAttrib);                                                          \
}
/** Stop recording an NVTX range for NSYS if accelerator present. */
#define POP_NVTX_cuda nvtxRangePop();

/*--------------------------------------------------------------------------
 * Define static unified memory allocation routines for CUDA
 *--------------------------------------------------------------------------*/

/**
 * @brief Allocates unified memory.
 * 
 * If RMM library is available, pool allocation is used for better performance.
 * 
 * @note Should not be called directly.
 *
 * @param size bytes to be allocated [IN]
 * @return a void pointer to the allocated dataspace
 */
static inline void *_talloc_cuda(size_t size)
{
  void *ptr = NULL;  
  
#ifdef PARFLOW_HAVE_RMM
  RMM_ERR(rmmAlloc(&ptr,size,0,__FILE__,__LINE__));
#else
  CUDA_ERR(cudaMallocManaged((void**)&ptr, size, cudaMemAttachGlobal));
  // CUDA_ERR(cudaHostAlloc((void**)&ptr, size, cudaHostAllocMapped));  
#endif
  
  return ptr;
}

/**
 * @brief Allocates unified memory initialized to 0.
 * 
 * If RMM library is available, pool allocation is used for better performance.
 * 
 * @note Should not be called directly.
 *
 * @param size bytes to be allocated [IN]
 * @return a void pointer to the allocated dataspace
 */
static inline void *_ctalloc_cuda(size_t size)
{
  void *ptr = NULL;  

#ifdef PARFLOW_HAVE_RMM
  RMM_ERR(rmmAlloc(&ptr,size,0,__FILE__,__LINE__));
#else
  CUDA_ERR(cudaMallocManaged((void**)&ptr, size, cudaMemAttachGlobal));
  // CUDA_ERR(cudaHostAlloc((void**)&ptr, size, cudaHostAllocMapped));
#endif  
  // memset(ptr, 0, size);
  CUDA_ERR(cudaMemset(ptr, 0, size));  
  
  return ptr;
}

/**
 * @brief Frees unified memory allocated with \ref _talloc_cuda or \ref _ctalloc_cuda.
 * 
 * @note Should not be called directly.
 *
 * @param ptr a void pointer to the allocated dataspace [IN]
 */
static inline void _tfree_cuda(void *ptr)
{
#ifdef PARFLOW_HAVE_RMM
  RMM_ERR(rmmFree(ptr,0,__FILE__,__LINE__));
#else
  CUDA_ERR(cudaFree(ptr));
  // CUDA_ERR(cudaFreeHost(ptr));
#endif
}

#endif // PF_CUDAMAIN_H
