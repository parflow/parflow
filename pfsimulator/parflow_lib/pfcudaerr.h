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
 * @brief Contains macros and functions for CUDA error handling and unified memory allocation.
 */

#ifndef PFCUDAERR_H
#define PFCUDAERR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>

/*--------------------------------------------------------------------------
 * CUDA error handling macros
 *--------------------------------------------------------------------------*/

#undef CUDA_ERR
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

#ifdef HAVE_RMM
#include <rmm/rmm_api.h>

#undef RMM_ERR
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
static inline void *talloc_cuda(size_t size)
{
  void *ptr = NULL;  
  
#ifdef HAVE_RMM
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
static inline void *ctalloc_cuda(size_t size)
{
  void *ptr = NULL;  

#ifdef HAVE_RMM
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
 * @brief Frees unified memory allocated with \ref talloc_cuda or \ref ctalloc_cuda.
 * 
 * @note Should not be called directly.
 *
 * @param ptr a void pointer to the allocated dataspace [IN]
 */
static inline void tfree_cuda(void *ptr)
{
#ifdef HAVE_RMM
  RMM_ERR(rmmFree(ptr,0,__FILE__,__LINE__));
#else
  CUDA_ERR(cudaFree(ptr));
  // CUDA_ERR(cudaFreeHost(ptr));
#endif
}

#endif // PFCUDAERR_H
