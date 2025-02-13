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
 * @brief Contains general device-related macros and functions.
 */

#ifndef PF_DEVICES_H
#define PF_DEVICES_H

#include <string.h>

#ifdef PARFLOW_HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>

#ifdef PARFLOW_HAVE_RMM
#include "amps_rmm_wrapper.h"
#endif

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
#define CUDA_ERR(expr)                                                                         \
        {                                                                                      \
          cudaError_t err = expr;                                                              \
          if (err != cudaSuccess) {                                                            \
            printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);  \
            exit(1);                                                                           \
          }                                                                                    \
        }

/*--------------------------------------------------------------------------
 * CUDA profiling macros
 *--------------------------------------------------------------------------*/

/** Record an NVTX range for NSYS if accelerator present. */
#include "nvtx3/nvToolsExt.h"
#define PUSH_NVTX_cuda(name, cid)                                                                   \
        {                                                                                           \
          const uint32_t colors_nvtx[] =                                                            \
          { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };   \
          const int num_colors_nvtx = sizeof(colors_nvtx) / sizeof(uint32_t);                       \
          int color_id_nvtx = cid;                                                                  \
          color_id_nvtx = color_id_nvtx % num_colors_nvtx;                                          \
          nvtxEventAttributes_t eventAttrib = { 0 };                                                \
          eventAttrib.version = NVTX_VERSION;                                                       \
          eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                         \
          eventAttrib.colorType = NVTX_COLOR_ARGB;                                                  \
          eventAttrib.color = colors_nvtx[color_id_nvtx];                                           \
          eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                                        \
          eventAttrib.message.ascii = name;                                                         \
          nvtxRangePushEx(&eventAttrib);                                                            \
        }
/** Stop recording an NVTX range for NSYS if accelerator present. */
#define POP_NVTX_cuda nvtxRangePop();

#endif // PARFLOW_HAVE_CUDA

#ifdef PARFLOW_HAVE_UMPIRE
#include "amps_umpire_wrapper.h"
#endif

/*--------------------------------------------------------------------------
 * Define static unified memory allocation routines for device
 *--------------------------------------------------------------------------*/

/**
 * @brief Kokkos C wrapper declaration for memory allocation.
 */
void* kokkosAlloc(size_t size);

/**
 * @brief Kokkos C wrapper declaration for memory deallocation.
 */
void kokkosFree(void *ptr);

/**
 * @brief Kokkos C wrapper declaration for memory copy.
 */
void kokkosMemCpy(char *dest, char *src, size_t size);

/**
 * @brief Kokkos C wrapper declaration for memset.
 */
void kokkosMemSet(char *ptr, size_t size);

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

static inline void *_talloc_device(size_t size)
{
  void *ptr = NULL;

#ifdef PARFLOW_HAVE_RMM
  ptr = amps_rmmAlloc(size);
#elif defined(PARFLOW_HAVE_UMPIRE)
  ptr = amps_umpireAlloc(size);
#elif defined(PARFLOW_HAVE_KOKKOS)
  ptr = kokkosAlloc(size);
#elif defined(PARFLOW_HAVE_CUDA)
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
static inline void *_ctalloc_device(size_t size)
{
  void *ptr = NULL;

#ifdef PARFLOW_HAVE_RMM
  ptr = amps_rmmAlloc(size);
#elif defined(PARFLOW_HAVE_UMPIRE)
  ptr = amps_umpireAlloc(size);
#elif defined(PARFLOW_HAVE_KOKKOS)
  ptr = kokkosAlloc(size);
#elif defined(PARFLOW_HAVE_CUDA)
  CUDA_ERR(cudaMallocManaged((void**)&ptr, size, cudaMemAttachGlobal));
  // CUDA_ERR(cudaHostAlloc((void**)&ptr, size, cudaHostAllocMapped));
#endif

#if defined(PARFLOW_HAVE_CUDA)
  CUDA_ERR(cudaMemset(ptr, 0, size));
#else
  // memset(ptr, 0, size);
  kokkosMemSet((char*)ptr, size);
#endif

  return ptr;
}

/**
 * @brief Frees unified memory allocated with \ref _talloc_device or \ref _ctalloc_device.
 *
 * @note Should not be called directly.
 *
 * @param ptr a void pointer to the allocated dataspace [IN]
 */
static inline void _tfree_device(void *ptr)
{
#ifdef PARFLOW_HAVE_RMM
  amps_rmmFree(ptr);
#elif defined(PARFLOW_HAVE_UMPIRE)
  amps_umpireFree(ptr);
#elif defined(PARFLOW_HAVE_KOKKOS)
  kokkosFree(ptr);
#elif defined(PARFLOW_HAVE_CUDA)
  CUDA_ERR(cudaFree(ptr));
  // CUDA_ERR(cudaFreeHost(ptr));
#endif
}

#endif // PF_DEVICES_H
