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
 * @brief Contains macro redefinitions for unified memory management.
 */

#ifndef PF_CUDAMALLOC_H
#define PF_CUDAMALLOC_H

#include <string.h>
#include "pf_cudamain.h"

/*--------------------------------------------------------------------------
 * Memory management macros for CUDA
 *--------------------------------------------------------------------------*/

#define talloc_cuda(type, count) \
  ((count) ? (type*)_talloc_cuda(sizeof(type) * (unsigned int)(count)) : NULL)

#define ctalloc_cuda(type, count) \
  ((count) ? (type*)_ctalloc_cuda(sizeof(type) * (unsigned int)(count)) : NULL)

#define tfree_cuda(ptr) if (ptr) _tfree_cuda(ptr); else {}

#define tmemcpy_cuda(dest, src, bytes) \
  CUDA_ERR(cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice))

#define MemPrefetchDeviceToHost_cuda(ptr, size, stream)              \
{                                                                    \
  CUDA_ERR(cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, stream));\
  CUDA_ERR(cudaStreamSynchronize(stream));                           \
}

#define MemPrefetchHostToDevice_cuda(ptr, size, stream)              \
{                                                                    \
  int device;                                                        \
  CUDA_ERR(cudaGetDevice(&device));                                  \
  CUDA_ERR(cudaMemPrefetchAsync(ptr, size, device, stream))          \
}

// Redefine amps.h definitions 
// @TODO: Find a better way, this is problematic because these are defined in amps.h
#undef amps_TAlloc
#define amps_TAlloc(type, count) \
  ((count) ? (type*)_talloc_cuda(sizeof(type) * (unsigned int)(count)) : NULL)

#undef amps_CTAlloc
#define amps_CTAlloc(type, count) \
  ((count) ? (type*)_ctalloc_cuda(sizeof(type) * (unsigned int)(count)) : NULL)

#undef amps_TFree
#define amps_TFree(ptr) if (ptr) _tfree_cuda(ptr); else {}


#endif // PF_CUDAMALLOC_H
