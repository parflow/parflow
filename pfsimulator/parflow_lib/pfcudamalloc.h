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
