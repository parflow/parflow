/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
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
**********************************************************************EHEADER*/

/** @file
 * @brief General macro definitions.
 */

#ifndef _GENERAL_HEADER
#define _GENERAL_HEADER

#include <float.h>

/*--------------------------------------------------------------------------
 * Error macros
 *--------------------------------------------------------------------------*/

#define PARFLOW_ERROR(X)                      \
        do {                                  \
          _amps_Abort(X, __FILE__, __LINE__); \
        } while (0)

/*--------------------------------------------------------------------------
 * Define memory allocation routines
 *--------------------------------------------------------------------------*/

/**
 * @brief Allocates memory that is passed to amps library.
 *
 * When using an accelerator device, allocates unified memory with redefined macro.
 *
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param type the C type name
 * @param count number of items of type to allocate
 * @return pointer to the allocated dataspace
 */
#define talloc_amps_default(type, count) amps_TAlloc(type, count)

/**
 * @brief Allocates memory initialized to 0 that is passed to amps library.
 *
 * When using an accelerator device, allocates unified memory with redefined macro.
 *
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param type the C type name
 * @param count number of items of type to allocate
 * @return pointer to the allocated dataspace
 */
#define ctalloc_amps_default(type, count) amps_CTAlloc(type, count)

/**
 * Deallocates memory for objects that were allocated by \ref talloc_amps_default or \ref ctalloc_amps_default.
 *
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param ptr pointer to dataspace to free
 * @return error code
 */
#define tfree_amps_default(ptr) amps_TFree(ptr)

#ifdef PF_MEMORY_ALLOC_CHECK

/*--------------------------------------
 * Check memory allocation
 *--------------------------------------*/

#define talloc_default(type, count) \
        (type*)malloc_chk((unsigned int)((count) * sizeof(type)), __FILE__, __LINE__)

#define ctalloc_default(type, count)                                           \
        (type*)calloc_chk((unsigned int)(count), (unsigned int)sizeof(type),   \
                          __FILE__, __LINE__)

#else

/*--------------------------------------
 * Do not check memory allocation
 *--------------------------------------*/

/**
 * @brief Allocates memory.
 *
 * When using an accelerator device, allocates unified memory with redefined macro.
 *
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param type the C type name
 * @param count number of items of type to allocate
 * @return pointer to the allocated dataspace
 */
#define talloc_default(type, count) \
        (((count) > 0) ? (type*)malloc(sizeof(type) * (unsigned int)(count)) : NULL)

/**
 * @brief Allocates memory initialized to 0.
 *
 * When using an accelerator device, allocates unified memory with redefined macro.
 *
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param type the C type name
 * @param count number of items of type to allocate
 * @return pointer to the allocated dataspace
 */
#define ctalloc_default(type, count) \
        (((count) > 0) ? (type*)calloc((unsigned int)(count), (unsigned int)sizeof(type)) : NULL)

#endif

/**
 * Deallocates memory for objects that were allocated by \ref talloc_default or \ref ctalloc_default.
 *
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param ptr pointer to dataspace to free
 * @return error code
 */
#define tfree_default(ptr) if (ptr) free(ptr); else {}

/**
 * Copies data from dest to src
 *
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param dest destination address
 * @param src source address
 * @param bytes amount of data to be copied
 * @return error code
 */
#define tmemcpy_default(dest, src, bytes) memcpy(dest, src, bytes)


/*--------------------------------------------------------------------------
 * TempData macros
 *--------------------------------------------------------------------------*/

#define NewTempData(temp_data_sz) amps_CTAlloc(double, (temp_data_sz))

#define FreeTempData(temp_data) amps_TFree(temp_data)


/*--------------------------------------------------------------------------
 * Define various functions
 *--------------------------------------------------------------------------*/

#define pfmax(a, b)  (((a) < (b)) ? (b) : (a))

#define pfmin(a, b)  (((a) < (b)) ? (a) : (b))

#define pfround(x)  (((x) < 0.0) ? ((int)(x - 0.5)) : ((int)(x + 0.5)))

/**
 * Thread-safe addition assignment. This macro
 * can be called anywhere in any compute kernel.
 *
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param a original value [IN], sum result [OUT]
 * @param b value to be added [IN]
 */
#define PlusEquals_default(a, b) (a += b)

/**
 * Thread-safe reduction to find maximum value for reduction loops.
 * Each thread must call this macro as the last statement inside the reduction loop body.
 *
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param a value 1 for comparison [IN], max value [OUT]
 * @param b value 2 for comparison [IN]
 */
#define ReduceMax_default(a, b) if (a < b) { a = b; } else {};

/**
 * Thread-safe reduction to find maximum value for reduction loops.
 * Each thread must call this macro as the last statement inside the reduction loop body.
 *
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param a value 1 for comparison [IN], min value [OUT]
 * @param b value 2 for comparison [IN]
 */
#define ReduceMin_default(a, b) if (a > b) { a = b; } else {};

/**
 * Thread-safe addition assignment for reduction loops.
 * Each thread must call this macro as the last statement inside the reduction loop body.
 *
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param a original value [IN], sum result [OUT]
 * @param b value to be added [IN]
 */
#define ReduceSum_default(a, b) (a += b)

/* return 2^e, where e >= 0 is an integer */
#define Pow2(e)   (((unsigned int)0x01) << (e))

/*--------------------------------------------------------------------------
 * Define various flags
 *--------------------------------------------------------------------------*/

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#define ON  1
#define OFF 0

#define DO   1
#define UNDO 0

#define XDIRECTION 0
#define YDIRECTION 1
#define ZDIRECTION 2

#define CALCFCN 0
#define CALCDER 1

#define GetInt(key) IDB_GetInt(amps_ThreadLocal(input_database), (key))
#define GetDouble(key) IDB_GetDouble(amps_ThreadLocal(input_database), (key))
#define GetString(key) IDB_GetString(amps_ThreadLocal(input_database), (key))

#define GetIntDefault(key, default) IDB_GetIntDefault(amps_ThreadLocal(input_database), (key), (default))
#define GetDoubleDefault(key, default) IDB_GetDoubleDefault(amps_ThreadLocal(input_database), (key), (default))
#define GetStringDefault(key, default) IDB_GetStringDefault(amps_ThreadLocal(input_database), (key), (default))

#define TIME_EPSILON (FLT_EPSILON * 10)

/*--------------------------------------------------------------------------
 * Define CUDA macros to do nothing if no GPU acceleration
 *--------------------------------------------------------------------------*/

#ifndef __host__
/** Defines an object accessible from host. @note Does nothing if not supported by the compiler. */
  #define __host__
#endif
#ifndef __device__
/** Defines an object accessible from device. @note Does nothing if not supported by the compiler. */
  #define __device__
#endif
#ifndef __managed__
/** Defines a variable that is automatically migrated between host/device. @note Does nothing if not supported by the compiler. */
  #define __managed__
#endif
#ifndef __restrict__
/** Defines a restricted pointer. @note Does nothing if not supported by the compiler. */
  #define __restrict__
#endif

/**
 * Used to prefetch data from host to device for better performance.
 *
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param ptr pointer to data [IN]
 * @param size bytes to be prefetched [IN]
 * @param stream the device stream (0, if no streams are launched) [IN]
 * @return CUDA error code
 */
#define MemPrefetchDeviceToHost_default(ptr, size, stream)

/**
 * Used to prefetch data from host to device for better performance.
 *
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param ptr pointer to data [IN]
 * @param size bytes to be prefetched [IN]
 * @param stream the device stream (0, if no streams are launched) [IN]
 * @return CUDA error code
 */
#define MemPrefetchHostToDevice_default(ptr, size, stream)

/**
 * Explicit sync between host and device default stream if accelerator present.
 * Can be called anywhere.
 *
 * @note Multiple definitions (see backend_mapping.h).
 */
#define PARALLEL_SYNC_default

/**
 * Skip sync after BoxLoop if accelerator present.
 * Must be the called as the last action inside the loop body.
 *
 * @note Multiple definitions (see backend_mapping.h).
 */
#define SKIP_PARALLEL_SYNC_default

/** Record an NVTX range for NSYS if accelerator present. @note Multiple definitions (see backend_mapping.h). */
#define PUSH_NVTX_default(name, cid)

/** Stop recording an NVTX range for NSYS if accelerator present. @note Multiple definitions (see backend_mapping.h). */
#define POP_NVTX_default


#endif
