/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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
/*****************************************************************************
*
* General structures and values
*
*****************************************************************************/

#ifndef _GENERAL_HEADER
#define _GENERAL_HEADER

#include <float.h>

/*--------------------------------------------------------------------------
 * Error macros
 *--------------------------------------------------------------------------*/

#define PARFLOW_ERROR(X)                \
  do {                                  \
    _amps_Abort(X, __FILE__, __LINE__); \
  } while (0)

/*--------------------------------------------------------------------------
 * Define memory allocation routines
 *--------------------------------------------------------------------------*/

/*--------------------------------------
 * Check memory allocation
 *--------------------------------------*/

#ifdef PF_MEMORY_ALLOC_CHECK

#define talloc(type, count) \
  (type*)malloc_chk((unsigned int)((count) * sizeof(type)), __FILE__, __LINE__)

#define ctalloc(type, count)                                           \
  (type*)calloc_chk((unsigned int)(count), (unsigned int)sizeof(type), \
                    __FILE__, __LINE__)

/* note: the `else' is required to guarantee termination of the `if' */
#define tfree(ptr) if (ptr) free(ptr); else {}

/*--------------------------------------
 * Do not check memory allocation
 *--------------------------------------*/

#else

#define talloc(type, count) \
  (((count) > 0) ? (type*)malloc(sizeof(type) * (unsigned int)(count)) : NULL)

#define ctalloc(type, count) \
  (((count) > 0) ? (type*)calloc((unsigned int)(count), (unsigned int)sizeof(type)) : NULL)

/* note: the `else' is required to guarantee termination of the `if' */
#define tfree(ptr) if (ptr) free(ptr); else {}

#endif


/*--------------------------------------------------------------------------
 * TempData macros
 *--------------------------------------------------------------------------*/

#define NewTempData(temp_data_sz)  amps_CTAlloc(double, (temp_data_sz))

#define FreeTempData(temp_data)    amps_TFree(temp_data)


/*--------------------------------------------------------------------------
 * Define various functions
 *--------------------------------------------------------------------------*/

#ifndef pfmax
#define pfmax(a, b)  (((a) < (b)) ? (b) : (a))
#endif
#ifndef pfmin
#define pfmin(a, b)  (((a) < (b)) ? (a) : (b))
#endif

#ifndef pfmax_atomic
#define pfmax_atomic(a, b)  if(a < b) a = b
#endif
#ifndef pfmin_atomic
#define pfmin_atomic(a, b)  if(a > b) a = b
#endif

#ifndef pfround
#define pfround(x)  (((x) < 0.0) ? ((int)(x - 0.5)) : ((int)(x + 0.5)))
#endif

#ifndef PlusEquals
#define PlusEquals(a, b) (a += b)
#endif

#ifndef ReduceSum
#define ReduceSum(a, b) (a += b)
#endif

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

//CUDA compiler specific keywords
#ifndef __host__
  #define __host__
#endif
#ifndef __device__
  #define __device__
#endif
#ifndef __managed__
  #define __managed__
#endif
#ifndef __restrict__
  #define __restrict__
#endif

// Helper macros for the new BC loop interface
#define InParallel
#define NewParallel
#define NoWait

#define EMPTY()
#define DEFER(x) x EMPTY()
#define DEFER2(x) x EMPTY EMPTY() ()
#define DEFER3(x) x EMPTY EMPTY EMPTY() () ()

#undef LOCALS
#define LOCALS(...) DEFER3(_LOCALS)(__VA_ARGS__)
#define _LOCALS(...) __VA_ARGS__
#define NO_LOCALS

//Memory Prefetching
#define MemPrefetchDeviceToHost(ptr, size, stream)
#define MemPrefetchHostToDevice(ptr, size, gpuid, stream)

//Parallel synchronizations
#define PARALLEL_SYNC
#define SKIP_PARALLEL_SYNC

//NVTX Ranges for NSYS profiling
#ifdef HAVE_CUDA
  #include "nvToolsExt.h"
  #define PUSH_NVTX(name,cid)                                                              \
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
  #define POP_NVTX nvtxRangePop();
#else
  #define PUSH_NVTX(name,cid)
  #define POP_NVTX
#endif

#endif
