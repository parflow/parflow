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
 * @brief Contains mappings for backend-specific preprocessor macros.
 */

/* PF_COMP_UNIT_TYPE determines the behavior of the NVCC compilation unit and/or OpenMP loops:
 * ------------------------------------------------------------
 * CUDA
 * ------------------------------------------------------------
 * 1:     NVCC compiler, Unified Memory allocation, Parallel loops on GPUs
 * 2:     NVCC compiler, Unified Memory allocation, Sequential loops on host
 * Other: NVCC compiler, Standard heap allocation, Sequential loops on host
 *
 *
 * ------------------------------------------------------------
 * OpenMP
 * ------------------------------------------------------------
 * 1:     CXX compiler, Standard heap allocation, Parallel loops on CPU
 * 2:     CXX compiler, Standard heap allocation, Sequential loops on CPU
 * Other: CXX compiler, Standard heap allocation, Sequential loops on CPU
 */

/* Include headers depending on the accelerator backend */
#ifdef PARFLOW_HAVE_KOKKOS

  #define ACC_ID _kokkos

  #include "pf_devices.h"

  #if PF_COMP_UNIT_TYPE == 1
    #include "pf_kokkosloops.h"
  #elif PF_COMP_UNIT_TYPE == 2
    #include "pf_kokkosmalloc.h"
  #endif

#elif defined(PARFLOW_HAVE_CUDA)

  #define ACC_ID _cuda

  #include "pf_devices.h"

  #if PF_COMP_UNIT_TYPE == 1
    #include "pf_cudaloops.h"
  #elif PF_COMP_UNIT_TYPE == 2
    #include "pf_cudamalloc.h"
  #endif

#elif defined(PARFLOW_HAVE_OMP)

  #define ACC_ID _omp

  #if PF_COMP_UNIT_TYPE == 1
    #include "pf_omploops.h" // For OMP loops
  #endif

#endif


/*--------------------------------------------------------------------------
 * Map backend-dependent macros
 *--------------------------------------------------------------------------*/

#define EMPTY()
#define DEFER(x) x EMPTY()
#define PASTER(x, y) x ## y
#define EVALUATOR(x, y) PASTER(x, y)
#define CHOOSE_BACKEND(name, id) EVALUATOR(name, id)

// Memory management

#if defined(talloc_amps_cuda) || defined(talloc_amps_kokkos) || defined(talloc_amps_omp)
  #define talloc_amps CHOOSE_BACKEND(DEFER(talloc_amps), ACC_ID)
#else
  #define talloc_amps talloc_amps_default
#endif

#if defined(ctalloc_amps_cuda) || defined(ctalloc_amps_kokkos) || defined(ctalloc_amps_omp)
  #define ctalloc_amps CHOOSE_BACKEND(DEFER(ctalloc_amps), ACC_ID)
#else
  #define ctalloc_amps ctalloc_amps_default
#endif

#if defined(tfree_amps_cuda) || defined(tfree_amps_kokkos) || defined(tfree_amps_omp)
  #define tfree_amps CHOOSE_BACKEND(DEFER(tfree_amps), ACC_ID)
#else
  #define tfree_amps tfree_amps_default
#endif

#if defined(talloc_cuda) || defined(talloc_kokkos) || defined(talloc_omp)
  #define talloc CHOOSE_BACKEND(DEFER(talloc), ACC_ID)
#else
  #define talloc talloc_default
#endif

#if defined(ctalloc_cuda) || defined(ctalloc_kokkos) || defined(ctalloc_omp)
  #define ctalloc CHOOSE_BACKEND(DEFER(ctalloc), ACC_ID)
#else
  #define ctalloc ctalloc_default
#endif

#if defined(tfree_cuda) || defined(tfree_kokkos) || defined(tfree_omp)
  #define tfree CHOOSE_BACKEND(DEFER(tfree), ACC_ID)
#else
  #define tfree tfree_default
#endif

#if defined(tmemcpy_cuda) || defined(tmemcpy_kokkos) || defined(tmemcpy_omp)
  #define tmemcpy CHOOSE_BACKEND(DEFER(tmemcpy), ACC_ID)
#else
  #define tmemcpy tmemcpy_default
#endif

#if defined(MemPrefetchDeviceToHost_cuda) || defined(MemPrefetchDeviceToHost_kokkos) || defined(MemPrefetchDeviceToHost_omp)
  #define MemPrefetchDeviceToHost CHOOSE_BACKEND(DEFER(MemPrefetchDeviceToHost), ACC_ID)
#else
  #define MemPrefetchDeviceToHost MemPrefetchDeviceToHost_default
#endif

#if defined(MemPrefetchHostToDevice_cuda) || defined(MemPrefetchHostToDevice_kokkos) || defined(MemPrefetchHostToDevice_omp)
  #define MemPrefetchHostToDevice CHOOSE_BACKEND(DEFER(MemPrefetchHostToDevice), ACC_ID)
#else
  #define MemPrefetchHostToDevice MemPrefetchHostToDevice_default
#endif

#if defined(SKIP_PARALLEL_SYNC_cuda) || defined(SKIP_PARALLEL_SYNC_kokkos) || defined(SKIP_PARALLEL_SYNC_omp)
  #define SKIP_PARALLEL_SYNC CHOOSE_BACKEND(DEFER(SKIP_PARALLEL_SYNC), ACC_ID)
#else
  #define SKIP_PARALLEL_SYNC SKIP_PARALLEL_SYNC_default
#endif

#if defined(PARALLEL_SYNC_cuda) || defined(PARALLEL_SYNC_kokkos) || defined(PARALLEL_SYNC_omp)
  #define PARALLEL_SYNC CHOOSE_BACKEND(DEFER(PARALLEL_SYNC), ACC_ID)
#else
  #define PARALLEL_SYNC PARALLEL_SYNC_default
#endif

// General

#if defined(PlusEquals_cuda) || defined(PlusEquals_kokkos) || defined(PlusEquals_omp)
  #define PlusEquals CHOOSE_BACKEND(DEFER(PlusEquals), ACC_ID)
#else
  #define PlusEquals PlusEquals_default
#endif

#if defined(ReduceMax_cuda) || defined(ReduceMax_kokkos) || defined(ReduceMax_omp)
  #define ReduceMax CHOOSE_BACKEND(DEFER(ReduceMax), ACC_ID)
#else
  #define ReduceMax ReduceMax_default
#endif

#if defined(ReduceMin_cuda) || defined(ReduceMin_kokkos) || defined(ReduceMin_omp)
  #define ReduceMin CHOOSE_BACKEND(DEFER(ReduceMin), ACC_ID)
#else
  #define ReduceMin ReduceMin_default
#endif

#if defined(ReduceSum_cuda) || defined(ReduceSum_kokkos) || defined(ReduceSum_omp)
  #define ReduceSum CHOOSE_BACKEND(DEFER(ReduceSum), ACC_ID)
#else
  #define ReduceSum ReduceSum_default
#endif

#if defined(PUSH_NVTX_cuda)
  #define PUSH_NVTX PUSH_NVTX_cuda
#else
  #define PUSH_NVTX PUSH_NVTX_default
#endif

#if defined(POP_NVTX_cuda)
  #define POP_NVTX POP_NVTX_cuda
#else
  #define POP_NVTX POP_NVTX_default
#endif

// Loops

#if defined(BoxLoopI0_cuda) || defined(BoxLoopI0_kokkos) || defined(BoxLoopI0_omp)
  #define BoxLoopI0 CHOOSE_BACKEND(DEFER(BoxLoopI0), ACC_ID)
#else
  #define BoxLoopI0 BoxLoopI0_default
#endif

#if defined(BoxLoopI1_cuda) || defined(BoxLoopI1_kokkos) || defined(BoxLoopI1_omp)
  #define BoxLoopI1 CHOOSE_BACKEND(DEFER(BoxLoopI1), ACC_ID)
#else
  #define BoxLoopI1 BoxLoopI1_default
#endif

#if defined(BoxLoopI2_cuda) || defined(BoxLoopI2_kokkos) || defined(BoxLoopI2_omp)
  #define BoxLoopI2 CHOOSE_BACKEND(DEFER(BoxLoopI2), ACC_ID)
#else
  #define BoxLoopI2 BoxLoopI2_default
#endif

#if defined(BoxLoopI3_cuda) || defined(BoxLoopI3_kokkos) || defined(BoxLoopI3_omp)
  #define BoxLoopI3 CHOOSE_BACKEND(DEFER(BoxLoopI3), ACC_ID)
#else
  #define BoxLoopI3 BoxLoopI3_default
#endif

#if defined(BoxLoopReduceI1_cuda) || defined(BoxLoopReduceI1_kokkos) || defined(BoxLoopReduceI1_omp)
  #define BoxLoopReduceI1 CHOOSE_BACKEND(DEFER(BoxLoopReduceI1), ACC_ID)
#else
  #define BoxLoopReduceI1 BoxLoopReduceI1_default
#endif

#if defined(BoxLoopReduceI2_cuda) || defined(BoxLoopReduceI2_kokkos) || defined(BoxLoopReduceI2_omp)
  #define BoxLoopReduceI2 CHOOSE_BACKEND(DEFER(BoxLoopReduceI2), ACC_ID)
#else
  #define BoxLoopReduceI2 BoxLoopReduceI2_default
#endif

#if defined(BoxLoopReduceI3_cuda) || defined(BoxLoopReduceI3_kokkos) || defined(BoxLoopReduceI3_omp)
  #define BoxLoopReduceI3 CHOOSE_BACKEND(DEFER(BoxLoopReduceI3), ACC_ID)
#else
  #define BoxLoopReduceI3 BoxLoopReduceI3_default
#endif

#if defined(GrGeomInLoopBoxes_cuda) || defined(GrGeomInLoopBoxes_kokkos) || defined(GrGeomInLoopBoxes_omp)
  #define GrGeomInLoopBoxes CHOOSE_BACKEND(DEFER(GrGeomInLoopBoxes), ACC_ID)
#else
  #define GrGeomInLoopBoxes GrGeomInLoopBoxes_default
#endif

#if defined(GrGeomSurfLoopBoxes_cuda) || defined(GrGeomSurfLoopBoxes_kokkos) || defined(GrGeomSurfLoopBoxes_omp)
  #define GrGeomSurfLoopBoxes CHOOSE_BACKEND(DEFER(GrGeomSurfLoopBoxes), ACC_ID)
#else
  #define GrGeomSurfLoopBoxes GrGeomSurfLoopBoxes_default
#endif

#if defined(GrGeomPatchLoopBoxes_cuda) || defined(GrGeomPatchLoopBoxes_kokkos) || defined(GrGeomPatchLoopBoxes_omp)
  #define GrGeomPatchLoopBoxes CHOOSE_BACKEND(DEFER(GrGeomPatchLoopBoxes), ACC_ID)
#else
  #define GrGeomPatchLoopBoxes GrGeomPatchLoopBoxes_default
#endif

#if defined(GrGeomPatchLoopBoxesNoFdir_cuda) || defined(GrGeomPatchLoopBoxesNoFdir_kokkos) || defined(GrGeomPatchLoopBoxesNoFdir_omp)
  #define GrGeomPatchLoopBoxesNoFdir CHOOSE_BACKEND(DEFER(GrGeomPatchLoopBoxesNoFdir), ACC_ID)
#else
  #define GrGeomPatchLoopBoxesNoFdir GrGeomPatchLoopBoxesNoFdir_default
#endif

#if defined(GrGeomOctreeExteriorNodeLoop_cuda) || defined(GrGeomOctreeExteriorNodeLoop_kokkos) || defined(GrGeomOctreeExteriorNodeLoop_omp)
  #define GrGeomOctreeExteriorNodeLoop CHOOSE_BACKEND(DEFER(GrGeomOctreeExteriorNodeLoop), ACC_ID)
#else
  #define GrGeomOctreeExteriorNodeLoop GrGeomOctreeExteriorNodeLoop_default
#endif

#if defined(GrGeomOutLoop_cuda) || defined(GrGeomOutLoop_kokkos) || defined(GrGeomOutLoop_omp)
  #define GrGeomOutLoop CHOOSE_BACKEND(DEFER(GrGeomOutLoop), ACC_ID)
#else
  #define GrGeomOutLoop GrGeomOutLoop_default
#endif
