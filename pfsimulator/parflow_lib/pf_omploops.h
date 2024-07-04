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

#ifndef _PF_OMPLOOPS_H_
#define _PF_OMPLOOPS_H_

/** @file
 * @brief OpenMP functions and loop definitions
 */

#ifndef PARFLOW_HAVE_OMP

#define NO_OMP_PARALLEL

#else

#include <omp.h>
#include <stdarg.h>

/**
 * @brief Assertion to prevent errant parallel region entries
 *
 * Function macro that checks active OpenMP parallel levels.
 * If currently in a parallel region, prints function name and line to stderr and calls exit(-1).
 *
 **/
#define NO_OMP_PARALLEL                                                 \
  if (omp_get_active_level() != 0)                                      \
  {                                                                     \
    fprintf(stderr,                                                     \
            "Node (%d) | Error: Hit Parallel region in %s:%d when not allowed\n", \
            amps_Rank(amps_CommWorld), __FUNCTION__, __LINE__);         \
    exit(-1);                                                           \
  } else {};

/**
 * @brief Interior macro for creating OpenMP pragma statements inside macro.  Relies on C99 _Pragma operator.
 **/
#define PRAGMA(args) _Pragma( #args )

/**************************************************************************
 * Reduction/Atomic Variants
 **************************************************************************/
extern "C++"{

#include <tuple>

  template <typename T>
  struct function_traits
    : public function_traits<decltype(&T::operator())>
  {};
  // For generic types, directly use the result of the signature of its 'operator()'

  template <typename ClassType, typename ReturnType, typename... Args>
  struct function_traits<ReturnType(ClassType::*)(Args...) const>
  // we specialize for pointers to member function
  {
    enum { arity = sizeof...(Args) };
    // arity is the number of arguments.

    typedef ReturnType result_type;

    template <size_t i>
    struct arg
    {
      typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
      // the i-th argument is equivalent to the i-th tuple element of a tuple
      // composed of those arguments.
    };
  };


#define PlusEquals_omp(a, b) AtomicAdd(&(a), b)
  template<typename T>
  static inline void AtomicAdd(T *addr, T val)
  {
    #pragma omp atomic update
    *addr += val;
  }

  /** Helper struct for type comparison (not for direct use). */
  template <typename T>
  struct ReduceMaxRes {T lambda_result;};
#define ReduceMax_omp(a, b) struct ReduceMaxRes<decltype(a)> reduce_struct {.lambda_result = b}; return reduce_struct;

  /** Helper struct for type comparison (not for direct use). */
  template <typename T>
  struct ReduceMinRes {T lambda_result;};
#define ReduceMin_omp(a, b) struct ReduceMinRes<decltype(a)> reduce_struct {.lambda_result = b}; return reduce_struct;

  /** Helper struct for type comparison (not for direct use). */
  template <typename T>
  struct ReduceSumRes {T lambda_result;};
#define ReduceSum_omp(a, b) struct ReduceSumRes<decltype(a)> reduce_struct {.lambda_result = b}; return reduce_struct;

  /** OpenMP BoxLoopReduceI1 definition.
      Last statement in the body of the loop must be one of the above Reduce helper structures.
      This macro will do compile-time type checking to determine which kind of reduction to perform.
      See innerprod.c or vector_utilities.c for example usages
   */
#define BoxLoopReduceI1_omp(sum,                                        \
                        i, j, k,                                        \
                        ix, iy, iz, nx, ny, nz,                         \
                        i1, nx1, ny1, nz1, sx1, sy1, sz1,               \
                        body)                                           \
  {                                                                     \
    auto lambda_body = [=](const int i, const int j, const int k,       \
                           const int i1)                                \
                       {                                                \
                         body;                                          \
                       };                                               \
                                                                        \
    int i1_start = i1;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
                                                                        \
    typedef function_traits<decltype(lambda_body)> traits;              \
    static_assert(std::is_same<traits::result_type,                     \
                  struct ReduceSumRes<decltype(sum)>>::value            \
                  ||                                                    \
                  std::is_same<traits::result_type,                     \
                  struct ReduceMaxRes<decltype(sum)>>::value            \
                  ||                                                    \
                  std::is_same<traits::result_type,                     \
                  struct ReduceMinRes<decltype(sum)>>::value,           \
                  "Not a valid reduction clause!  Check compiler error message for file and line number." \
                  );                                                    \
                                                                        \
    if (std::is_same<traits::result_type,                               \
        struct ReduceSumRes<decltype(sum)>>::value)                     \
      {                                                                 \
        PRAGMA(omp parallel for reduction(+:sum) collapse(3) private(i, j, k, i1)) \
          for (k = iz; k < iz + nz; k++)                                \
            {                                                           \
              for (j = iy; j < iy + ny; j++)                            \
                {                                                       \
                  for (i = ix; i < ix + nx; i++)                        \
                    {                                                   \
                      i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz), \
                                   nx, ny, sx1, PV_jinc_1, PV_kinc_1);  \
                      auto rhs = lambda_body(i, j, k, i1);              \
                      sum += rhs.lambda_result;                         \
                    }                                                   \
                }                                                       \
            }                                                           \
      }                                                                 \
    else                                                                \
      if (std::is_same<traits::result_type,                             \
          struct ReduceMaxRes<decltype(sum)>>::value)                   \
        {                                                               \
          PRAGMA(omp parallel for reduction(max:sum) collapse(3) private(i, j, k, i1)) \
            for (k = iz; k < iz + nz; k++)                              \
              {                                                         \
                for (j = iy; j < iy + ny; j++)                          \
                  {                                                     \
                    for (i = ix; i < ix + nx; i++)                      \
                      {                                                 \
                        i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz), \
                                     nx, ny, sx1, PV_jinc_1, PV_kinc_1); \
                        auto rhs = lambda_body(i, j, k, i1);            \
                        pfmax_atomic(sum, rhs.lambda_result);           \
                      }                                                 \
                  }                                                     \
              }                                                         \
        }                                                               \
      else                                                              \
        if (std::is_same<traits::result_type,                           \
            struct ReduceMinRes<decltype(sum)>>::value)                 \
          {                                                             \
            PRAGMA(omp parallel for reduction(min:sum) collapse(3) private(i, j, k, i1)) \
              for (k = iz; k < iz + nz; k++)                            \
                {                                                       \
                  for (j = iy; j < iy + ny; j++)                        \
                    {                                                   \
                      for (i = ix; i < ix + nx; i++)                    \
                        {                                               \
                          i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz), \
                                       nx, ny, sx1, PV_jinc_1, PV_kinc_1); \
                          auto rhs = lambda_body(i, j, k, i1);          \
                          pfmax_atomic(sum, rhs.lambda_result);         \
                        }                                               \
                    }                                                   \
                }                                                       \
          }                                                             \
  }

  /** OpenMP BoxLoopReduceI2 definition.
      Last statement in the body of the loop must be one of the above Reduce helper structures.
      This macro will do compile-time type checking to determine which kind of reduction to perform.
  */
#define BoxLoopReduceI2_omp(sum,                                        \
                        i, j, k,                                        \
                        ix, iy, iz, nx, ny, nz,                         \
                        i1, nx1, ny1, nz1, sx1, sy1, sz1,               \
                        i2, nx2, ny2, nz2, sx2, sy2, sz2,               \
                        body)                                           \
  {                                                                     \
    auto lambda_body = [=](const int i, const int j, const int k,       \
                           const int i1, const int i2)                  \
                       {                                                \
                         body;                                          \
                       };                                               \
                                                                        \
    int i1_start = i1;                                                  \
    int i2_start = i2;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2); \
                                                                        \
    typedef function_traits<decltype(lambda_body)> traits;              \
    static_assert(std::is_same<traits::result_type,                     \
                  struct ReduceSumRes<decltype(sum)>>::value            \
                  ||                                                    \
                  std::is_same<traits::result_type,                     \
                  struct ReduceMaxRes<decltype(sum)>>::value            \
                  ||                                                    \
                  std::is_same<traits::result_type,                     \
                  struct ReduceMinRes<decltype(sum)>>::value,           \
                  "Not a valid reduction clause!  Check compiler error message for file and line number." \
                  );                                                    \
                                                                        \
    if (std::is_same<traits::result_type,                               \
        struct ReduceSumRes<decltype(sum)>>::value)                     \
      {                                                                 \
        PRAGMA(omp parallel for reduction(+:sum) collapse(3) private(i, j, k, i1, i2)) \
          for (k = iz; k < iz + nz; k++)                                \
            {                                                           \
              for (j = iy; j < iy + ny; j++)                            \
                {                                                       \
                  for (i = ix; i < ix + nx; i++)                        \
                    {                                                   \
                      i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz), \
                                   nx, ny, sx1, PV_jinc_1, PV_kinc_1);  \
                      i2 = INC_IDX(i2_start, (i - ix), (j - iy), (k - iz), \
                                   nx, ny, sx2, PV_jinc_2, PV_kinc_2);  \
                      auto rhs = lambda_body(i, j, k, i1, i2);          \
                      sum += rhs.lambda_result;                         \
                    }                                                   \
                }                                                       \
            }                                                           \
      }                                                                 \
    else                                                                \
      if (std::is_same<traits::result_type,                             \
          struct ReduceMaxRes<decltype(sum)>>::value)                   \
        {                                                               \
          PRAGMA(omp parallel for reduction(max:sum) collapse(3) private(i, j, k, i1, i2)) \
            for (k = iz; k < iz + nz; k++)                              \
              {                                                         \
                for (j = iy; j < iy + ny; j++)                          \
                  {                                                     \
                    for (i = ix; i < ix + nx; i++)                      \
                      {                                                 \
                        i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz), \
                                     nx, ny, sx1, PV_jinc_1, PV_kinc_1); \
                        i2 = INC_IDX(i2_start, (i - ix), (j - iy), (k - iz), \
                                     nx, ny, sx2, PV_jinc_2, PV_kinc_2); \
                        auto rhs = lambda_body(i, j, k, i1, i2);        \
                        pfmax_atomic(sum, rhs.lambda_result);           \
                      }                                                 \
                  }                                                     \
              }                                                         \
        }                                                               \
      else                                                              \
        if (std::is_same<traits::result_type,                           \
            struct ReduceMinRes<decltype(sum)>>::value)                 \
          {                                                             \
            PRAGMA(omp parallel for reduction(min:sum) collapse(3) private(i, j, k, i1, i2)) \
              for (k = iz; k < iz + nz; k++)                            \
                {                                                       \
                  for (j = iy; j < iy + ny; j++)                        \
                    {                                                   \
                      for (i = ix; i < ix + nx; i++)                    \
                        {                                               \
                          i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz), \
                                       nx, ny, sx1, PV_jinc_1, PV_kinc_1); \
                          i2 = INC_IDX(i2_start, (i - ix), (j - iy), (k - iz), \
                                       nx, ny, sx2, PV_jinc_2, PV_kinc_2); \
                          auto rhs = lambda_body(i, j, k, i1, i2);      \
                          pfmax_atomic(sum, rhs.lambda_result);         \
                        }                                               \
                    }                                                   \
                }                                                       \
          }                                                             \
  }

} // Extern C++


/* @MCB:
   NOTE | These are expected to be used inside of OMP reduction loops with the max or min clause
   They are NOT atomic, the reduction clause it meant to handle that for better performance
*/
#define pfmin_atomic(a,b) (a) = (a) < (b) ? (a) : (b)

#define pfmax_atomic(a,b) (a) = (a) > (b) ? (a) : (b)

/**************************************************************************
 * OpenMP BoxLoop Variants
 **************************************************************************/

/**
 * @brief Calculate the appropriate accessor index for use in BoxLoops
 *
 * Pragma uses `declare simd` to tell the compiler this is SIMD safe.
 * `uniform` pragma tells the compiler which parameters will be constant for each SIMD pass.
 * BoxLoops are arranged in k -> j -> i ordering, so the only thing changing on each interior loop pass is i.
 *
 * @param idx Original starting index
 * @param i Current i of loop
 * @param j Current j of loop
 * @param k Current k of loop
 * @param nx Subregion X size
 * @param ny Subregion Y size
 * @param sx Subregion X striding factor
 * @param jinc Striding factor for J
 * @param kinc Striding factor for K
 * @return Current accessor index calculated from input parameters
 **/
#pragma omp declare simd uniform(idx, j, k, nx, ny, sx, jinc, kinc)
inline int
INC_IDX(int idx, int i, int j, int k,
        int nx, int ny, int sx,
        int jinc, int kinc)
{
  return (k * kinc + (k * ny + j) * jinc +
          (k * ny * nx + j * nx + i) * sx) + idx;
}

#define BoxLoopI0_omp(i, j, k, ix, iy, iz, nx, ny, nz, body)    \
  {                                                             \
    PRAGMA(omp parallel for collapse(3) private(i, j, k))       \
      for (k = iz; k < iz + nz; k++)                            \
      {                                                         \
        for (j = iy; j < iy + ny; j++)                          \
        {                                                       \
          for (i = ix; i < ix + nx; i++)                        \
          {                                                     \
            body;                                               \
          }                                                     \
        }                                                       \
      }                                                         \
  }

#define BoxLoopI1_omp(i, j, k,                                          \
                  ix, iy, iz, nx, ny, nz,                               \
                  i1, nx1, ny1, nz1, sx1, sy1, sz1,                     \
                  body)                                                 \
  {                                                                     \
    int i1_start = i1;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    PRAGMA(omp parallel for collapse(3) private(i, j, k, i1))           \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx1, PV_jinc_1, PV_kinc_1);            \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }

#define BoxLoopI2_omp(i, j, k,                                          \
                   ix, iy, iz, nx, ny, nz,                              \
                   i1, nx1, ny1, nz1, sx1, sy1, sz1,                    \
                   i2, nx2, ny2, nz2, sx2, sy2, sz2,                    \
                   body)                                                \
  {                                                                     \
    int i1_start = i1;                                                  \
    int i2_start = i2;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2); \
    PRAGMA(omp parallel for collapse(3) private(i, j, k, i1, i2))       \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx1, PV_jinc_1, PV_kinc_1);            \
            i2 = INC_IDX(i2_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx2, PV_jinc_2, PV_kinc_2);            \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }

#define BoxLoopI3_omp(i, j, k,                                          \
                   ix, iy, iz, nx, ny, nz,                              \
                   i1, nx1, ny1, nz1, sx1, sy1, sz1,                    \
                   i2, nx2, ny2, nz2, sx2, sy2, sz2,                    \
                   i3, nx3, ny3, nz3, sx3, sy3, sz3,                    \
                   body)                                                \
  {                                                                     \
    int i1_start = i1;                                                  \
    int i2_start = i2;                                                  \
    int i3_start = i3;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2); \
    DeclareInc(PV_jinc_3, PV_kinc_3, nx, ny, nz, nx3, ny3, nz3, sx3, sy3, sz3); \
    PRAGMA(omp parallel for collapse(3) private(i, j, k, i1, i2, i3))   \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            i1 = INC_IDX(i1_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx1, PV_jinc_1, PV_kinc_1);            \
            i2 = INC_IDX(i2_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx2, PV_jinc_2, PV_kinc_2);            \
            i3 = INC_IDX(i3_start, (i - ix), (j - iy), (k - iz),        \
                         nx, ny, sx3, PV_jinc_3, PV_kinc_3);            \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }

/**************************************************************************
 * BoxLoop Reduction Variants
 **************************************************************************/

#define BoxLoopReduceI0_omp(sum,                                        \
                        i, j, k,                                        \
                        ix, iy, iz, nx, ny, nz,                         \
                        body)                                           \
  {                                                                     \
    PRAGMA(omp parallel for reduction(+:sum) collapse(3) private(i, j, k)) \
      for (k = iz; k < iz + nz; k++)                                    \
      {                                                                 \
        for (j = iy; j < iy + ny; j++)                                  \
        {                                                               \
          for (i = ix; i < ix + nx; i++)                                \
          {                                                             \
            body;                                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
  }


/**************************************************************************
 * SIMD BoxLoop Variants
 * @MCB: Note, currently not used.  May or may not yield better performance.
 * Many BoxLoop bodies are very straightforward and more heavy-handed simd
 * hints may yield better vectorization, even when involving conditions.
 * OpenMP 4.5+ specs have done a lot of work on improving SIMD pragmas.
 **************************************************************************/
#define SIMD_BoxLoopI0(i, j, k,                             \
                       ix, iy, iz,                          \
                       nx, ny, nz,                          \
                       body)                                \
  {                                                         \
    PRAGMA(omp pragma collapse(2) private(j, k))            \
      for (k = iz; k < iz + nz; k++)                        \
      {                                                     \
        for (j = iy; j < iy + ny; j++)                      \
        {                                                   \
          PRAGMA(omp simd private(i))                       \
            for (i = ix; i < ix + nx; i++)                  \
            {                                               \
              body;                                         \
            }                                               \
        }                                                   \
      }                                                     \
  }



/**************************************************************************
 * OpenMP Cluster LoopBox Variants
 **************************************************************************/

/**
 * @brief Calculate ival value in boundary condition loop body for a given iteration
 *
 *  Input variables depend on the bounds of the current box being iterated over.
 *
 * @param diff X/Y/Z offset of the current box
 * @param a Normalized J or K iteration
 * @param b Normalized I or J iteration
 * @param prev Previous maximum ival value
*/
#define CALC_IVAL(diff, a, b, prev) ((diff) * (a) + (a) + (b)) + (prev)

#define GrGeomPatchLoopBoxesNoFdir_omp(i, j, k, grgeom, patch_num, ovrlnd,     \
                                   ix, iy, iz, nx, ny, nz,              \
                                   locals, setup,                       \
                                   f_left, f_right,                     \
                                   f_down, f_up,                        \
                                   f_back, f_front,                     \
                                   finalize)                            \
  PRAGMA(omp parallel)                                                  \
  {                                                                     \
    /* @MCB: Redeclare locals instead of using OMP Private pragma for */\
    /* architecture portability. */                                     \
    UNPACK(locals);                                                     \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;                 \
    int *PV_visiting = NULL;                                            \
    PF_UNUSED(PV_visiting);                                             \
                                                                        \
    int prev_ival = 0;                                                  \
    for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)             \
    {                                                                   \
      BoxArray* boxes = GrGeomSolidPatchBoxes(grgeom, patch_num, PV_f); \
      for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)      \
      {                                                                 \
        Box box = BoxArrayGetBox(boxes, PV_box);                        \
        /* find octree and region intersection */                       \
        PV_ixl = pfmax(ix, box.lo[0]);                                  \
        PV_iyl = pfmax(iy, box.lo[1]);                                  \
        PV_izl = pfmax(iz, box.lo[2]);                                  \
        PV_ixu = pfmin((ix + nx - 1), box.up[0]);                       \
        PV_iyu = pfmin((iy + ny - 1), box.up[1]);                       \
        PV_izu = pfmin((iz + nz - 1), box.up[2]);                       \
                                                                        \
        /* Used to calculate individual ival values for each threads iteration */\
        int PV_diff_x = PV_ixu - PV_ixl;                                \
        int PV_diff_y = PV_iyu - PV_iyl;                                \
        int PV_diff_z = PV_izu - PV_izl;                                \
        int x_scale = !!PV_diff_x;                                      \
        int y_scale = !!PV_diff_y;                                      \
        int z_scale = !!PV_diff_z;                                      \
                                                                        \
        PRAGMA(omp for collapse(3) private(i, j, k, ival))              \
          for (k = PV_izl; k <= PV_izu; k++)                            \
          {                                                             \
            for (j = PV_iyl; j <= PV_iyu; j++)                          \
            {                                                           \
              for (i = PV_ixl; i <= PV_ixu; i++)                        \
              {                                                         \
                int PV_tmp_i = i - PV_ixl;                              \
                int PV_tmp_j = j - PV_iyl;                              \
                int PV_tmp_k = k - PV_izl;                              \
                if (!z_scale) {                                         \
                  ival = CALC_IVAL(PV_diff_x, PV_tmp_j, PV_tmp_i, prev_ival);     \
                } else if (!y_scale) {                                  \
                  ival = CALC_IVAL(PV_diff_x, PV_tmp_k, PV_tmp_i, prev_ival);     \
                } else {                                                \
                  ival = CALC_IVAL(PV_diff_y, PV_tmp_k, PV_tmp_j, prev_ival);     \
                }                                                       \
                setup;                                                  \
                                                                        \
                switch(PV_f)                                            \
                {                                                       \
                  f_left;                                               \
                  f_right;                                              \
                  f_down;                                               \
                  f_up;                                                 \
                  f_back;                                               \
                  f_front;                                              \
                }                                                       \
                                                                        \
                finalize;                                               \
              }                                                         \
            }                                                           \
          }                                                             \
        prev_ival += (PV_diff_x+1) * (PV_diff_y+1) * (PV_diff_z+1);     \
      }                                                                 \
    }                                                                   \
  }


/* Per SGS suggestion on 12/3/2008 */
static const int FDIR_TABLE[6][3] = {
  {-1, 0,  0}, // Left
  {1,  0,  0}, // Right
  {0, -1,  0}, // Down
  {0,  1,  0}, // Up
  {0,  0, -1}, // Back
  {0,  0,  1}, // Front
};


#define GrGeomPatchLoopBoxes_omp(i, j, k, fdir, grgeom, patch_num,      \
                             ix, iy, iz, nx, ny, nz, body)              \
  PRAGMA(omp parallel)                                                  \
  {                                                                     \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;                 \
    int *PV_visiting = NULL;                                            \
    PF_UNUSED(PV_visiting);                                             \
                                                                        \
    int prev_ival = 0;                                                  \
    for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)             \
    {                                                                   \
      const int *fdir = FDIR_TABLE[PV_f];                               \
                                                                        \
      BoxArray* boxes = GrGeomSolidPatchBoxes(grgeom, patch_num, PV_f); \
      for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)      \
      {                                                                 \
        Box box = BoxArrayGetBox(boxes, PV_box);                        \
        /* find octree and region intersection */                       \
        PV_ixl = pfmax(ix, box.lo[0]);                                  \
        PV_iyl = pfmax(iy, box.lo[1]);                                  \
        PV_izl = pfmax(iz, box.lo[2]);                                  \
        PV_ixu = pfmin((ix + nx - 1), box.up[0]);                       \
        PV_iyu = pfmin((iy + ny - 1), box.up[1]);                       \
        PV_izu = pfmin((iz + nz - 1), box.up[2]);                       \
                                                                        \
        int PV_diff_x = PV_ixu - PV_ixl;                                \
        int PV_diff_y = PV_iyu - PV_iyl;                                \
        int PV_diff_z = PV_izu - PV_izl;                                \
        int x_scale = !!PV_diff_x;                                      \
        int y_scale = !!PV_diff_y;                                      \
        int z_scale = !!PV_diff_z;                                      \
                                                                        \
        PRAGMA(omp for collapse(3) private(i, j, k))                    \
          for (k = PV_izl; k <= PV_izu; k++)                            \
            for (j = PV_iyl; j <= PV_iyu; j++)                          \
              for (i = PV_ixl; i <= PV_ixu; i++)                        \
              {                                                         \
                int ival = 0;                                           \
                int PV_tmp_i = i - PV_ixl;                              \
                int PV_tmp_j = j - PV_iyl;                              \
                int PV_tmp_k = k - PV_izl;                              \
                if (!z_scale) {                                         \
                  ival = CALC_IVAL(PV_diff_x, PV_tmp_j, PV_tmp_i, prev_ival);     \
                } else if (!y_scale) {                                  \
                  ival = CALC_IVAL(PV_diff_x, PV_tmp_k, PV_tmp_i, prev_ival); \
                } else {                                                \
                  ival = CALC_IVAL(PV_diff_y, PV_tmp_k, PV_tmp_j, prev_ival); \
                }                                                       \
                                                                        \
                body;                                                   \
              }                                                         \
        prev_ival += (PV_diff_x+1) * (PV_diff_y+1) * (PV_diff_z+1);     \
      }                                                                 \
    }                                                                   \
  }

#define GrGeomInLoopBoxes_omp(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body) \
    PRAGMA(omp parallel)                                                \
    {                                                                   \
      int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;               \
      int *PV_visiting = NULL;                                          \
      PF_UNUSED(PV_visiting);                                           \
      BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);               \
      for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)      \
      {                                                                 \
        Box box = BoxArrayGetBox(boxes, PV_box);                        \
        /* find octree and region intersection */                       \
        PV_ixl = pfmax(ix, box.lo[0]);                                  \
        PV_iyl = pfmax(iy, box.lo[1]);                                  \
        PV_izl = pfmax(iz, box.lo[2]);                                  \
        PV_ixu = pfmin((ix + nx - 1), box.up[0]);                       \
        PV_iyu = pfmin((iy + ny - 1), box.up[1]);                       \
        PV_izu = pfmin((iz + nz - 1), box.up[2]);                       \
                                                                        \
        PRAGMA(omp for collapse(3) private(i, j, k))                    \
          for (k = PV_izl; k <= PV_izu; k++)                            \
            for (j = PV_iyl; j <= PV_iyu; j++)                          \
              for (i = PV_ixl; i <= PV_ixu; i++)                        \
              {                                                         \
                body;                                                   \
              }                                                         \
      }                                                                 \
    }


#define GrGeomSurfLoopBoxes_omp(i, j, k, fdir, grgeom, ix, iy, iz, nx, ny, nz, body) \
  PRAGMA(omp parallel firstprivate(fdir))                               \
  {                                                                     \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;                 \
    int *PV_visiting = NULL;                                            \
    PF_UNUSED(PV_visiting);                                             \
    for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)             \
    {                                                                   \
      const int *fdir = FDIR_TABLE[PV_f];                               \
                                                                        \
      BoxArray* boxes = GrGeomSolidSurfaceBoxes(grgeom, PV_f);          \
      for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)      \
      {                                                                 \
        Box box = BoxArrayGetBox(boxes, PV_box);                        \
        /* find octree and region intersection */                       \
        PV_ixl = pfmax(ix, box.lo[0]);                                  \
        PV_iyl = pfmax(iy, box.lo[1]);                                  \
        PV_izl = pfmax(iz, box.lo[2]);                                  \
        PV_ixu = pfmin((ix + nx - 1), box.up[0]);                       \
        PV_iyu = pfmin((iy + ny - 1), box.up[1]);                       \
        PV_izu = pfmin((iz + nz - 1), box.up[2]);                       \
                                                                        \
        PRAGMA(omp for collapse(3) private(i, j, k))                    \
        for (k = PV_izl; k <= PV_izu; k++)                              \
          for (j = PV_iyl; j <= PV_iyu; j++)                            \
            for (i = PV_ixl; i <= PV_ixu; i++)                          \
            {                                                           \
              body;                                                     \
            }                                                           \
      }                                                                 \
    }                                                                   \
  }

#endif // PARFLOW_HAVE_OMP
#endif // _PF_OMPLOOPS_H_
