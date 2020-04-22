#ifndef _PF_OMPLOOPS_H_
#define _PF_OMPLOOPS_H_

#if ACC_BACKEND != BACKEND_OMP

#define NO_OMP_PARALLEL

#else

#include <omp.h>
#include <stdarg.h>

/**
 * @brief Assertion to prevent errant parallel region entries
 *
 * Function macro that checks active OpenMP parallel levels.
 * If current in a parallel region, prints function name and line to stderr and calls exit(-1).
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


//#undef pfmax_atomic
//#define pfmax_atomic(a, b) AtomicMax(&(a), b)
/* TODO: This should be replaced by turning the calling loop into reduction with (max: sum) clause */
  template <typename T>
  static inline void AtomicMax(T *addr, T val)
  {
#pragma omp critical
    {
      if (*addr < val)
        *addr = val;
    }
  }

//#undef pfmin_atomic
//#define pfmin_atomic(a, b) AtomicMin(&(a), b)
  template <typename T>
  static inline void AtomicMin(T *addr, T val)
  {
    #pragma omp critical
    {
      if (*addr > val)
        *addr = val;
    }
  }

#undef PlusEquals
#define PlusEquals(a, b) AtomicAdd(&(a), b)
  template<typename T>
  static inline void AtomicAdd(T *addr, T val)
  {
    #pragma omp atomic update
    *addr += val;
  }

  template <typename T>
  struct ReduceMaxRes {T lambda_result;};
#undef ReduceMax
#define ReduceMax(a, b) struct ReduceMaxRes<decltype(a)> reduce_struct {.lambda_result = b}; return reduce_struct;

  template <typename T>
  struct ReduceMinRes {T lambda_result;};
#undef ReduceMin
#define ReduceMin(a, b) struct ReduceMinRes<decltype(a)> reduce_struct {.lambda_result = b}; return reduce_struct;

  template <typename T>
  struct ReduceSumRes {T lambda_result;};
#undef ReduceSum
#define ReduceSum(a, b) struct ReduceSumRes<decltype(a)> reduce_struct {.lambda_result = b}; return reduce_struct;

#undef BoxLoopReduceI1
#define BoxLoopReduceI1(sum,                                            \
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

#undef BoxLoopReduceI2
#define BoxLoopReduceI2(sum,                                            \
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
#undef pfmin_atomic
#define pfmin_atomic(a,b) (a) = (a) < (b) ? (a) : (b)

#undef pfmax_atomic
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
 * TODO: Detail the math used to calculate appropriate offsets.
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

/**
 * @brief BoxLoopI0 for OpenMP.  Spawns new parallel for, collapses all 3 loops.
 *
 * TODO: May be a good candidate for `collapse(2)` and a `for simd` on the innermost loop.  See SIMD_BoxLoopI0 for an example.
 **/
#undef BoxLoopI0
#define BoxLoopI0(i, j, k, ix, iy, iz, nx, ny, nz, body)    \
  {                                                         \
    PRAGMA(omp parallel for collapse(3) private(i, j, k))   \
      for (k = iz; k < iz + nz; k++)                        \
      {                                                     \
        for (j = iy; j < iy + ny; j++)                      \
        {                                                   \
          for (i = ix; i < ix + nx; i++)                    \
          {                                                 \
            body;                                           \
          }                                                 \
        }                                                   \
      }                                                     \
  }

#undef BoxLoopI1
#define BoxLoopI1(i, j, k,                                              \
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

#undef BoxLoopI2
#define BoxLoopI2(i, j, k,                                              \
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

#undef BoxLoopI3
#define BoxLoopI3(i, j, k,                                              \
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

#undef BoxLoopReduceI0
#define BoxLoopReduceI0(sum,                                            \
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

#if 0
#undef BoxLoopReduceI1
#define BoxLoopReduceI1(sum,                                            \
                        i, j, k,                                        \
                        ix, iy, iz, nx, ny, nz,                         \
                        i1, nx1, ny1, nz1, sx1, sy1, sz1,               \
                        body)                                           \
  {                                                                     \
    int i1_start = i1;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    PRAGMA(omp parallel for reduction(+:sum) collapse(3) private(i, j, k, i1)) \
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


#undef BoxLoopReduceI2
#define BoxLoopReduceI2(sum,                                            \
                        i, j, k,                                        \
                        ix, iy, iz, nx, ny, nz,                         \
                        i1, nx1, ny1, nz1, sx1, sy1, sz1,               \
                        i2, nx2, ny2, nz2, sx2, sy2, sz2,               \
                        body)                                           \
  {                                                                     \
    int i1_start = i1;                                                  \
    int i2_start = i2;                                                  \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1); \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2); \
    PRAGMA(omp parallel for reduction(+:sum) collapse(3) private(i, j, k, i1, i2)) \
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
#endif


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

/* Used to calculate correct ival */
#define CALC_IVAL(diff, a, b, prev) ((diff) * (a) + (a) + (b)) + (prev)

#undef GrGeomPatchLoopBoxesNoFdir
#define GrGeomPatchLoopBoxesNoFdir(i, j, k, grgeom, patch_num,\
                                   ix, iy, iz, nx, ny, nz,    \
                                   locals, setup,            \
                                   f_left, f_right,          \
                                   f_down, f_up,             \
                                   f_back, f_front,          \
                                   finalize)                 \
  PRAGMA(omp parallel)                                                  \
  {                                                                     \
    /* @MCB: Redeclare locals instead of using OMP Private pragma for */ \
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


/* Per SGS suggestion on 12/3/2008
   @MCB: C won't allow *int to const *int conversion */
static const int FDIR_TABLE[6][3] = {
  {-1, 0,  0}, // Left
  {1,  0,  0}, // Right
  {0, -1,  0}, // Down
  {0,  1,  0}, // Up
  {0,  0, -1}, // Back
  {0,  0,  1}, // Front
};


#undef GrGeomPatchLoopBoxes
#define GrGeomPatchLoopBoxes(i, j, k, fdir, grgeom, patch_num,          \
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

#undef GrGeomInLoopBoxes
#define GrGeomInLoopBoxes(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, body) \
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


#undef GrGeomSurfLoopBoxes
#define GrGeomSurfLoopBoxes(i, j, k, fdir, grgeom, ix, iy, iz, nx, ny, nz, body) \
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

#endif // ACC_BACKEND != BACKEND_OMP
#endif // _PF_OMPLOOPS_H_
