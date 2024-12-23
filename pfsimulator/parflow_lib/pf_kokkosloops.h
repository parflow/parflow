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

/* @file
 * @brief Contains macros, functions, and structs for Kokkos compute kernels.
 */

#ifndef PF_KOKKOSLOOPS_H
#define PF_KOKKOSLOOPS_H

/*--------------------------------------------------------------------------
 * Include headers
 *--------------------------------------------------------------------------*/
#include "pf_devices.h"
#include "pf_kokkosmalloc.h"

extern "C++" {
#include <tuple>
#include <Kokkos_Core.hpp>

/*--------------------------------------------------------------------------
 * Helper macros and functions
 *--------------------------------------------------------------------------*/
#define RAND48_SEED_0   (0x330e)
#define RAND48_SEED_1   (0xabcd)
#define RAND48_SEED_2   (0x1234)
#define RAND48_MULT_0   (0xe66d)
#define RAND48_MULT_1   (0xdeec)
#define RAND48_MULT_2   (0x0005)
#define RAND48_ADD      (0x000b)

/** Helper struct for type comparison. @note Not for direct use! */
template < typename T >
struct function_traits
  : public function_traits < decltype(&T::operator()) >
{};
// For generic types, directly use the result of the signature of its 'operator()'

/** Helper struct for type comparison. @note Not for direct use! */
template < typename ClassType, typename ReturnType, typename ... Args >
struct function_traits < ReturnType (ClassType::*)(Args...) const >
// we specialize for pointers to member function
{
  enum { arity = sizeof...(Args) };
  // arity is the number of arguments.

  typedef ReturnType result_type;

  template < size_t i >
  struct arg {
    typedef typename std::tuple_element < i, std::tuple < Args... > > ::type type;
    // the i-th argument is equivalent to the i-th tuple element of a tuple
    // composed of those arguments.
  };
};

/** Device-callable dorand48() function for Kokkos compute kernels. */
KOKKOS_FORCEINLINE_FUNCTION static void dev_dorand48(unsigned short xseed[3])
{
  unsigned long accu;

  unsigned short _rand48_mult[3] = {
    RAND48_MULT_0,
    RAND48_MULT_1,
    RAND48_MULT_2
  };
  unsigned short _rand48_add = RAND48_ADD;
  unsigned short temp[2];

  accu = (unsigned long)_rand48_mult[0] * (unsigned long)xseed[0] +
         (unsigned long)_rand48_add;
  temp[0] = (unsigned short)accu;               /* lower 16 bits */
  accu >>= sizeof(unsigned short) * 8;
  accu += (unsigned long)_rand48_mult[0] * (unsigned long)xseed[1] +
          (unsigned long)_rand48_mult[1] * (unsigned long)xseed[0];
  temp[1] = (unsigned short)accu;               /* middle 16 bits */
  accu >>= sizeof(unsigned short) * 8;
  accu += _rand48_mult[0] * xseed[2] + _rand48_mult[1] * xseed[1] + _rand48_mult[2] * xseed[0];
  xseed[0] = temp[0];
  xseed[1] = temp[1];
  xseed[2] = (unsigned short)accu;
}

/** Device-callable erand48() function for Kokkos compute kernels. */
KOKKOS_FORCEINLINE_FUNCTION static double dev_erand48(unsigned short xseed[3])
{
  dev_dorand48(xseed);
  return ldexp((double)xseed[0], -48) +
         ldexp((double)xseed[1], -32) +
         ldexp((double)xseed[2], -16);
}

/** Device-callable drand48() function for Kokkos compute kernels. */
KOKKOS_FORCEINLINE_FUNCTION static double dev_drand48(void)
{
  unsigned short _rand48_seed[3] = {
    RAND48_SEED_0,
    RAND48_SEED_1,
    RAND48_SEED_2
  };

  return dev_erand48(_rand48_seed);
}

/** Device-callable RPowerR() function for Kokkos compute kernels. */
template < typename T >
KOKKOS_FORCEINLINE_FUNCTION static T RPowerR(T base, T exponent)
{
  if (base <= 0.0)
    return(0.0);

  return((T)pow((double)base, (double)exponent));
}

/** Helper struct for type comparison. @note Not for direct use! */
struct SkipParallelSync { const int dummy = 0; };
#define SKIP_PARALLEL_SYNC_kokkos struct SkipParallelSync sync_struct; return sync_struct;

#define PARALLEL_SYNC_kokkos Kokkos::fence();

#define PlusEquals_kokkos(a, b) Kokkos::atomic_add(&(a), b)

/** Helper struct for type comparison. @note Not for direct use! */
template < typename T >
struct ReduceMaxType { T value; };
#define ReduceMax_kokkos(a, b) if (lsum < b) { lsum = b; } struct ReduceMaxType < std::decay < decltype(a) > ::type > reduce_struct; return reduce_struct;

/** Helper struct for type comparison. @note Not for direct use! */
template < typename T >
struct ReduceMinType { T value; };
#define ReduceMin_kokkos(a, b) if (lsum > b) { lsum = b; } struct ReduceMinType < std::decay < decltype(a) > ::type > reduce_struct; return reduce_struct;

/** Helper struct for type comparison. @note Not for direct use! */
template < typename T >
struct ReduceSumType { T value; };
#define ReduceSum_kokkos(a, b) lsum += b; struct ReduceSumType < std::decay < decltype(a) > ::type > reduce_struct; return reduce_struct;


/** A constant table for fdir (borrowed from OMP backend) */
static const int FDIR_TABLE[6][3] = {
  { -1, 0, 0 }, // Left
  { 1, 0, 0 }, // Right
  { 0, -1, 0 }, // Down
  { 0, 1, 0 }, // Up
  { 0, 0, -1 }, // Back
  { 0, 0, 1 }, // Front
};

/*--------------------------------------------------------------------------
 * Kokkos loop macro redefinitions
 *--------------------------------------------------------------------------*/

/**
 * @brief A macro for checking if cell flag data array must be reallocated.
 *
 * @note Not for direct use!
 *
 * @param grgeom The geometry details [IN/OUT]
 * @param nx The size of the first dim [IN]
 * @param ny The size of the second dim [IN]
 * @param nz The size of the third dim [IN]
 */
#define CheckCellFlagAllocation(grgeom, nx, ny, nz)                                            \
        {                                                                                      \
          int flagdata_size = sizeof(char) * (nz * ny * nx);                                   \
          if (GrGeomSolidCellFlagDataSize(grgeom) < flagdata_size)                             \
          {                                                                                    \
            char *flagdata = (char*)_ctalloc_device(flagdata_size);                            \
                                                                                               \
            if (GrGeomSolidCellFlagDataSize(grgeom) > 0)                                       \
            memcpy(flagdata, GrGeomSolidCellFlagData(grgeom),                                  \
                   GrGeomSolidCellFlagDataSize(grgeom));                                       \
                                                                                               \
            _tfree_device(GrGeomSolidCellFlagData(grgeom));                                    \
            GrGeomSolidCellFlagData(grgeom) = flagdata;                                        \
            GrGeomSolidCellFlagDataSize(grgeom) = flagdata_size;                               \
          }                                                                                    \
        }

/** Loop definition for Kokkos. */
#define BoxLoopI1_kokkos(i, j, k,                                                                           \
                         ix, iy, iz, nx, ny, nz,                                                            \
                         i1, nx1, ny1, nz1, sx1, sy1, sz1,                                                  \
                         loop_body)                                                                         \
        {                                                                                                   \
          if (nx > 0 && ny > 0 && nz > 0)                                                                   \
          {                                                                                                 \
            DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);                     \
                                                                                                            \
            const auto &ref_i1 = i1;                                                                        \
                                                                                                            \
            auto lambda_body =                                                                              \
              KOKKOS_LAMBDA(int i, int j, int k)                                                            \
            {                                                                                               \
              const int i1 = k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                                       \
                             + (k * ny * nx + j * nx + i) * sx1 + ref_i1;                                   \
                                                                                                            \
              i += ix;                                                                                      \
              j += iy;                                                                                      \
              k += iz;                                                                                      \
                                                                                                            \
              loop_body;                                                                                    \
            };                                                                                              \
                                                                                                            \
            using MDPolicyType_3D = typename Kokkos::MDRangePolicy < Kokkos::Rank < 3 > >;                  \
            MDPolicyType_3D mdpolicy_3d({ { 0, 0, 0 } }, { { nx, ny, nz } });                               \
            Kokkos::parallel_for(mdpolicy_3d, lambda_body);                                                 \
                                                                                                            \
            typedef function_traits < decltype(lambda_body) > traits;                                       \
            if (!std::is_same < traits::result_type, struct SkipParallelSync > ::value)                     \
            Kokkos::fence();                                                                                \
          }                                                                                                 \
          (void)i; (void)j; (void)k;                                                                        \
        }

/** Loop definition for Kokkos. */
#define BoxLoopI2_kokkos(i, j, k,                                                                           \
                         ix, iy, iz, nx, ny, nz,                                                            \
                         i1, nx1, ny1, nz1, sx1, sy1, sz1,                                                  \
                         i2, nx2, ny2, nz2, sx2, sy2, sz2,                                                  \
                         loop_body)                                                                         \
        {                                                                                                   \
          if (nx > 0 && ny > 0 && nz > 0)                                                                   \
          {                                                                                                 \
            DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);                     \
            DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2);                     \
                                                                                                            \
            const auto &ref_i1 = i1;                                                                        \
            const auto &ref_i2 = i2;                                                                        \
                                                                                                            \
            auto lambda_body =                                                                              \
              KOKKOS_LAMBDA(int i, int j, int k)                                                            \
            {                                                                                               \
              const int i1 = k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                                       \
                             + (k * ny * nx + j * nx + i) * sx1 + ref_i1;                                   \
              const int i2 = k * PV_kinc_2 + (k * ny + j) * PV_jinc_2                                       \
                             + (k * ny * nx + j * nx + i) * sx2 + ref_i2;                                   \
                                                                                                            \
              i += ix;                                                                                      \
              j += iy;                                                                                      \
              k += iz;                                                                                      \
                                                                                                            \
              loop_body;                                                                                    \
            };                                                                                              \
                                                                                                            \
            using MDPolicyType_3D = typename Kokkos::MDRangePolicy < Kokkos::Rank < 3 > >;                  \
            MDPolicyType_3D mdpolicy_3d({ { 0, 0, 0 } }, { { nx, ny, nz } });                               \
            Kokkos::parallel_for(mdpolicy_3d, lambda_body);                                                 \
                                                                                                            \
            typedef function_traits < decltype(lambda_body) > traits;                                       \
            if (!std::is_same < traits::result_type, struct SkipParallelSync > ::value)                     \
            Kokkos::fence();                                                                                \
          }                                                                                                 \
          (void)i; (void)j; (void)k;                                                                        \
        }

/** Loop definition for Kokkos. */
#define BoxLoopI3_kokkos(i, j, k,                                                                           \
                         ix, iy, iz, nx, ny, nz,                                                            \
                         i1, nx1, ny1, nz1, sx1, sy1, sz1,                                                  \
                         i2, nx2, ny2, nz2, sx2, sy2, sz2,                                                  \
                         i3, nx3, ny3, nz3, sx3, sy3, sz3,                                                  \
                         loop_body)                                                                         \
        {                                                                                                   \
          if (nx > 0 && ny > 0 && nz > 0)                                                                   \
          {                                                                                                 \
            DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);                     \
            DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2);                     \
            DeclareInc(PV_jinc_3, PV_kinc_3, nx, ny, nz, nx3, ny3, nz3, sx3, sy3, sz3);                     \
                                                                                                            \
            const auto &ref_i1 = i1;                                                                        \
            const auto &ref_i2 = i2;                                                                        \
            const auto &ref_i3 = i3;                                                                        \
                                                                                                            \
            auto lambda_body =                                                                              \
              KOKKOS_LAMBDA(int i, int j, int k)                                                            \
            {                                                                                               \
              const int i1 = k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                                       \
                             + (k * ny * nx + j * nx + i) * sx1 + ref_i1;                                   \
              const int i2 = k * PV_kinc_2 + (k * ny + j) * PV_jinc_2                                       \
                             + (k * ny * nx + j * nx + i) * sx2 + ref_i2;                                   \
              const int i3 = k * PV_kinc_3 + (k * ny + j) * PV_jinc_3                                       \
                             + (k * ny * nx + j * nx + i) * sx3 + ref_i3;                                   \
                                                                                                            \
              i += ix;                                                                                      \
              j += iy;                                                                                      \
              k += iz;                                                                                      \
                                                                                                            \
              loop_body;                                                                                    \
            };                                                                                              \
                                                                                                            \
            using MDPolicyType_3D = typename Kokkos::MDRangePolicy < Kokkos::Rank < 3 > >;                  \
            MDPolicyType_3D mdpolicy_3d({ { 0, 0, 0 } }, { { nx, ny, nz } });                               \
            Kokkos::parallel_for(mdpolicy_3d, lambda_body);                                                 \
                                                                                                            \
            typedef function_traits < decltype(lambda_body) > traits;                                       \
            if (!std::is_same < traits::result_type, struct SkipParallelSync > ::value)                     \
            Kokkos::fence();                                                                                \
          }                                                                                                 \
          (void)i; (void)j; (void)k;                                                                        \
        }

/** Loop definition for Kokkos. */
#define BoxLoopReduceI1_kokkos(rslt, i, j, k,                                                                     \
                               ix, iy, iz, nx, ny, nz,                                                            \
                               i1, nx1, ny1, nz1, sx1, sy1, sz1,                                                  \
                               loop_body)                                                                         \
        {                                                                                                         \
          if (nx > 0 && ny > 0 && nz > 0)                                                                         \
          {                                                                                                       \
            DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);                           \
                                                                                                                  \
            const auto &ref_rslt = rslt;                                                                          \
            const auto &ref_i1 = i1;                                                                              \
                                                                                                                  \
            auto lambda_body =                                                                                    \
              KOKKOS_LAMBDA(int i, int j, int k, double& lsum)                                                    \
            {                                                                                                     \
              const int i1 = k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                                             \
                             + (k * ny * nx + j * nx + i) * sx1 + ref_i1;                                         \
                                                                                                                  \
              i += ix;                                                                                            \
              j += iy;                                                                                            \
              k += iz;                                                                                            \
                                                                                                                  \
              loop_body;                                                                                          \
            };                                                                                                    \
                                                                                                                  \
            auto lambda_outer =                                                                                   \
              KOKKOS_LAMBDA(int i, int j, int k, double& lsum)                                                    \
            {                                                                                                     \
              lambda_body(i, j, k, lsum);                                                                         \
            };                                                                                                    \
                                                                                                                  \
            using MDPolicyType_3D = typename Kokkos::MDRangePolicy < Kokkos::Rank < 3 > >;                        \
            MDPolicyType_3D mdpolicy_3d({ { 0, 0, 0 } }, { { nx, ny, nz } });                                     \
            typedef function_traits < decltype(lambda_body) > traits;                                             \
            if (std::is_same < traits::result_type, struct ReduceSumType < double > > ::value)                    \
            {                                                                                                     \
              Kokkos::parallel_reduce(mdpolicy_3d, lambda_outer, rslt);                                           \
            }                                                                                                     \
            else if (std::is_same < traits::result_type, struct ReduceMaxType < double > > ::value)               \
            {                                                                                                     \
              Kokkos::parallel_reduce(mdpolicy_3d, lambda_outer, Kokkos::Max < double > (rslt));                  \
            }                                                                                                     \
            else if (std::is_same < traits::result_type, struct ReduceMinType < double > > ::value)               \
            {                                                                                                     \
              Kokkos::parallel_reduce(mdpolicy_3d, lambda_outer, Kokkos::Min < double > (rslt));                  \
            }                                                                                                     \
            else                                                                                                  \
            {                                                                                                     \
              printf("ERROR at %s:%d: Invalid reduction identifier,                         \
      likely a problem with a BoxLoopReduce body.", __FILE__, __LINE__);                                          \
            }                                                                                                     \
            Kokkos::fence();                                                                                      \
          }                                                                                                       \
          (void)i; (void)j; (void)k;                                                                              \
        }

/** Loop definition for Kokkos. */
#define BoxLoopReduceI2_kokkos(rslt, i, j, k,                                                                     \
                               ix, iy, iz, nx, ny, nz,                                                            \
                               i1, nx1, ny1, nz1, sx1, sy1, sz1,                                                  \
                               i2, nx2, ny2, nz2, sx2, sy2, sz2,                                                  \
                               loop_body)                                                                         \
        {                                                                                                         \
          if (nx > 0 && ny > 0 && nz > 0)                                                                         \
          {                                                                                                       \
            DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);                           \
            DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2);                           \
                                                                                                                  \
            const auto &ref_rslt = rslt;                                                                          \
            const auto &ref_i1 = i1;                                                                              \
            const auto &ref_i2 = i2;                                                                              \
                                                                                                                  \
            auto lambda_body =                                                                                    \
              KOKKOS_LAMBDA(int i, int j, int k, double& lsum)                                                    \
            {                                                                                                     \
              const int i1 = k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                                             \
                             + (k * ny * nx + j * nx + i) * sx1 + ref_i1;                                         \
              const int i2 = k * PV_kinc_2 + (k * ny + j) * PV_jinc_2                                             \
                             + (k * ny * nx + j * nx + i) * sx2 + ref_i2;                                         \
                                                                                                                  \
              i += ix;                                                                                            \
              j += iy;                                                                                            \
              k += iz;                                                                                            \
                                                                                                                  \
              loop_body;                                                                                          \
            };                                                                                                    \
                                                                                                                  \
            auto lambda_outer =                                                                                   \
              KOKKOS_LAMBDA(int i, int j, int k, double& lsum)                                                    \
            {                                                                                                     \
              lambda_body(i, j, k, lsum);                                                                         \
            };                                                                                                    \
                                                                                                                  \
            using MDPolicyType_3D = typename Kokkos::MDRangePolicy < Kokkos::Rank < 3 > >;                        \
            MDPolicyType_3D mdpolicy_3d({ { 0, 0, 0 } }, { { nx, ny, nz } });                                     \
            typedef function_traits < decltype(lambda_body) > traits;                                             \
            if (std::is_same < traits::result_type, struct ReduceSumType < double > > ::value)                    \
            {                                                                                                     \
              Kokkos::parallel_reduce(mdpolicy_3d, lambda_outer, rslt);                                           \
            }                                                                                                     \
            else if (std::is_same < traits::result_type, struct ReduceMaxType < double > > ::value)               \
            {                                                                                                     \
              Kokkos::parallel_reduce(mdpolicy_3d, lambda_outer, Kokkos::Max < double > (rslt));                  \
            }                                                                                                     \
            else if (std::is_same < traits::result_type, struct ReduceMinType < double > > ::value)               \
            {                                                                                                     \
              Kokkos::parallel_reduce(mdpolicy_3d, lambda_outer, Kokkos::Min < double > (rslt));                  \
            }                                                                                                     \
            else                                                                                                  \
            {                                                                                                     \
              printf("ERROR at %s:%d: Invalid reduction identifier,                         \
      likely a problem with a BoxLoopReduce body.", __FILE__, __LINE__);                                          \
            }                                                                                                     \
            Kokkos::fence();                                                                                      \
          }                                                                                                       \
          (void)i; (void)j; (void)k;                                                                              \
        }

/** Loop definition for Kokkos. */
#define GrGeomInLoopBoxes_kokkos(i, j, k,                                                                           \
                                 grgeom, ix, iy, iz, nx, ny, nz, loop_body)                                         \
        {                                                                                                           \
          BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                                                       \
          int ix_bxs = BoxArrayMinCell(boxes, 0);                                                                   \
          int iy_bxs = BoxArrayMinCell(boxes, 1);                                                                   \
          int iz_bxs = BoxArrayMinCell(boxes, 2);                                                                   \
                                                                                                                    \
          int nx_bxs = BoxArrayMaxCell(boxes, 0) - ix_bxs + 1;                                                      \
          int ny_bxs = BoxArrayMaxCell(boxes, 1) - iy_bxs + 1;                                                      \
          int nz_bxs = BoxArrayMaxCell(boxes, 2) - iz_bxs + 1;                                                      \
                                                                                                                    \
          if (!(GrGeomSolidCellFlagInitialized(grgeom) & 1))                                                        \
          {                                                                                                         \
            CheckCellFlagAllocation(grgeom, nx_bxs, ny_bxs, nz_bxs);                                                \
            char *inflag = GrGeomSolidCellFlagData(grgeom);                                                         \
                                                                                                                    \
            for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                                            \
            {                                                                                                       \
              Box box = BoxArrayGetBox(boxes, PV_box);                                                              \
              int PV_ixl = box.lo[0];                                                                               \
              int PV_iyl = box.lo[1];                                                                               \
              int PV_izl = box.lo[2];                                                                               \
              int PV_ixu = box.up[0];                                                                               \
              int PV_iyu = box.up[1];                                                                               \
              int PV_izu = box.up[2];                                                                               \
                                                                                                                    \
              if (PV_ixl <= PV_ixu && PV_iyl <= PV_iyu && PV_izl <= PV_izu)                                         \
              {                                                                                                     \
                int PV_nx = PV_ixu - PV_ixl + 1;                                                                    \
                int PV_ny = PV_iyu - PV_iyl + 1;                                                                    \
                int PV_nz = PV_izu - PV_izl + 1;                                                                    \
                                                                                                                    \
                Globals *globals = ::globals;                                                                       \
                auto lambda_body =                                                                                  \
                  KOKKOS_LAMBDA(int i, int j, int k)                                                                \
                {                                                                                                   \
                  i += PV_ixl;                                                                                      \
                  j += PV_iyl;                                                                                      \
                  k += PV_izl;                                                                                      \
                                                                                                                    \
                  /* Set inflag for all cells in boxes regardless of loop limits */                                 \
                  inflag[(k - iz_bxs) * ny_bxs * nx_bxs +                                                           \
                         (j - iy_bxs) * nx_bxs + (i - ix_bxs)] |= 1;                                                \
                                                                                                                    \
                  /* Only evaluate loop body if the cell is within loop limits */                                   \
                  if (i >= ix && j >= iy && k >= iz &&                                                              \
                      i < ix + nx && j < iy + ny && k < iz + nz)                                                    \
                  {                                                                                                 \
                    loop_body;                                                                                      \
                  }                                                                                                 \
                };                                                                                                  \
                                                                                                                    \
                using MDPolicyType_3D = typename Kokkos::MDRangePolicy < Kokkos::Rank < 3 > >;                      \
                MDPolicyType_3D mdpolicy_3d({ { 0, 0, 0 } }, { { PV_nx, PV_ny, PV_nz } });                          \
                Kokkos::parallel_for(mdpolicy_3d, lambda_body);                                                     \
                Kokkos::fence();                                                                                    \
              }                                                                                                     \
            }                                                                                                       \
            GrGeomSolidCellFlagInitialized(grgeom) |= 1;                                                            \
          }                                                                                                         \
          else                                                                                                      \
          {                                                                                                         \
            int ixl_gpu = pfmax(ix, ix_bxs);                                                                        \
            int iyl_gpu = pfmax(iy, iy_bxs);                                                                        \
            int izl_gpu = pfmax(iz, iz_bxs);                                                                        \
            int nx_gpu = pfmin((ix + nx - 1), BoxArrayMaxCell(boxes, 0)) - ixl_gpu + 1;                             \
            int ny_gpu = pfmin((iy + ny - 1), BoxArrayMaxCell(boxes, 1)) - iyl_gpu + 1;                             \
            int nz_gpu = pfmin((iz + nz - 1), BoxArrayMaxCell(boxes, 2)) - izl_gpu + 1;                             \
                                                                                                                    \
            if (nx_gpu > 0 && ny_gpu > 0 && nz_gpu > 0)                                                             \
            {                                                                                                       \
              Globals *globals = ::globals;                                                                         \
              char *inflag = GrGeomSolidCellFlagData(grgeom);                                                       \
              auto lambda_body =                                                                                    \
                KOKKOS_LAMBDA(int i, int j, int k)                                                                  \
              {                                                                                                     \
                i += ixl_gpu;                                                                                       \
                j += iyl_gpu;                                                                                       \
                k += izl_gpu;                                                                                       \
                if (inflag[(k - iz_bxs) * ny_bxs * nx_bxs +                                                         \
                           (j - iy_bxs) * nx_bxs + (i - ix_bxs)] & 1)                                               \
                {                                                                                                   \
                  loop_body;                                                                                        \
                }                                                                                                   \
              };                                                                                                    \
                                                                                                                    \
              using MDPolicyType_3D = typename Kokkos::MDRangePolicy < Kokkos::Rank < 3 > >;                        \
              MDPolicyType_3D mdpolicy_3d({ { 0, 0, 0 } }, { { nx_gpu, ny_gpu, nz_gpu } });                         \
              Kokkos::parallel_for(mdpolicy_3d, lambda_body);                                                       \
              Kokkos::fence();                                                                                      \
            }                                                                                                       \
          }                                                                                                         \
          (void)i; (void)j; (void)k;                                                                                \
        }

/** Loop definition for Kokkos. */
#define GrGeomSurfLoopBoxes_kokkos(i, j, k, fdir, grgeom,                                                             \
                                   ix, iy, iz, nx, ny, nz, loop_body)                                                 \
        {                                                                                                             \
          for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)                                                     \
          {                                                                                                           \
            const int *fdir = FDIR_TABLE[PV_f];                                                                       \
                                                                                                                      \
            BoxArray* boxes = GrGeomSolidSurfaceBoxes(grgeom, PV_f);                                                  \
            int ix_bxs = BoxArrayMinCell(boxes, 0);                                                                   \
            int iy_bxs = BoxArrayMinCell(boxes, 1);                                                                   \
            int iz_bxs = BoxArrayMinCell(boxes, 2);                                                                   \
                                                                                                                      \
            int nx_bxs = BoxArrayMaxCell(boxes, 0) - ix_bxs + 1;                                                      \
            int ny_bxs = BoxArrayMaxCell(boxes, 1) - iy_bxs + 1;                                                      \
            int nz_bxs = BoxArrayMaxCell(boxes, 2) - iz_bxs + 1;                                                      \
                                                                                                                      \
            if (!(GrGeomSolidCellFlagInitialized(grgeom) & (1 << (2 + PV_f))))                                        \
            {                                                                                                         \
              CheckCellFlagAllocation(grgeom, nx_bxs, ny_bxs, nz_bxs);                                                \
              char *surfflag = GrGeomSolidCellFlagData(grgeom);                                                       \
                                                                                                                      \
              for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                                            \
              {                                                                                                       \
                Box box = BoxArrayGetBox(boxes, PV_box);                                                              \
                int PV_ixl = box.lo[0];                                                                               \
                int PV_iyl = box.lo[1];                                                                               \
                int PV_izl = box.lo[2];                                                                               \
                int PV_ixu = box.up[0];                                                                               \
                int PV_iyu = box.up[1];                                                                               \
                int PV_izu = box.up[2];                                                                               \
                                                                                                                      \
                if (PV_ixl <= PV_ixu && PV_iyl <= PV_iyu && PV_izl <= PV_izu)                                         \
                {                                                                                                     \
                  int PV_nx = PV_ixu - PV_ixl + 1;                                                                    \
                  int PV_ny = PV_iyu - PV_iyl + 1;                                                                    \
                  int PV_nz = PV_izu - PV_izl + 1;                                                                    \
                                                                                                                      \
                  const int _fdir0 = fdir[0];                                                                         \
                  const int _fdir1 = fdir[1];                                                                         \
                  const int _fdir2 = fdir[2];                                                                         \
                                                                                                                      \
                  auto lambda_body =                                                                                  \
                    KOKKOS_LAMBDA(int i, int j, int k)                                                                \
                  {                                                                                                   \
                    i += PV_ixl;                                                                                      \
                    j += PV_iyl;                                                                                      \
                    k += PV_izl;                                                                                      \
                                                                                                                      \
                    /* Set surfflag for all cells in boxes regardless of loop limits */                               \
                    surfflag[(k - iz_bxs) * ny_bxs * nx_bxs +                                                         \
                             (j - iy_bxs) * nx_bxs + (i - ix_bxs)] |= (1 << (2 + PV_f));                              \
                                                                                                                      \
                    /* Only evaluate loop body if the cell is within loop limits */                                   \
                    if (i >= ix && j >= iy && k >= iz &&                                                              \
                        i < ix + nx && j < iy + ny && k < iz + nz)                                                    \
                    {                                                                                                 \
                      const int fdir[3] = { _fdir0, _fdir1, _fdir2 };                                                 \
                      loop_body;                                                                                      \
                      (void)fdir;                                                                                     \
                    }                                                                                                 \
                  };                                                                                                  \
                                                                                                                      \
                  using MDPolicyType_3D = typename Kokkos::MDRangePolicy < Kokkos::Rank < 3 > >;                      \
                  MDPolicyType_3D mdpolicy_3d({ { 0, 0, 0 } }, { { PV_nx, PV_ny, PV_nz } });                          \
                  Kokkos::parallel_for(mdpolicy_3d, lambda_body);                                                     \
                  Kokkos::fence();                                                                                    \
                }                                                                                                     \
              }                                                                                                       \
              GrGeomSolidCellFlagInitialized(grgeom) |= (1 << (2 + PV_f));                                            \
            }                                                                                                         \
            else                                                                                                      \
            {                                                                                                         \
              int ixl_gpu = pfmax(ix, ix_bxs);                                                                        \
              int iyl_gpu = pfmax(iy, iy_bxs);                                                                        \
              int izl_gpu = pfmax(iz, iz_bxs);                                                                        \
              int nx_gpu = pfmin((ix + nx - 1), BoxArrayMaxCell(boxes, 0)) - ixl_gpu + 1;                             \
              int ny_gpu = pfmin((iy + ny - 1), BoxArrayMaxCell(boxes, 1)) - iyl_gpu + 1;                             \
              int nz_gpu = pfmin((iz + nz - 1), BoxArrayMaxCell(boxes, 2)) - izl_gpu + 1;                             \
                                                                                                                      \
              if (nx_gpu > 0 && ny_gpu > 0 && nz_gpu > 0)                                                             \
              {                                                                                                       \
                char *surfflag = GrGeomSolidCellFlagData(grgeom);                                                     \
                                                                                                                      \
                const int _fdir0 = fdir[0];                                                                           \
                const int _fdir1 = fdir[1];                                                                           \
                const int _fdir2 = fdir[2];                                                                           \
                                                                                                                      \
                auto lambda_body =                                                                                    \
                  KOKKOS_LAMBDA(int i, int j, int k)                                                                  \
                {                                                                                                     \
                  i += ixl_gpu;                                                                                       \
                  j += iyl_gpu;                                                                                       \
                  k += izl_gpu;                                                                                       \
                                                                                                                      \
                  if (surfflag[(k - iz_bxs) * ny_bxs * nx_bxs +                                                       \
                               (j - iy_bxs) * nx_bxs + (i - ix_bxs)] & (1 << (2 + PV_f)))                             \
                  {                                                                                                   \
                    const int fdir[3] = { _fdir0, _fdir1, _fdir2 };                                                   \
                    loop_body;                                                                                        \
                    (void)fdir;                                                                                       \
                  }                                                                                                   \
                };                                                                                                    \
                                                                                                                      \
                using MDPolicyType_3D = typename Kokkos::MDRangePolicy < Kokkos::Rank < 3 > >;                        \
                MDPolicyType_3D mdpolicy_3d({ { 0, 0, 0 } }, { { nx_gpu, ny_gpu, nz_gpu } });                         \
                Kokkos::parallel_for(mdpolicy_3d, lambda_body);                                                       \
                Kokkos::fence();                                                                                      \
              }                                                                                                       \
            }                                                                                                         \
          }                                                                                                           \
          (void)i; (void)j; (void)k;                                                                                  \
        }

/** Loop definition for Kokkos. */
#define GrGeomPatchLoopBoxes_kokkos(i, j, k, fdir, grgeom, patch_num,                                                  \
                                    ix, iy, iz, nx, ny, nz, loop_body)                                                 \
        {                                                                                                              \
          for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)                                                      \
          {                                                                                                            \
            const int *fdir = FDIR_TABLE[PV_f];                                                                        \
                                                                                                                       \
            int n_prev = 0;                                                                                            \
            BoxArray* boxes = GrGeomSolidPatchBoxes(grgeom, patch_num, PV_f);                                          \
            for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                                               \
            {                                                                                                          \
              Box box = BoxArrayGetBox(boxes, PV_box);                                                                 \
              /* find octree and region intersection */                                                                \
              int PV_ixl = pfmax(ix, box.lo[0]);                                                                       \
              int PV_iyl = pfmax(iy, box.lo[1]);                                                                       \
              int PV_izl = pfmax(iz, box.lo[2]);                                                                       \
              int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                                                            \
              int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                                                            \
              int PV_izu = pfmin((iz + nz - 1), box.up[2]);                                                            \
                                                                                                                       \
              if (PV_ixl <= PV_ixu && PV_iyl <= PV_iyu && PV_izl <= PV_izu)                                            \
              {                                                                                                        \
                int PV_diff_x = PV_ixu - PV_ixl;                                                                       \
                int PV_diff_y = PV_iyu - PV_iyl;                                                                       \
                int PV_diff_z = PV_izu - PV_izl;                                                                       \
                                                                                                                       \
                int nx = PV_diff_x + 1;                                                                                \
                int ny = PV_diff_y + 1;                                                                                \
                int nz = PV_diff_z + 1;                                                                                \
                                                                                                                       \
                const int _fdir0 = fdir[0];                                                                            \
                const int _fdir1 = fdir[1];                                                                            \
                const int _fdir2 = fdir[2];                                                                            \
                                                                                                                       \
                auto lambda_body =                                                                                     \
                  KOKKOS_LAMBDA(int i, int j, int k)                                                                   \
                {                                                                                                      \
                  const int fdir[3] = { _fdir0, _fdir1, _fdir2 };                                                      \
                  int ival = n_prev + k * ny * nx + j * nx + i;                                                        \
                                                                                                                       \
                  i += PV_ixl;                                                                                         \
                  j += PV_iyl;                                                                                         \
                  k += PV_izl;                                                                                         \
                                                                                                                       \
                  loop_body;                                                                                           \
                  (void)fdir;                                                                                          \
                };                                                                                                     \
                n_prev += nz * ny * nx;                                                                                \
                                                                                                                       \
                using MDPolicyType_3D = typename Kokkos::MDRangePolicy < Kokkos::Rank < 3 > >;                         \
                MDPolicyType_3D mdpolicy_3d({ { 0, 0, 0 } }, { { nx, ny, nz } });                                      \
                Kokkos::parallel_for(mdpolicy_3d, lambda_body);                                                        \
                Kokkos::fence();                                                                                       \
              }                                                                                                        \
            }                                                                                                          \
          }                                                                                                            \
          (void)i; (void)j; (void)k;                                                                                   \
        }

/** Loop definition for Kokkos. */
#define GrGeomPatchLoopBoxesNoFdir_kokkos(i, j, k, grgeom, patch_num, ovrlnd,                                                \
                                          ix, iy, iz, nx, ny, nz, locals, setup,                                             \
                                          f_left, f_right, f_down, f_up, f_back, f_front, finalize)                          \
        {                                                                                                                    \
          int n_ival = 0;                                                                                                    \
          for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)                                                            \
          {                                                                                                                  \
            BoxArray* boxes = GrGeomSolidPatchBoxes(grgeom, patch_num, PV_f);                                                \
                                                                                                                             \
            int ix_bxs = BoxArrayMinCell(boxes, 0);                                                                          \
            int iy_bxs = BoxArrayMinCell(boxes, 1);                                                                          \
            int iz_bxs = BoxArrayMinCell(boxes, 2);                                                                          \
                                                                                                                             \
            int nx_bxs = BoxArrayMaxCell(boxes, 0) - ix_bxs + 1;                                                             \
            int ny_bxs = BoxArrayMaxCell(boxes, 1) - iy_bxs + 1;                                                             \
            int nz_bxs = BoxArrayMaxCell(boxes, 2) - iz_bxs + 1;                                                             \
                                                                                                                             \
            int patch_loc;                                                                                                   \
            if (ovrlnd)                                                                                                      \
            patch_loc = GrGeomSolidNumPatches(grgeom) +patch_num;                                                            \
            else                                                                                                             \
            patch_loc = patch_num;                                                                                           \
                                                                                                                             \
            int *ptr_ival = GrGeomSolidCellIval(grgeom, patch_loc, PV_f);                                                    \
            if (!(ptr_ival))                                                                                                 \
            {                                                                                                                \
              GrGeomSolidCellIval(grgeom, patch_loc, PV_f) =                                                                 \
                (int*)_talloc_device(sizeof(int) * nx_bxs * ny_bxs * nz_bxs);                                                \
                                                                                                                             \
              ptr_ival = GrGeomSolidCellIval(grgeom, patch_loc, PV_f);                                                       \
              for (int idx = 0; idx < nx_bxs * ny_bxs * nz_bxs; idx++)                                                       \
              ptr_ival[idx] = -1;                                                                                            \
                                                                                                                             \
              for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                                                   \
              {                                                                                                              \
                Box box = BoxArrayGetBox(boxes, PV_box);                                                                     \
                int PV_ixl = pfmax(ix, box.lo[0]);                                                                           \
                int PV_iyl = pfmax(iy, box.lo[1]);                                                                           \
                int PV_izl = pfmax(iz, box.lo[2]);                                                                           \
                int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                                                                \
                int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                                                                \
                int PV_izu = pfmin((iz + nz - 1), box.up[2]);                                                                \
                                                                                                                             \
                for (k = PV_izl; k <= PV_izu; k++)                                                                           \
                for (j = PV_iyl; j <= PV_iyu; j++)                                                                           \
                for (i = PV_ixl; i <= PV_ixu; i++)                                                                           \
                {                                                                                                            \
                  UNPACK(locals);                                                                                            \
                  setup;                                                                                                     \
                  switch (PV_f)                                                                                              \
                  {                                                                                                          \
                  f_left;                                                                                                    \
                  f_right;                                                                                                   \
                  f_down;                                                                                                    \
                  f_up;                                                                                                      \
                  f_back;                                                                                                    \
                  f_front;                                                                                                   \
                  }                                                                                                          \
                  finalize;                                                                                                  \
                  ptr_ival[(k - iz_bxs) * ny_bxs * nx_bxs + (j - iy_bxs) *                                                   \
                           nx_bxs + (i - ix_bxs)] = n_ival++;                                                                \
                }                                                                                                            \
              }                                                                                                              \
            }                                                                                                                \
            else                                                                                                             \
            {                                                                                                                \
              int ixl_gpu = pfmax(ix, ix_bxs);                                                                               \
              int iyl_gpu = pfmax(iy, iy_bxs);                                                                               \
              int izl_gpu = pfmax(iz, iz_bxs);                                                                               \
              int nx_gpu = pfmin((ix + nx - 1), BoxArrayMaxCell(boxes, 0)) - ixl_gpu + 1;                                    \
              int ny_gpu = pfmin((iy + ny - 1), BoxArrayMaxCell(boxes, 1)) - iyl_gpu + 1;                                    \
              int nz_gpu = pfmin((iz + nz - 1), BoxArrayMaxCell(boxes, 2)) - izl_gpu + 1;                                    \
                                                                                                                             \
              if (nx_gpu > 0 && ny_gpu > 0 && nz_gpu > 0)                                                                    \
              {                                                                                                              \
                auto lambda_body =                                                                                           \
                  KOKKOS_LAMBDA(int i, int j, int k)                                                                         \
                {                                                                                                            \
                  i += ixl_gpu;                                                                                              \
                  j += iyl_gpu;                                                                                              \
                  k += izl_gpu;                                                                                              \
                                                                                                                             \
                  int ival = ptr_ival[(k - iz_bxs) * ny_bxs * nx_bxs +                                                       \
                                      (j - iy_bxs) * nx_bxs + (i - ix_bxs)];                                                 \
                  if (ival >= 0)                                                                                             \
                  {                                                                                                          \
                    UNPACK(locals);                                                                                          \
                    setup;                                                                                                   \
                    switch (PV_f)                                                                                            \
                    {                                                                                                        \
                    f_left;                                                                                                  \
                    f_right;                                                                                                 \
                    f_down;                                                                                                  \
                    f_up;                                                                                                    \
                    f_back;                                                                                                  \
                    f_front;                                                                                                 \
                    }                                                                                                        \
                    finalize;                                                                                                \
                  }                                                                                                          \
                };                                                                                                           \
                                                                                                                             \
                using MDPolicyType_3D = typename Kokkos::MDRangePolicy < Kokkos::Rank < 3 > >;                               \
                MDPolicyType_3D mdpolicy_3d({ { 0, 0, 0 } }, { { nx_gpu, ny_gpu, nz_gpu } });                                \
                Kokkos::parallel_for(mdpolicy_3d, lambda_body);                                                              \
                Kokkos::fence();                                                                                             \
              }                                                                                                              \
            }                                                                                                                \
          }                                                                                                                  \
          (void)i; (void)j; (void)k;                                                                                         \
        }

/** Loop definition for Kokkos. */
#define GrGeomOctreeExteriorNodeLoop_kokkos(i, j, k, node, octree, level,                                                      \
                                            ix, iy, iz, nx, ny, nz, val_test, loop_body)                                       \
        {                                                                                                                      \
          int PV_i, PV_j, PV_k, PV_l;                                                                                          \
          int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;                                                                  \
                                                                                                                               \
          PV_i = i;                                                                                                            \
          PV_j = j;                                                                                                            \
          PV_k = k;                                                                                                            \
                                                                                                                               \
          GrGeomOctreeExteriorLoop(PV_i, PV_j, PV_k, PV_l, node, octree, level, val_test,                                      \
    {                                                                                                                          \
      if ((PV_i >= ix) && (PV_i < (ix + nx)) &&                                                                                \
          (PV_j >= iy) && (PV_j < (iy + ny)) &&                                                                                \
          (PV_k >= iz) && (PV_k < (iz + nz)))                                                                                  \
      {                                                                                                                        \
        i = PV_i;                                                                                                              \
        j = PV_j;                                                                                                              \
        k = PV_k;                                                                                                              \
        loop_body;                                                                                                             \
      }                                                                                                                        \
    },                                                                                                                         \
    {                                                                                                                          \
      /* find octree and region intersection */                                                                                \
      PV_ixl = pfmax(ix, PV_i);                                                                                                \
      PV_iyl = pfmax(iy, PV_j);                                                                                                \
      PV_izl = pfmax(iz, PV_k);                                                                                                \
      PV_ixu = pfmin((ix + nx), (PV_i + (int)PV_inc));                                                                         \
      PV_iyu = pfmin((iy + ny), (PV_j + (int)PV_inc));                                                                         \
      PV_izu = pfmin((iz + nz), (PV_k + (int)PV_inc));                                                                         \
                                                                                                                               \
      if (PV_ixl < PV_ixu && PV_iyl < PV_iyu && PV_izl < PV_izu)                                                               \
      {                                                                                                                        \
        const int PV_diff_x = PV_ixu - PV_ixl;                                                                                 \
        const int PV_diff_y = PV_iyu - PV_iyl;                                                                                 \
        const int PV_diff_z = PV_izu - PV_izl;                                                                                 \
                                                                                                                               \
        auto lambda_body =                                                                                                     \
          KOKKOS_LAMBDA(int i, int j, int k)                                                                                   \
        {                                                                                                                      \
          i += PV_ixl;                                                                                                         \
          j += PV_iyl;                                                                                                         \
          k += PV_izl;                                                                                                         \
                                                                                                                               \
          loop_body;                                                                                                           \
        };                                                                                                                     \
                                                                                                                               \
        using MDPolicyType_3D = typename Kokkos::MDRangePolicy < Kokkos::Rank < 3 > >;                                         \
        MDPolicyType_3D mdpolicy_3d({ { 0, 0, 0 } }, { { PV_diff_x, PV_diff_y, PV_diff_z } });                                 \
        Kokkos::parallel_for(mdpolicy_3d, lambda_body);                                                                        \
        Kokkos::fence();                                                                                                       \
      }                                                                                                                        \
      i = PV_ixu;                                                                                                              \
      j = PV_iyu;                                                                                                              \
      k = PV_izu;                                                                                                              \
    })                                                                                                                         \
          (void)i; (void)j; (void)k;                                                                                           \
        }

/** Loop definition for Kokkos. */
#define GrGeomOutLoop_kokkos(i, j, k, grgeom, r,                                                                \
                             ix, iy, iz, nx, ny, nz, body)                                                      \
        {                                                                                                       \
          if (nx > 0 && ny > 0 && nz > 0)                                                                       \
          {                                                                                                     \
            if (!(GrGeomSolidCellFlagInitialized(grgeom) & (1 << 1)))                                           \
            {                                                                                                   \
              CheckCellFlagAllocation(grgeom, nx, ny, nz);                                                      \
              char *outflag = GrGeomSolidCellFlagData(grgeom);                                                  \
                                                                                                                \
              GrGeomOctree  *PV_node;                                                                           \
              double PV_ref = pow(2.0, r);                                                                      \
                                                                                                                \
              i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;                                                    \
              j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;                                                    \
              k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;                                                    \
              GrGeomOctreeExteriorNodeLoop(i, j, k, PV_node,                                                    \
                                           GrGeomSolidData(grgeom),                                             \
                                           GrGeomSolidOctreeBGLevel(grgeom) + r,                                \
                                           ix, iy, iz, nx, ny, nz,                                              \
                                           TRUE,                                                                \
        {                                                                                                       \
          body;                                                                                                 \
          outflag[(k - iz) * ny * nx + (j - iy) * nx + (i - ix)] |= (1 << 1);                                   \
        });                                                                                                     \
              GrGeomSolidCellFlagInitialized(grgeom) |= (1 << 1);                                               \
            }                                                                                                   \
            else                                                                                                \
            {                                                                                                   \
              char *outflag = GrGeomSolidCellFlagData(grgeom);                                                  \
              auto lambda_body =                                                                                \
                KOKKOS_LAMBDA(int i, int j, int k)                                                              \
              {                                                                                                 \
                i += ix;                                                                                        \
                j += iy;                                                                                        \
                k += iz;                                                                                        \
                                                                                                                \
                if (outflag[(k - iz) * ny * nx + (j - iy) * nx + (i - ix)] & (1 << 1))                          \
                {                                                                                               \
                  body;                                                                                         \
                }                                                                                               \
              };                                                                                                \
                                                                                                                \
              using MDPolicyType_3D = typename Kokkos::MDRangePolicy < Kokkos::Rank < 3 > >;                    \
              MDPolicyType_3D mdpolicy_3d({ { 0, 0, 0 } }, { { nx, ny, nz } });                                 \
              Kokkos::parallel_for(mdpolicy_3d, lambda_body);                                                   \
              Kokkos::fence();                                                                                  \
            }                                                                                                   \
          }                                                                                                     \
          (void)i; (void)j; (void)k;                                                                            \
        }
}
#endif // PF_KOKKOSLOOPS_H
