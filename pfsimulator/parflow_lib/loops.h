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
 * @brief Generic loop macros
 */

/*****************************************************************************
* Generic Loop Macros
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/
#ifndef _LOOPS_HEADER
#define _LOOPS_HEADER

/**
 * @brief Declare increment variables for striding indices in BoxLoops
 * @param[in,out] jinc Name of the Y increment variable to be declared
 * @param[in,out] kinc Name of the Z increment variable to be declared
 * @param[in] nx X size of subgrid
 * @param[in] ny Y size of subgrid
 * @param[in] nz Z size of subgrid
 * @param[in] nxd X size of vector subregion
 * @param[in] nyd Y size of vector subregion
 * @param[in] nzd Z size of vector subregion
 * @param[in] sx X striding factor
 * @param[in] sy Y striding factor
 * @param[in] sz Z striding factor
 */
#define DeclareInc(jinc, kinc, nx, ny, nz, nxd, nyd, nzd, sx, sy, sz)       \
        int jinc = (sy) * (nxd) - (nx) * (sx);                              \
        int kinc = (sz) * (nxd) * (nyd) - (ny) * (sy) * (nxd);              \
        PF_UNUSED(nz);                                                      \
        PF_UNUSED(nzd)

/**
 * @brief Perform a reduction over a BoxLoopI1 iteration space
 *
 * @note Last statement in loop body must be a valid reduction clause (see general.h).
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param[in,out] sum Variable to perform reduction operation on.
 */
#define BoxLoopReduceI1_default(sum, ...) BoxLoopI1(__VA_ARGS__)

/**
 * @brief Perform a reduction over a BoxLoopI2 iteration space
 *
 * @note Last statement in loop body must be a valid reduction clause (see general.h).
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param[in,out] sum Variable to perform reduction operation on.
 */
#define BoxLoopReduceI2_default(sum, ...) BoxLoopI2(__VA_ARGS__)

/**
 * @brief Perform a reduction over a BoxLoopI3 iteration space
 *
 * @note Last statement in loop body must be a valid reduction clause (see general.h).
 * @note Multiple definitions (see backend_mapping.h).
 *
 * @param[in,out] sum Variable to perform reduction operation on.
 */
#define BoxLoopReduceI3_default(sum, ...) BoxLoopI3(__VA_ARGS__)

#define BoxLoopI0_default(i, j, k,                \
                          ix, iy, iz, nx, ny, nz, \
                          body)                   \
        {                                         \
          for (k = iz; k < iz + nz; k++)          \
          {                                       \
            for (j = iy; j < iy + ny; j++)        \
            {                                     \
              for (i = ix; i < ix + nx; i++)      \
              {                                   \
                body;                             \
              }                                   \
            }                                     \
          }                                       \
        }

#define BoxLoopI1_default(i, j, k,                                                      \
                          ix, iy, iz, nx, ny, nz,                                       \
                          i1, nx1, ny1, nz1, sx1, sy1, sz1,                             \
                          body)                                                         \
        {                                                                               \
          DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);   \
          for (k = iz; k < iz + nz; k++)                                                \
          {                                                                             \
            for (j = iy; j < iy + ny; j++)                                              \
            {                                                                           \
              for (i = ix; i < ix + nx; i++)                                            \
              {                                                                         \
                body;                                                                   \
                i1 += sx1;                                                              \
              }                                                                         \
              i1 += PV_jinc_1;                                                          \
            }                                                                           \
            i1 += PV_kinc_1;                                                            \
          }                                                                             \
        }

#define BoxLoopI2_default(i, j, k,                                                      \
                          ix, iy, iz, nx, ny, nz,                                       \
                          i1, nx1, ny1, nz1, sx1, sy1, sz1,                             \
                          i2, nx2, ny2, nz2, sx2, sy2, sz2,                             \
                          body)                                                         \
        {                                                                               \
          DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);   \
          DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2);   \
          for (k = iz; k < iz + nz; k++)                                                \
          {                                                                             \
            for (j = iy; j < iy + ny; j++)                                              \
            {                                                                           \
              for (i = ix; i < ix + nx; i++)                                            \
              {                                                                         \
                body;                                                                   \
                i1 += sx1;                                                              \
                i2 += sx2;                                                              \
              }                                                                         \
              i1 += PV_jinc_1;                                                          \
              i2 += PV_jinc_2;                                                          \
            }                                                                           \
            i1 += PV_kinc_1;                                                            \
            i2 += PV_kinc_2;                                                            \
          }                                                                             \
        }

#define BoxLoopI3_default(i, j, k,                                                      \
                          ix, iy, iz, nx, ny, nz,                                       \
                          i1, nx1, ny1, nz1, sx1, sy1, sz1,                             \
                          i2, nx2, ny2, nz2, sx2, sy2, sz2,                             \
                          i3, nx3, ny3, nz3, sx3, sy3, sz3,                             \
                          body)                                                         \
        {                                                                               \
          DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);   \
          DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2);   \
          DeclareInc(PV_jinc_3, PV_kinc_3, nx, ny, nz, nx3, ny3, nz3, sx3, sy3, sz3);   \
          for (k = iz; k < iz + nz; k++)                                                \
          {                                                                             \
            for (j = iy; j < iy + ny; j++)                                              \
            {                                                                           \
              for (i = ix; i < ix + nx; i++)                                            \
              {                                                                         \
                body;                                                                   \
                i1 += sx1;                                                              \
                i2 += sx2;                                                              \
                i3 += sx3;                                                              \
              }                                                                         \
              i1 += PV_jinc_1;                                                          \
              i2 += PV_jinc_2;                                                          \
              i3 += PV_jinc_3;                                                          \
            }                                                                           \
            i1 += PV_kinc_1;                                                            \
            i2 += PV_kinc_2;                                                            \
            i3 += PV_kinc_3;                                                            \
          }                                                                             \
        }

/******************************************************************************
*     SPECIAL NOTE! SPECIAL NOTE! SPECIAL NOTE! SPECIAL NOTE! SPECIAL NOTE!   *
*                                                                             *
*    The Cray T3D C compiler only allows 31 arguments to a macro, thus the    *
*       standard definition of BoxLoopI4 would not compile (it has 38).       *
*                                                                             *
*   But it REALLY is what needs to be done to maintain consistency with the   *
*   code as well as just plain being easier to understand.    At this point   *
*   the BoxLoopI4 is only used in the advection routines, which only needs    *
*   stride = 1.  Thus I've taken the striding factors out and hardcoded them  *
*   to be 1.  If you stumble upon this message because your trying to use the *
*   BoxLoopI4 like it should be used, take note.  You can put it back like    *
*   it's supposed to be and try it out.  Otherwise your stuck.                *
******************************************************************************/

#define BoxLoopI4(i, j, k,                                                      \
                  ix, iy, iz, nx, ny, nz,                                       \
                  i1, nx1, ny1, nz1,                                            \
                  i2, nx2, ny2, nz2,                                            \
                  i3, nx3, ny3, nz3,                                            \
                  i4, nx4, ny4, nz4,                                            \
                  body)                                                         \
        {                                                                       \
          DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, 1, 1, 1); \
          DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, 1, 1, 1); \
          DeclareInc(PV_jinc_3, PV_kinc_3, nx, ny, nz, nx3, ny3, nz3, 1, 1, 1); \
          DeclareInc(PV_jinc_4, PV_kinc_4, nx, ny, nz, nx4, ny4, nz4, 1, 1, 1); \
          for (k = iz; k < iz + nz; k++)                                        \
          {                                                                     \
            for (j = iy; j < iy + ny; j++)                                      \
            {                                                                   \
              for (i = ix; i < ix + nx; i++)                                    \
              {                                                                 \
                body;                                                           \
                i1 += 1;                                                        \
                i2 += 1;                                                        \
                i3 += 1;                                                        \
                i4 += 1;                                                        \
              }                                                                 \
              i1 += PV_jinc_1;                                                  \
              i2 += PV_jinc_2;                                                  \
              i3 += PV_jinc_3;                                                  \
              i4 += PV_jinc_4;                                                  \
            }                                                                   \
            i1 += PV_kinc_1;                                                    \
            i2 += PV_kinc_2;                                                    \
            i3 += PV_kinc_3;                                                    \
            i4 += PV_kinc_4;                                                    \
          }                                                                     \
        }

#define pgs_BoxLoopI2(i, j, k,                      \
                      ix, iy, iz, nx, ny, nz,       \
                      sx, sy, sz,                   \
                      i1, nx1, ny1, nz1,            \
                      i2, nx2, ny2, nz2,            \
                      body)                         \
        {                                           \
          i1 = 0;                                   \
          i2 = 0;                                   \
          for (k = iz; k < nz; k += sz)             \
          {                                         \
            for (j = iy; j < ny; j += sy)           \
            {                                       \
              for (i = ix; i < nx; i += sx)         \
              {                                     \
                body;                               \
                i1 += sx;                           \
                i2 += sx;                           \
              }                                     \
              i1 += (sy * nx1 - i + ix);            \
              i2 += (sy * nx2 - i + ix);            \
            }                                       \
            i1 = (k - iz + sz) * nx1 * ny1;         \
            i2 = (k - iz + sz) * nx2 * ny2;         \
          }                                         \
        }

#endif
