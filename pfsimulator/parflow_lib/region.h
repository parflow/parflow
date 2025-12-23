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
/*****************************************************************************
*
* Header info for the Region structures
*
*****************************************************************************/

#ifndef _REGION_HEADER
#define _REGION_HEADER


/*--------------------------------------------------------------------------
 * Terminology:
 *   The Background is a uniform grid with position in real-space
 *   given by the quantities X, Y, Z, NX, NY, NZ, DX, DY, DZ.
 *   It is a global structure.
 *
 *   A Subregion is defined in terms of a uniform "index space".
 *   Each index space is a refinement of the Background given by the
 *   resolutions rx, ry, and rz (note, these quantities may be negative
 *   indicating coarser spacing).  Each of these index spaces define a
 *   unique "level", and these levels are labeled as (rx + ry + rz).
 *   Since levels are unique, this means that rs^{l+1} >= rs^{l} for all
 *   levels l, and all s = {x,y,z}.
 *
 *   A Subregion defines a cartesian region of index-space.  It is
 *   described by the quantities xi, yi, zi, nx, ny, nz, sx, sy, sz.
 *   The sx, sy, sz values are striding factors in each coordinate
 *   direction.  These striding factors will allow us to define things
 *   like "red points" or "black points" for red/black iterative methods.
 *   We will also be able to define "coarse points" and "fine points"
 *   for use in multigrid methods.
 *
 *   A SubregionArray is just an array of Subregions.
 *
 *   A Region is an array of SubregionArrays, where each SubregionArray
 *   is usually associated with a particular Subgrid (see grid.h).
 *
 *   Note: Since Subgrids and Subregions are so similar, we use the same
 *   structure to define them both.  Hence, a Subgrid should be thought
 *   of as a Subregion with striding factors 1.
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Subregion:
 *   Structure describing an index-space cartesian region.
 *--------------------------------------------------------------------------*/

typedef struct {
  int ix, iy, iz;        /* Bottom-lower-left corner in index-space */
  int nx, ny, nz;        /* Size */
  int sx, sy, sz;        /* Striding factors */
  int rx, ry, rz;        /* Refinement over the background grid */
  int level;             /* Refinement level = rx + ry + rz */

  int process;           /* Process containing this subgrid */
} Subregion;

/*--------------------------------------------------------------------------
 * SubregionArray:
 *   This array is intended to be ordered by level.
 *--------------------------------------------------------------------------*/

typedef struct {
  Subregion  **subregions;    /* Array of pointers to subregions */
  int size;                   /* Size of subregion array */
} SubregionArray;

#define SubregionArrayBlocksize 10

/*--------------------------------------------------------------------------
 * Region:
 *--------------------------------------------------------------------------*/

typedef struct {
  SubregionArray  **subregion_arrays;    /* Array of pointers to
                                          * subregion arrays */
  int size;                              /* Size of region */
} Region;


/*--------------------------------------------------------------------------
 * Accessor macros: Subregion
 *--------------------------------------------------------------------------*/

#define SubregionIX(subregion)  ((subregion)->ix)
#define SubregionIY(subregion)  ((subregion)->iy)
#define SubregionIZ(subregion)  ((subregion)->iz)

#define SubregionNX(subregion)  ((subregion)->nx)
#define SubregionNY(subregion)  ((subregion)->ny)
#define SubregionNZ(subregion)  ((subregion)->nz)

#define SubregionSX(subregion)  ((subregion)->sx)
#define SubregionSY(subregion)  ((subregion)->sy)
#define SubregionSZ(subregion)  ((subregion)->sz)

#define SubregionRX(subregion)  ((subregion)->rx)
#define SubregionRY(subregion)  ((subregion)->ry)
#define SubregionRZ(subregion)  ((subregion)->rz)

#define SubregionLevel(subregion) ((subregion)->level)

#define SubregionProcess(subregion) ((subregion)->process)

/*--------------------------------------------------------------------------
 * Accessor macros: SubregionArray
 *--------------------------------------------------------------------------*/

#define SubregionArraySubregions(subregion_array) \
        ((subregion_array)->subregions)
#define SubregionArraySubregion(subregion_array, i) \
        ((subregion_array)->subregions[(i)])
#define SubregionArraySize(subregion_array)  ((subregion_array)->size)

/*--------------------------------------------------------------------------
 * Accessor macros: Region
 *--------------------------------------------------------------------------*/

#define RegionSubregionArrays(region)    ((region)->subregion_arrays)
#define RegionSubregionArray(region, i)  ((region)->subregion_arrays[(i)])
#define RegionSize(region)               ((region)->size)

/*--------------------------------------------------------------------------
 * RealSpace/IndexSpace macros:
 *--------------------------------------------------------------------------*/

#define RealSpaceDX(rx) \
        (BackgroundDX(GlobalsBackground) / pow(2.0, (double)rx))
#define RealSpaceDY(ry) \
        (BackgroundDY(GlobalsBackground) / pow(2.0, (double)ry))
#define RealSpaceDZ(rz) \
        (BackgroundDZ(GlobalsBackground) / pow(2.0, (double)rz))

#define RealSpaceX(ix, rx) \
        (BackgroundX(GlobalsBackground) + (ix + 0.5) * RealSpaceDX(rx))
#define RealSpaceY(iy, ry) \
        (BackgroundY(GlobalsBackground) + (iy + 0.5) * RealSpaceDY(ry))
#define RealSpaceZ(iz, rz) \
        (BackgroundZ(GlobalsBackground) + (iz + 0.5) * RealSpaceDZ(rz))

#define IndexSpaceNX(rx) \
        (BackgroundNX(GlobalsBackground) * (int)pow(2.0, rx))
#define IndexSpaceNY(ry) \
        (BackgroundNY(GlobalsBackground) * (int)pow(2.0, ry))
#define IndexSpaceNZ(rz) \
        (BackgroundNZ(GlobalsBackground) * (int)pow(2.0, rz))

#define IndexSpaceX(x, rx) \
        (pfround((x - RealSpaceX(0, rx)) / RealSpaceDX(rx)))
#define IndexSpaceY(y, ry) \
        (pfround((y - RealSpaceY(0, ry)) / RealSpaceDY(ry)))
#define IndexSpaceZ(z, rz) \
        (pfround((z - RealSpaceZ(0, rz)) / RealSpaceDZ(rz)))

/*--------------------------------------------------------------------------
 * Utility macros:
 *--------------------------------------------------------------------------*/

#define SubregionDX(subregion) \
        (SubregionSX(subregion) * (RealSpaceDX(SubregionRX(subregion))))
#define SubregionDY(subregion) \
        (SubregionSY(subregion) * (RealSpaceDY(SubregionRY(subregion))))
#define SubregionDZ(subregion) \
        (SubregionSZ(subregion) * (RealSpaceDZ(SubregionRZ(subregion))))

#define SubregionX(subregion) \
        RealSpaceX(SubregionIX(subregion), SubregionRX(subregion))
#define SubregionY(subregion) \
        RealSpaceY(SubregionIY(subregion), SubregionRY(subregion))
#define SubregionZ(subregion) \
        RealSpaceZ(SubregionIZ(subregion), SubregionRZ(subregion))

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define ForSubregionI(i, subregion_array) \
        for (i = 0; i < SubregionArraySize(subregion_array); i++)

#define ForSubregionArrayI(i, region) \
        for (i = 0; i < RegionSize(region); i++)


#endif
