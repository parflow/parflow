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
* Header info for the Grid structures
*
*****************************************************************************/

#ifndef _GRID_HEADER
#define _GRID_HEADER

/*--------------------------------------------------------------------------
 * Terminology:
 *   See region.h
 *
 *   A Subgrid is a Subregion with striding factors equal 1.
 *
 *   A Grid is a composition of Subgrids.
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Subgrid:
 *   Structure describing an index-space cartesian grid.
 *--------------------------------------------------------------------------*/

typedef Subregion Subgrid;

/*--------------------------------------------------------------------------
 * SubgridArray:
 *   This array is intended to be ordered by level.
 *--------------------------------------------------------------------------*/

typedef SubregionArray SubgridArray;

/*--------------------------------------------------------------------------
 * Grid:
 *--------------------------------------------------------------------------*/

typedef struct {
  SubgridArray  *subgrids;      /* Array of subgrids in this process */

  SubgridArray  *all_subgrids;  /* Array of all subgrids in the grid */

  int size;                     /* Total number of grid points */

  ComputePkg   **compute_pkgs;

  Subgrid        *background;    /* The background reference grid for
                                  * this grid.   Includes the entire
                                  * space of points that all subgrids
                                  * lie in.  Basically the bounding
                                  * box for the subgrids */
} Grid;


/*--------------------------------------------------------------------------
 * Accessor macros: Subgrid
 *--------------------------------------------------------------------------*/

#define SubgridIX(subgrid)  SubregionIX(subgrid)
#define SubgridIY(subgrid)  SubregionIY(subgrid)
#define SubgridIZ(subgrid)  SubregionIZ(subgrid)

#define SubgridNX(subgrid)  SubregionNX(subgrid)
#define SubgridNY(subgrid)  SubregionNY(subgrid)
#define SubgridNZ(subgrid)  SubregionNZ(subgrid)

#define SubgridRX(subgrid)  SubregionRX(subgrid)
#define SubgridRY(subgrid)  SubregionRY(subgrid)
#define SubgridRZ(subgrid)  SubregionRZ(subgrid)

#define SubgridLevel(subgrid)  SubregionLevel(subgrid)

#define SubgridProcess(subgrid)  SubregionProcess(subgrid)

/*--------------------------------------------------------------------------
 * Accessor macros: SubgridArray
 *--------------------------------------------------------------------------*/

#define SubgridArraySubgrid(subgrid_array, i) \
        ((Subgrid*)SubregionArraySubregion(subgrid_array, i))
#define SubgridArraySize(subgrid_array)  SubregionArraySize(subgrid_array)

/*--------------------------------------------------------------------------
 * Accessor macros: Grid
 *--------------------------------------------------------------------------*/

#define GridBackground(grid)  ((grid)->background)

#define GridSubgrids(grid)    ((grid)->subgrids)
#define GridAllSubgrids(grid) ((grid)->all_subgrids)

#define GridSize(grid)   ((grid)->size)

#define GridComputePkgs(grid)   ((grid)->compute_pkgs)
#define GridComputePkg(grid, i) ((grid)->compute_pkgs[(i)])

#define GridSubgrid(grid, i)  (SubgridArraySubgrid(GridSubgrids(grid), i))
#define GridNumSubgrids(grid) (SubgridArraySize(GridSubgrids(grid)))

/*--------------------------------------------------------------------------
 * Utility macros:
 *--------------------------------------------------------------------------*/

#define SubgridDX(subgrid) \
        RealSpaceDX(SubgridRX(subgrid))
#define SubgridDY(subgrid) \
        RealSpaceDY(SubgridRY(subgrid))
#define SubgridDZ(subgrid) \
        RealSpaceDZ(SubgridRZ(subgrid))

#define SubgridX(subgrid) \
        RealSpaceX(SubgridIX(subgrid), SubgridRX(subgrid))
#define SubgridY(subgrid) \
        RealSpaceY(SubgridIY(subgrid), SubgridRY(subgrid))
#define SubgridZ(subgrid) \
        RealSpaceZ(SubgridIZ(subgrid), SubgridRZ(subgrid))

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define ForSubgridI(i, subgrid_array)  ForSubregionI(i, subgrid_array)

/*--------------------------------------------------------------------------
 * Class member functions:
 *--------------------------------------------------------------------------*/

#define NewSubgrid(x, y, z, nx, ny, nz, rx, ry, rz, process) \
        ((Subgrid*)NewSubregion(x, y, z, nx, ny, nz, 1, 1, 1, rx, ry, rz, process))

#define NewSubgridArray()  ((SubgridArray*)NewSubregionArray())

#define FreeSubgrid(subgrid)  FreeSubregion((Subregion*)subgrid)

#define FreeSubgridArray(subgrid_array) \
        FreeSubregionArray((SubregionArray*)subgrid_array)

#define DuplicateSubgrid(subgrid) \
        ((Subgrid*)DuplicateSubregion((Subregion*)subgrid))

#define AppendSubgrid(subgrid, subgrid_array) \
        AppendSubregion((Subregion*)subgrid, (SubregionArray*)subgrid_array)

#define AppendSubgridArray(subgrid_array_0, subgrid_array_1)       \
        AppendSubregionArray((SubregionArray*)subgrid_array_0,     \
                             (SubregionArray*)subgrid_array_1)

#define ConvertToSubregion(subgrid)  ((Subregion*)subgrid)

#define SubgridEltIndex(subgrid, x, y, z)       \
        (((x) - SubgridIX(subgrid)) +           \
         (((y) - SubgridIY(subgrid)) +          \
          (((z) - SubgridIZ(subgrid))) *        \
          SubgridNY(subgrid)) *                 \
         SubgridNX(subgrid))

#endif
