/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#ifndef _GRID_HEADER
#define _GRID_HEADER

#include "region.h"

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

typedef Subregion  Subgrid;

/*--------------------------------------------------------------------------
 * SubgridArray:
 *   This array is intended to be ordered by level.
 *--------------------------------------------------------------------------*/

typedef SubregionArray  SubgridArray;

/*--------------------------------------------------------------------------
 * Grid:
 *--------------------------------------------------------------------------*/

typedef struct
{
   SubgridArray  *subgrids;     /* Array of subgrids in this process */

   SubgridArray  *all_subgrids; /* Array of all subgrids in the grid */

   SubgridArray  *neighbors;    /* Array of nearest neighbor subgrids */

   int            size;         /* Total number of grid points */

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
((Subgrid *) SubregionArraySubregion(subgrid_array, i))
#define SubgridArraySize(subgrid_array)  SubregionArraySize(subgrid_array)

/*--------------------------------------------------------------------------
 * Accessor macros: Grid
 *--------------------------------------------------------------------------*/

#define GridSubgrids(grid)    ((grid) -> subgrids)
#define GridAllSubgrids(grid) ((grid) -> all_subgrids)
#define GridNeighbors(grid)   ((grid) -> neighbors)

#define GridSize(grid)   ((grid) -> size)

#define GridSubgrid(grid, i)  (SubgridArraySubgrid(GridSubgrids(grid), i))
#define GridNumSubgrids(grid) (SubgridArraySize(GridSubgrids(grid)))

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define ForSubgridI(i, subgrid_array)  ForSubregionI(i, subgrid_array)

/*--------------------------------------------------------------------------
 * Class member functions:
 *--------------------------------------------------------------------------*/

#define NewSubgrid(x, y, z, nx, ny, nz, rx, ry, rz, process)  \
((Subgrid *) NewSubregion(x, y, z, nx, ny, nz, 1, 1, 1, rx, ry, rz, process))

#define NewSubgridArray()  ((SubgridArray *) NewSubregionArray())

#define FreeSubgrid(subgrid)  FreeSubregion((Subregion *) subgrid)

#define FreeSubgridArray(subgrid_array)  \
FreeSubregionArray((SubregionArray *) subgrid_array)

#define AppendSubgrid(subgrid, subgrid_array)  \
AppendSubregion((Subregion *) subgrid, (SubregionArray **) subgrid_array)

#define AppendSubgridArray(subgrid_array_0, subgrid_array_1)  \
AppendSubregionArray((SubregionArray *)  subgrid_array_0, \
		     (SubregionArray **) subgrid_array_1)


#ifdef __STDC__
# define        ANSI_PROTO(s) s
#else
# define ANSI_PROTO(s) ()
#endif


/* grid.c */
Grid *NewGrid ANSI_PROTO((SubgridArray *subgrids , SubgridArray *all_subgrids , SubgridArray *neighbors ));
void FreeGrid ANSI_PROTO((Grid *grid ));

#undef ANSI_PROTO

#endif


