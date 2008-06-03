/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.3 $
 *********************************************************************EHEADER*/

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

typedef struct
{
   int  ix, iy, iz;      /* Bottom-lower-left corner in index-space */
   int  nx, ny, nz;      /* Size */
   int  sx, sy, sz;      /* Striding factors */
   int  rx, ry, rz;      /* Refinement over the background grid */
   int  level;           /* Refinement level = rx + ry + rz */

   int  process;        /* Process containing this subgrid */

} Subregion;

/*--------------------------------------------------------------------------
 * SubregionArray:
 *   This array is intended to be ordered by level.
 *--------------------------------------------------------------------------*/

typedef struct
{
   Subregion  **subregions;   /* Array of pointers to subregions */
   int          size;         /* Size of subgregion array */

} SubregionArray;

#define SubregionArrayBlocksize 10

/*--------------------------------------------------------------------------
 * Region:
 *--------------------------------------------------------------------------*/

typedef struct
{
   SubregionArray  **subregion_arrays;   /* Array of pointers to
					  * subregion arrays */
   int               size;               /* Size of region */

} SGSRegion;


/*--------------------------------------------------------------------------
 * Accessor macros: Subregion
 *--------------------------------------------------------------------------*/

#define SubregionIX(subregion)  ((subregion) -> ix)
#define SubregionIY(subregion)  ((subregion) -> iy)
#define SubregionIZ(subregion)  ((subregion) -> iz)

#define SubregionNX(subregion)  ((subregion) -> nx)
#define SubregionNY(subregion)  ((subregion) -> ny)
#define SubregionNZ(subregion)  ((subregion) -> nz)
  
#define SubregionSX(subregion)  ((subregion) -> sx)
#define SubregionSY(subregion)  ((subregion) -> sy)
#define SubregionSZ(subregion)  ((subregion) -> sz)
  
#define SubregionRX(subregion)  ((subregion) -> rx)
#define SubregionRY(subregion)  ((subregion) -> ry)
#define SubregionRZ(subregion)  ((subregion) -> rz)

#define SubregionLevel(subregion) ((subregion) -> level)

#define SubregionProcess(subregion) ((subregion) -> process)

/*--------------------------------------------------------------------------
 * Accessor macros: SubregionArray
 *--------------------------------------------------------------------------*/

#define SubregionArraySubregion(subregion_array, i) \
((subregion_array) -> subregions[(i)])
#define SubregionArraySize(subregion_array)  ((subregion_array) -> size)

/*--------------------------------------------------------------------------
 * Accessor macros: Region
 *--------------------------------------------------------------------------*/

#define RegionSubregionArray(region, i)  ((region) -> subregion_arrays[(i)])
#define RegionSize(region)               ((region) -> size)

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define ForSubregionI(i, subregion_array) \
for (i = 0; i < SubregionArraySize(subregion_array); i++)

#define ForSubregionArrayI(i, region) \
for (i = 0; i < RegionSize(region); i++)


#ifdef __STDC__
# define        ANSI_PROTO(s) s
#else
# define ANSI_PROTO(s) ()
#endif


/* region.c */
Subregion *NewSubregion ANSI_PROTO((int ix , int iy , int iz , int nx , int ny , int nz , int sx , int sy , int sz , int rx , int ry , int rz , int process ));
SubregionArray *NewSubregionArray ANSI_PROTO((void ));
SGSRegion *NewRegion ANSI_PROTO((int size ));
void FreeSubregion ANSI_PROTO((Subregion *subregion ));
void FreeSubregionArray ANSI_PROTO((SubregionArray *subregion_array ));
void FreeRegion ANSI_PROTO((SGSRegion *region ));
void AppendSubregion ANSI_PROTO((Subregion *subregion , SubregionArray **subregion_array ));
void AppendSubregionArray ANSI_PROTO((SubregionArray *subregion_array_0 , SubregionArray **subregion_array_1 ));

#undef ANSI_PROTO

#endif
