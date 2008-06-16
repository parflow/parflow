/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Geometry class structures and accessors
 *
 *****************************************************************************/

#ifndef _GR_GEOMETRY_HEADER
#define _GR_GEOMETRY_HEADER

#include "geometry.h"
#include "grgeom_octree.h"
#include "grgeom_list.h"


/*--------------------------------------------------------------------------
 * Miscellaneous structures:
 *--------------------------------------------------------------------------*/

typedef int GrGeomExtents[6];

typedef struct
{
   GrGeomExtents  *extents;
   int             size;

} GrGeomExtentArray;


/*--------------------------------------------------------------------------
 * Solid structures:
 *--------------------------------------------------------------------------*/

typedef struct
{
   GrGeomOctree  *data;

   GrGeomOctree **patches;
   int            num_patches;

   /* these fields are used to relate the background with the octree */
   int            octree_bg_level;
   int            octree_ix, octree_iy, octree_iz;

} GrGeomSolid;


/*--------------------------------------------------------------------------
 * Accessor macros:
 *--------------------------------------------------------------------------*/

#define GrGeomExtentsIXLower(extents)  ((extents)[0])
#define GrGeomExtentsIXUpper(extents)  ((extents)[1])
#define GrGeomExtentsIYLower(extents)  ((extents)[2])
#define GrGeomExtentsIYUpper(extents)  ((extents)[3])
#define GrGeomExtentsIZLower(extents)  ((extents)[4])
#define GrGeomExtentsIZUpper(extents)  ((extents)[5])

#define GrGeomExtentArrayExtents(ext_array)  ((ext_array) -> extents)
#define GrGeomExtentArraySize(ext_array)     ((ext_array) -> size)

#define GrGeomSolidData(solid)          ((solid) -> data)
#define GrGeomSolidPatches(solid)       ((solid) -> patches)
#define GrGeomSolidNumPatches(solid)    ((solid) -> num_patches)
#define GrGeomSolidOctreeBGLevel(solid) ((solid) -> octree_bg_level)
#define GrGeomSolidOctreeIX(solid)      ((solid) -> octree_ix)
#define GrGeomSolidOctreeIY(solid)      ((solid) -> octree_iy)
#define GrGeomSolidOctreeIZ(solid)      ((solid) -> octree_iz)
#define GrGeomSolidPatch(solid, i)      ((solid) -> patches[i])


/*==========================================================================
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the inside of a solid.
 *--------------------------------------------------------------------------*/

#define GrGeomInLoop(i, j, k, grgeom,\
		     r, ix, iy, iz, nx, ny, nz, body)\
{\
   GrGeomOctree  *PV_node;\
   double         PV_ref = pow(2.0, r);\
\
\
   i = GrGeomSolidOctreeIX(grgeom)*(int)PV_ref;\
   j = GrGeomSolidOctreeIY(grgeom)*(int)PV_ref;\
   k = GrGeomSolidOctreeIZ(grgeom)*(int)PV_ref;\
   GrGeomOctreeNodeLoop(i, j, k, PV_node,\
			GrGeomSolidData(grgeom),\
			GrGeomSolidOctreeBGLevel(grgeom) + r,\
			ix, iy, iz, nx, ny, nz,\
			(GrGeomOctreeCellIsInside(PV_node) ||\
			 GrGeomOctreeCellIsFull(PV_node)),\
			body);\
}

/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the inside of a solid with non-unitary strides.
 *--------------------------------------------------------------------------*/

#define GrGeomInLoop2(i, j, k, grgeom,\
		      r, ix, iy, iz, nx, ny, nz, sx, sy, sz, body)\
{\
   GrGeomOctree  *PV_node;\
   double         PV_ref = pow(2.0, r);\
\
\
   i = GrGeomSolidOctreeIX(grgeom)*PV_ref;\
   j = GrGeomSolidOctreeIY(grgeom)*PV_ref;\
   k = GrGeomSolidOctreeIZ(grgeom)*PV_ref;\
   GrGeomOctreeNodeLoop2(i, j, k, PV_node,\
			 GrGeomSolidData(grgeom),\
			 GrGeomSolidOctreeBGLevel(grgeom) + r,\
			 ix, iy, iz, nx, ny, nz, sx, sy, sz,\
			 (GrGeomOctreeCellIsInside(PV_node) ||\
			  GrGeomOctreeCellIsFull(PV_node)),\
			 body);\
}

/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the outside of a solid.
 *--------------------------------------------------------------------------*/

#define GrGeomOutLoop(i, j, k, grgeom,\
		      r, ix, iy, iz, nx, ny, nz, body)\
{\
   GrGeomOctree  *PV_node;\
   double         PV_ref = pow(2.0, r);\
\
\
   i = GrGeomSolidOctreeIX(grgeom)*(int)PV_ref;\
   j = GrGeomSolidOctreeIY(grgeom)*(int)PV_ref;\
   k = GrGeomSolidOctreeIZ(grgeom)*(int)PV_ref;\
   GrGeomOctreeNodeLoop(i, j, k, PV_node,\
			GrGeomSolidData(grgeom),\
			GrGeomSolidOctreeBGLevel(grgeom) + r,\
			ix, iy, iz, nx, ny, nz,\
			(GrGeomOctreeCellIsOutside(PV_node) ||\
			 GrGeomOctreeCellIsEmpty(PV_node)),\
			body);\
}

/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the outside of a solid with non-unitary strides.
 *--------------------------------------------------------------------------*/

#define GrGeomOutLoop2(i, j, k, grgeom,\
		       r, ix, iy, iz, nx, ny, nz, sx, sy, sz, body)\
{\
   GrGeomOctree  *PV_node;\
   double         PV_ref = pow(2.0, r);\
\
\
   i = GrGeomSolidOctreeIX(grgeom)*(int)PV_ref;\
   j = GrGeomSolidOctreeIY(grgeom)*(int)PV_ref;\
   k = GrGeomSolidOctreeIZ(grgeom)*(int)PV_ref;\
   GrGeomOctreeNodeLoop2(i, j, k, PV_node,\
			 GrGeomSolidData(grgeom),\
			 GrGeomSolidOctreeBGLevel(grgeom) + r,\
			 ix, iy, iz, nx, ny, nz, sx, sy, sz,\
			 (GrGeomOctreeCellIsOutside(PV_node) ||\
			  GrGeomOctreeCellIsEmpty(PV_node)),\
			 body);\
}

/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the faces of a solid surface.
 *--------------------------------------------------------------------------*/

#define GrGeomSurfLoop(i, j, k, fdir, grgeom,\
		       r, ix, iy, iz, nx, ny, nz, body)\
{\
   GrGeomOctree  *PV_node;\
   double         PV_ref = pow(2.0, r);\
\
\
   i = GrGeomSolidOctreeIX(grgeom)*(int)PV_ref;\
   j = GrGeomSolidOctreeIY(grgeom)*(int)PV_ref;\
   k = GrGeomSolidOctreeIZ(grgeom)*(int)PV_ref;\
   GrGeomOctreeFaceLoop(i, j, k, fdir, PV_node,\
			GrGeomSolidData(grgeom),\
			GrGeomSolidOctreeBGLevel(grgeom) + r,\
			ix, iy, iz, nx, ny, nz, body);\
}

/*--------------------------------------------------------------------------
 * GrGeomSolid looping macro:
 *   Macro for looping over the faces of a solid patch.
 *--------------------------------------------------------------------------*/

#define GrGeomPatchLoop(i, j, k, fdir, grgeom, patch_num,\
			r, ix, iy, iz, nx, ny, nz, body)\
{\
   GrGeomOctree  *PV_node;\
   double         PV_ref = pow(2.0, r);\
\
\
   i = GrGeomSolidOctreeIX(grgeom)*(int)PV_ref;\
   j = GrGeomSolidOctreeIY(grgeom)*(int)PV_ref;\
   k = GrGeomSolidOctreeIZ(grgeom)*(int)PV_ref;\
   GrGeomOctreeFaceLoop(i, j, k, fdir, PV_node,\
			GrGeomSolidPatch(grgeom, patch_num),\
			GrGeomSolidOctreeBGLevel(grgeom) + r,\
			ix, iy, iz, nx, ny, nz, body);\
}


#endif
