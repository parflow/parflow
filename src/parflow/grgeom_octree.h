/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#ifndef _GRGEOM_OCTREE_HEADER
#define _GRGEOM_OCTREE_HEADER

/*--------------------------------------------------------------------------
 * GrGeomOctree class
 *
 * Accessors are consistent with the labeling in the literature :
 *       L - left  (x coord. small)
 *       R - right (x coord. big)
 *       D - down  (y coord. small)
 *       U - up    (y coord. big)
 *       B - back  (z coord. small)
 *       F - front (z coord. big)
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * GrGeomOctree structure
 *--------------------------------------------------------------------------*/

typedef struct grgeom_octree
{
   unsigned char           cell;
   unsigned char           faces;
   struct grgeom_octree   *parent;
   struct grgeom_octree  **children;

} GrGeomOctree;

/*--------------------------------------------------------------------------
 * GrGeomOctree Size constants
 *--------------------------------------------------------------------------*/

#define GrGeomOctreeNumFaces        6
#define GrGeomOctreeNumChildren     8

/*--------------------------------------------------------------------------
 * GrGeomOctree Cell values
 *
 * A cell can have only one of the following values:
 *
 *       GrGeomOctreeCellEmpty
 *       GrGeomOctreeCellOutside
 *       GrGeomOctreeCellInside
 *       GrGeomOctreeCellFull
 *
 * In addition, each cell is either a GrGeomOctreeCellLeaf or not.
 *
 * To keep the above rules consistent, use the GrGeomOctreeSetCell
 * macro below to set cell values, and use the GrGeomOctreeSetCellLeaf
 * and GrGeomOctreeClearCellLeaf macros to set leaf indicator.
 *--------------------------------------------------------------------------*/

#define GrGeomOctreeCellEmpty     ((unsigned char) 0x01)
#define GrGeomOctreeCellOutside   ((unsigned char) 0x02)
#define GrGeomOctreeCellInside    ((unsigned char) 0x04)
#define GrGeomOctreeCellFull      ((unsigned char) 0x08)
#define GrGeomOctreeCellLeaf      ((unsigned char) 0x10)

/*--------------------------------------------------------------------------
 * GrGeomOctree face constants
 *--------------------------------------------------------------------------*/

#define GrGeomOctreeFaceL 0
#define GrGeomOctreeFaceR 1
#define GrGeomOctreeFaceD 2
#define GrGeomOctreeFaceU 3
#define GrGeomOctreeFaceB 4
#define GrGeomOctreeFaceF 5

/*--------------------------------------------------------------------------
 * GrGeomOctree octant constants
 *--------------------------------------------------------------------------*/

#define GrGeomOctreeOctantLDB 0
#define GrGeomOctreeOctantRDB 1
#define GrGeomOctreeOctantLUB 2
#define GrGeomOctreeOctantRUB 3
#define GrGeomOctreeOctantLDF 4
#define GrGeomOctreeOctantRDF 5
#define GrGeomOctreeOctantLUF 6
#define GrGeomOctreeOctantRUF 7

/*--------------------------------------------------------------------------
 * GrGeomOctree misc
 *--------------------------------------------------------------------------*/

#define OUTSIDE 0
#define INSIDE  1

/*--------------------------------------------------------------------------
 * GrGeomOctree accessor macros
 *--------------------------------------------------------------------------*/

#define GrGeomOctreeCell(octree)      ((octree) -> cell)
#define GrGeomOctreeParent(octree)    ((octree) -> parent)
#define GrGeomOctreeChildren(octree)  ((octree) -> children)
#define GrGeomOctreeChild(octree, i)  ((octree) -> children[i])
#define GrGeomOctreeFaces(octree)     ((octree) -> faces)

/*--------------------------------------------------------------------------
 * GrGeomOctree macro functions for cells
 *--------------------------------------------------------------------------*/

/* Set cell values */
#define GrGeomOctreeSetCell(octree, cell_value) \
GrGeomOctreeCell(octree) =\
(GrGeomOctreeCell(octree) & GrGeomOctreeCellLeaf) | cell_value

/* Set leaf indicator */
#define GrGeomOctreeSetCellLeaf(octree) \
(GrGeomOctreeCell(octree) |= GrGeomOctreeCellLeaf)
#define GrGeomOctreeClearCellLeaf(octree) \
(GrGeomOctreeCell(octree) &= ~GrGeomOctreeCellLeaf)

/* Test cell entries */
#define GrGeomOctreeCellIs(octree, cell_value) \
((GrGeomOctreeCell(octree) & cell_value) == cell_value)
#define GrGeomOctreeCellIsEmpty(octree) \
GrGeomOctreeCellIs(octree, GrGeomOctreeCellEmpty)
#define GrGeomOctreeCellIsOutside(octree) \
GrGeomOctreeCellIs(octree, GrGeomOctreeCellOutside)
#define GrGeomOctreeCellIsInside(octree) \
GrGeomOctreeCellIs(octree, GrGeomOctreeCellInside)
#define GrGeomOctreeCellIsFull(octree) \
GrGeomOctreeCellIs(octree, GrGeomOctreeCellFull)
#define GrGeomOctreeCellIsLeaf(octree) \
GrGeomOctreeCellIs(octree, GrGeomOctreeCellLeaf)

/*--------------------------------------------------------------------------
 * GrGeomOctree macro functions for faces
 *--------------------------------------------------------------------------*/

/* Face representation conversion macros */
#define GrGeomOctreeFaceIndex(i, j, k) \
(int) (( (i*(2*i+1)) + (j*(6*j+1)) + (k*(10*k+1)) ) / 2)
#define GrGeomOctreeFaceValue(face_index) \
(((unsigned char) 0x01) << face_index)

/* Set face indicators */
#define GrGeomOctreeSetFace(octree, face_index) \
(GrGeomOctreeFaces(octree) |= GrGeomOctreeFaceValue(face_index))
#define GrGeomOctreeClearFace(octree, face_index) \
(GrGeomOctreeFaces(octree) &= ~GrGeomOctreeFaceValue(face_index))

/* Test face entries */
#define GrGeomOctreeHasFaces(octree) \
(GrGeomOctreeFaces(octree) != ((unsigned char) 0x00))
#define GrGeomOctreeHasNoFaces(octree) \
(GrGeomOctreeFaces(octree) == ((unsigned char) 0x00))
#define GrGeomOctreeHasFace(octree, face_index) \
((GrGeomOctreeFaces(octree) & GrGeomOctreeFaceValue(face_index)) != \
 ((unsigned char) 0x00))
#define GrGeomOctreeHasNoFace(octree, face_index) \
((GrGeomOctreeFaces(octree) & GrGeomOctreeFaceValue(face_index)) == \
 ((unsigned char) 0x00))

/*--------------------------------------------------------------------------
 * GrGeomOctree macro functions for children
 *--------------------------------------------------------------------------*/

/* Test children entries */
#define GrGeomOctreeHasChildren(octree) \
(GrGeomOctreeChildren(octree) != NULL)
#define GrGeomOctreeHasNoChildren(octree) \
(GrGeomOctreeChildren(octree) == NULL)
#define GrGeomOctreeHasChild(octree,i) \
(GrGeomOctreeChild(octree,i) != NULL)
#define GrGeomOctreeHasNoChild(octree,i) \
(GrGeomOctreeChild(octree,i) == NULL)


/*==========================================================================
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * GrGeomOctree looping macro:
 *   Generic macro.
 *
 * NOTE: Do not call directly!
 *--------------------------------------------------------------------------*/

#define GrGeomOctreeLoop(i, j, k, l, node, octree, level, value_test,\
			 level_body, leaf_body)\
{\
   int            PV_level = level;\
   unsigned int   PV_inc;\
   int           *PV_visiting;\
   int            PV_visit_child;\
\
\
   node  = octree;\
\
   l = 0;\
   PV_inc = 1 << PV_level;\
   PV_visiting = ctalloc(int, PV_level+2);\
   PV_visiting++;\
   PV_visiting[0] = 0;\
\
   while (l >= 0)\
   {\
      /* if at the level of interest */\
      if (l == PV_level)\
      {\
	 if (value_test)\
	    level_body;\
\
	 PV_visit_child = FALSE;\
      }\
\
      /* if this is a leaf node */\
      else if (GrGeomOctreeCellIsLeaf(node))\
      {\
	 if (value_test)\
	    leaf_body;\
\
	 PV_visit_child = FALSE;\
      }\
\
      /* have I visited all of the children? */\
      else if (PV_visiting[l] < GrGeomOctreeNumChildren)\
	 PV_visit_child = TRUE;\
      else\
	 PV_visit_child = FALSE;\
\
      /* visit either a child or the parent node */\
      if (PV_visit_child)\
      {\
	 node = GrGeomOctreeChild(node, PV_visiting[l]);\
	 PV_inc = PV_inc >> 1;\
	 i += PV_inc * ((PV_visiting[l] & 1) ? 1 : 0);\
	 j += PV_inc * ((PV_visiting[l] & 2) ? 1 : 0);\
	 k += PV_inc * ((PV_visiting[l] & 4) ? 1 : 0);\
	 l++;\
	 PV_visiting[l] = 0;\
      }\
      else\
      {\
	 l--;\
	 i -= PV_inc * ((PV_visiting[l] & 1) ? 1 : 0);\
	 j -= PV_inc * ((PV_visiting[l] & 2) ? 1 : 0);\
	 k -= PV_inc * ((PV_visiting[l] & 4) ? 1 : 0);\
	 PV_inc = PV_inc << 1;\
	 node = GrGeomOctreeParent(node);\
	 PV_visiting[l]++;\
      }\
   }\
\
   tfree(PV_visiting-1);\
}

/*--------------------------------------------------------------------------
 * GrGeomOctree looping macro:
 *   Generic macro for looping over cell nodes.
 *--------------------------------------------------------------------------*/

#define GrGeomOctreeNodeLoop(i, j, k, node, octree, level,\
			     ix, iy, iz, nx, ny, nz, value_test,\
			     body)\
{\
   int  PV_i, PV_j, PV_k, PV_l;\
   int  PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;\
\
\
   PV_i = i;\
   PV_j = j;\
   PV_k = k;\
\
   GrGeomOctreeLoop(PV_i, PV_j, PV_k, PV_l, node, octree, level, value_test,\
		 {\
		    if ((PV_i >= ix) && (PV_i < (ix + nx)) &&\
			(PV_j >= iy) && (PV_j < (iy + ny)) &&\
			(PV_k >= iz) && (PV_k < (iz + nz)))\
		    {\
		       i = PV_i;\
		       j = PV_j;\
		       k = PV_k;\
		 \
		       body;\
		    }\
		 },\
		 {\
		    /* find octree and region intersection */\
		    PV_ixl = max(ix, PV_i);\
		    PV_iyl = max(iy, PV_j);\
		    PV_izl = max(iz, PV_k);\
		    PV_ixu = min((ix + nx), (PV_i + (int)PV_inc));\
		    PV_iyu = min((iy + ny), (PV_j + (int)PV_inc));\
		    PV_izu = min((iz + nz), (PV_k + (int)PV_inc));\
		 \
		    /* loop over indexes and execute the body */\
		    for (k = PV_izl; k < PV_izu; k++)\
		       for (j = PV_iyl; j < PV_iyu; j++)\
			  for (i = PV_ixl; i < PV_ixu; i++)\
			  {\
			     body;\
			  }\
		 })\
}

/*--------------------------------------------------------------------------
 * GrGeomOctree looping macro:
 *   Generic macro for looping over cell nodes with non-unitary strides.
 *--------------------------------------------------------------------------*/

#define GrGeomOctreeNodeLoop2(i, j, k, node, octree, level,\
			      ix, iy, iz, nx, ny, nz, sx, sy, sz, value_test,\
			      body)\
{\
   int  PV_i, PV_j, PV_k, PV_l;\
   int  PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;\
\
\
   PV_i = i;\
   PV_j = j;\
   PV_k = k;\
\
   GrGeomOctreeLoop(PV_i, PV_j, PV_k, PV_l, node, octree, level, value_test,\
		 {\
		    if ((PV_i >= ix) && (PV_i < (ix + nx)) &&\
			(PV_j >= iy) && (PV_j < (iy + ny)) &&\
			(PV_k >= iz) && (PV_k < (iz + nz)) &&\
			((PV_i - ix) % sx == 0) &&\
			((PV_j - iy) % sy == 0) &&\
			((PV_k - iz) % sz == 0))\
		    {\
		       i = PV_i;\
		       j = PV_j;\
		       k = PV_k;\
		 \
		       body;\
		    }\
		 },\
		 {\
		    /* find octree and region intersection */\
		    PV_ixl = max(ix, PV_i);\
		    PV_iyl = max(iy, PV_j);\
		    PV_izl = max(iz, PV_k);\
		    PV_ixu = min((ix + nx), (PV_i + PV_inc));\
		    PV_iyu = min((iy + ny), (PV_j + PV_inc));\
		    PV_izu = min((iz + nz), (PV_k + PV_inc));\
		 \
		    /* project intersection onto strided index space */\
		    PV_ixl = PV_ixl + ix;\
		    PV_ixu = PV_ixu + ix;\
		    PV_ixl = ((int) ((PV_ixl + (sx-1)) / sx)) * sx - ix;\
		    PV_ixu = ((int) ((PV_ixu + (sx-1)) / sx)) * sx - ix;\
		    PV_iyl = PV_iyl + iy;\
		    PV_iyu = PV_iyu + iy;\
		    PV_iyl = ((int) ((PV_iyl + (sy-1)) / sy)) * sy - iy;\
		    PV_iyu = ((int) ((PV_iyu + (sy-1)) / sy)) * sy - iy;\
		    PV_izl = PV_izl + iz;\
		    PV_izu = PV_izu + iz;\
		    PV_izl = ((int) ((PV_izl + (sz-1)) / sz)) * sz - iz;\
		    PV_izu = ((int) ((PV_izu + (sz-1)) / sz)) * sz - iz;\
		 \
		    /* loop over indexes and execute the body */\
		    for (k = PV_izl; k < PV_izu; k += sz)\
		       for (j = PV_iyl; j < PV_iyu; j += sy)\
			  for (i = PV_ixl; i < PV_ixu; i += sx)\
			  {\
			     body;\
			  }\
		 })\
}

/*--------------------------------------------------------------------------
 * GrGeomOctree looping macro:
 *   Macro for looping over cell faces.
 *--------------------------------------------------------------------------*/

#define GrGeomOctreeFaceLoop(i, j, k, fdir, node, octree, level,\
			     ix, iy, iz, nx, ny, nz, body)\
{\
   int  PV_f;\
   int  PV_fdir[3];\
\
\
   fdir = PV_fdir;\
   GrGeomOctreeNodeLoop(i, j, k, node, octree, level,\
			ix, iy, iz, nx, ny, nz,\
			(GrGeomOctreeCellIsInside(node)),\
		     {\
			for (PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)\
			   if (GrGeomOctreeHasFace(node, PV_f))\
			   {\
			      switch(PV_f)\
			      {\
			      case GrGeomOctreeFaceL:\
				 fdir[0] = -1; fdir[1] =  0; fdir[2] =  0;\
				 break;\
			      case GrGeomOctreeFaceR:\
				 fdir[0] =  1; fdir[1] =  0; fdir[2] =  0;\
				 break;\
			      case GrGeomOctreeFaceD:\
				 fdir[0] =  0; fdir[1] = -1; fdir[2] =  0;\
				 break;\
			      case GrGeomOctreeFaceU:\
				 fdir[0] =  0; fdir[1] =  1; fdir[2] =  0;\
				 break;\
			      case GrGeomOctreeFaceB:\
				 fdir[0] =  0; fdir[1] =  0; fdir[2] = -1;\
				 break;\
			      case GrGeomOctreeFaceF:\
				 fdir[0] =  0; fdir[1] =  0; fdir[2] =  1;\
				 break;\
			      }\
\
			      body;\
			   }\
		     })\
}
      
/*==========================================================================
 *==========================================================================*/

#endif
