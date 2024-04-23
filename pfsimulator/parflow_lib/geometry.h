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
* Geometry class structures and accessors
*
*****************************************************************************/

#ifndef _GEOMETRY_HEADER
#define _GEOMETRY_HEADER


/*--------------------------------------------------------------------------
 * Miscellaneous structures:
 *--------------------------------------------------------------------------*/

typedef struct {
  double x, y, z;
} GeomVertex;

typedef struct {
  int v0, v1, v2;
} GeomTriangle;

typedef struct {
  GeomVertex **vertices;
  int nV;
  int num_ptrs_to;            /* Number of pointers to this structure. */
} GeomVertexArray;

typedef struct {
  GeomVertexArray  *vertex_array;
  GeomTriangle    **triangles;
  int nT;
} GeomTIN;


/*--------------------------------------------------------------------------
 * Solid structures:
 *--------------------------------------------------------------------------*/

#define GeomTSolidType      0

typedef struct {
  GeomTIN  *surface;

  int     **patches;               /* arrays of surface triangle indices */
  int num_patches;
  int      *num_patch_triangles;
} GeomTSolid;

typedef struct {
  void  *data;
  int type;

  NameArray patches;
} GeomSolid;


/*--------------------------------------------------------------------------
 * Accessor macros:
 *--------------------------------------------------------------------------*/

#define GeomVertexX(vertex)       ((vertex)->x)
#define GeomVertexY(vertex)       ((vertex)->y)
#define GeomVertexZ(vertex)       ((vertex)->z)

#define GeomTriangleV0(triangle)  ((triangle)->v0)
#define GeomTriangleV1(triangle)  ((triangle)->v1)
#define GeomTriangleV2(triangle)  ((triangle)->v2)

#define GeomTINVertices(TIN)      ((TIN)->vertex_array->vertices)
#define GeomTINNumVertices(TIN)   ((TIN)->vertex_array->nV)
#define GeomTINTriangles(TIN)     ((TIN)->triangles)
#define GeomTINNumTriangles(TIN)  ((TIN)->nT)
#define GeomTINVertex(TIN, i)     (GeomTINVertices(TIN)[i])
#define GeomTINTriangle(TIN, i)   (GeomTINTriangles(TIN)[i])

#define GeomSolidData(solid)      ((solid)->data)
#define GeomSolidType(solid)      ((solid)->type)

#define GeomSolidPatches(solid)     ((solid)->patches)


#endif
