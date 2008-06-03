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

#ifndef _GEOMETRY_HEADER
#define _GEOMETRY_HEADER


/*--------------------------------------------------------------------------
 * Miscellaneous structures:
 *--------------------------------------------------------------------------*/

typedef struct
{
   double x, y, z;

} GeomVertex;

typedef struct
{
   int  v0, v1, v2;

} GeomTriangle;

typedef struct
{
   GeomVertex **vertices;
   int          nV;
   int          num_ptrs_to;  /* Number of pointers to this structure. */

} GeomVertexArray;

typedef struct
{
   GeomVertexArray  *vertex_array;
   GeomTriangle    **triangles;
   int	             nT;

} GeomTIN;


/*--------------------------------------------------------------------------
 * Solid structures:
 *--------------------------------------------------------------------------*/

#define GeomTSolidType      0

typedef struct
{
   GeomTIN  *surface;

   int     **patches;              /* arrays of surface triangle indices */
   int       num_patches;
   int      *num_patch_triangles;

} GeomTSolid;

typedef struct
{
   void  *data;
   int    type;

   NameArray patches;

} GeomSolid;


/*--------------------------------------------------------------------------
 * Accessor macros:
 *--------------------------------------------------------------------------*/

#define GeomVertexX(vertex)       ((vertex) -> x)
#define GeomVertexY(vertex)       ((vertex) -> y)
#define GeomVertexZ(vertex)       ((vertex) -> z)

#define GeomTriangleV0(triangle)  ((triangle) -> v0)
#define GeomTriangleV1(triangle)  ((triangle) -> v1)
#define GeomTriangleV2(triangle)  ((triangle) -> v2)

#define GeomTINVertices(TIN)      ((TIN) -> vertex_array -> vertices)
#define GeomTINNumVertices(TIN)   ((TIN) -> vertex_array -> nV)
#define GeomTINTriangles(TIN)     ((TIN) -> triangles)
#define GeomTINNumTriangles(TIN)  ((TIN) -> nT)
#define GeomTINVertex(TIN, i)     (GeomTINVertices(TIN)[i])
#define GeomTINTriangle(TIN, i)   (GeomTINTriangles(TIN)[i])

#define GeomSolidData(solid)      ((solid) -> data)
#define GeomSolidType(solid)      ((solid) -> type)

#define GeomSolidPatches(solid)     ((solid) -> patches)


#endif
