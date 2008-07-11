/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.2 $
 *********************************************************************EHEADER*/

#ifndef _GEOMETRY_HEADER
#define _GEOMETRY_HEADER


/*--------------------------------------------------------------------------
 * Structures:
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  x, y, z;

} Vertex;

typedef struct
{
   int  v0, v1, v2;

} Triangle;


/*--------------------------------------------------------------------------
 * Prototypes:
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif


/* geom_sort_vertices.c */
int *SortVertices P((Vertex **vertices , int nvertices , int return_permute ));

/* geom_sort_xy_vertices.c */
int *SortXYVertices P((Vertex **vertices , int nvertices , int return_permute ));

#undef P


#endif
