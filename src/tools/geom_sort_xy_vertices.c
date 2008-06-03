/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1 $
 *********************************************************************EHEADER*/

#include "general.h"
#include "geometry.h"


/*--------------------------------------------------------------------------
 * Some definitions needed for QuickSort
 *--------------------------------------------------------------------------*/

typedef Vertex *EltType;

#define QSORT_CROSSOVER 10

#define Compare(result, vertex0, vertex1, compare_op)\
{\
   result = 0;\
   if ((vertex0 -> y) compare_op (vertex1 -> y))\
      result = 1;\
   else if ((vertex0 -> y) == (vertex1 -> y))\
   {\
      if ((vertex0 -> x) compare_op (vertex1 -> x))\
	 result = 1;\
   }\
}

#define CompareLessThan(result, vertex0, vertex1) \
Compare(result, vertex0, vertex1, <)

#define CompareGreaterThan(result, vertex0, vertex1) \
Compare(result, vertex0, vertex1, >)

#define Swap(array, i, j, tmp)\
{\
   tmp      = array[i];\
   array[i] = array[j];\
   array[j] = tmp;\
}


/*--------------------------------------------------------------------------
 * QuickSort
 *--------------------------------------------------------------------------*/

#include "quicksort.c"

/*--------------------------------------------------------------------------
 * SortXYVertices
 *--------------------------------------------------------------------------*/

int      *SortXYVertices(vertices, nvertices, return_permute)
Vertex  **vertices;
int       nvertices;
int       return_permute;
{
   int  *permute = NULL;
   int   i;

   if (return_permute)
   {
      permute = ctalloc(int, nvertices);
      for (i = 0; i < nvertices; i++)
	 permute[i] = i;
   }

   QuickSort(0, (nvertices-1), vertices, permute);

   return permute;
}


