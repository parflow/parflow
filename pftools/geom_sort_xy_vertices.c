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

#include "general.h"
#include "geometry.h"


/*--------------------------------------------------------------------------
 * Some definitions needed for QuickSort
 *--------------------------------------------------------------------------*/

typedef Vertex *EltType;

#define QSORT_CROSSOVER 10

#define Compare(result, vertex0, vertex1, compare_op)       \
        {                                                   \
          result = 0;                                       \
          if ((vertex0->y)compare_op(vertex1->y))           \
          result = 1;                                       \
          else if ((vertex0->y) == (vertex1->y))            \
          {                                                 \
            if ((vertex0->x)compare_op(vertex1->x))         \
            result = 1;                                     \
          }                                                 \
        }

#define CompareLessThan(result, vertex0, vertex1) \
        Compare(result, vertex0, vertex1, <)

#define CompareGreaterThan(result, vertex0, vertex1) \
        Compare(result, vertex0, vertex1, >)

#define Swap(array, i, j, tmp)       \
        {                            \
          tmp = array[i];            \
          array[i] = array[j];       \
          array[j] = tmp;            \
        }


/*--------------------------------------------------------------------------
 * QuickSort
 *--------------------------------------------------------------------------*/

#include "quicksort.c"

/*--------------------------------------------------------------------------
 * SortXYVertices
 *--------------------------------------------------------------------------*/

int      *SortXYVertices(
                         Vertex **vertices,
                         int      nvertices,
                         int      return_permute)
{
  int  *permute = NULL;
  int i;

  if (return_permute)
  {
    permute = ctalloc(int, nvertices);
    for (i = 0; i < nvertices; i++)
      permute[i] = i;
  }

  QuickSort(0, (nvertices - 1), vertices, permute);

  return permute;
}


