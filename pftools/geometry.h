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

#ifndef _GEOMETRY_HEADER
#define _GEOMETRY_HEADER

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Structures:
 *--------------------------------------------------------------------------*/

typedef struct {
  double x, y, z;
} Vertex;

typedef struct {
  int v0, v1, v2;
} Triangle;


/*--------------------------------------------------------------------------
 * Prototypes:
 *--------------------------------------------------------------------------*/

/* geom_sort_vertices.c */
int *SortVertices(Vertex * *vertices, int nvertices, int return_permute);

/* geom_sort_xy_vertices.c */
int *SortXYVertices(Vertex * *vertices, int nvertices, int return_permute);

#ifdef __cplusplus
}
#endif

#endif
