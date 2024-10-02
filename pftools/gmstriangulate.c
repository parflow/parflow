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

#include "gms.h"

#include <stdio.h>

/*--------------------------------------------------------------------------
 * Compare macro
 *--------------------------------------------------------------------------*/

#define Compare(result, vertex0, vertex1)         \
        {                                         \
          result = 0;                             \
          if ((vertex0->y) < (vertex1->y))        \
          result = -1;                            \
          else if ((vertex0->y) > (vertex1->y))   \
          result = 1;                             \
          else                                    \
          {                                       \
            if ((vertex0->x) < (vertex1->x))      \
            result = -1;                          \
            else if ((vertex0->x) > (vertex1->x)) \
            result = 1;                           \
          }                                       \
        }


/*--------------------------------------------------------------------------
 * Main routine
 *--------------------------------------------------------------------------*/

int main(
         int    argc,
         char **argv)
{
  gms_TIN      *mask_TIN;
  gms_TIN     **TINs;
  int nTINs;

  Vertex      **mask_vertices;
  int mask_nvertices;
  Triangle    **mask_triangles;
  int mask_ntriangles;

  Vertex      **vertices;
  int nvertices;
  Triangle    **triangles;
  int ntriangles;

  int max_nvertices;

  int          *mask_new_to_old;
  int          *mask_old_to_tin;
  int          *old_to_new;

  int v0, v1, v2;

  int compare_result;
  int T;
  int mask_t, mask_v;
  int t, v, new_v;


  if (argc < 4)
  {
    fprintf(stderr,
            "Usage:  gmstriangulate <TIN mask> <TIN input> <TIN output>\n");
    exit(1);
  }

  /*-----------------------------------------------------------------------
   * Read in the gms TIN files
   *-----------------------------------------------------------------------*/

  /* read the <TIN mask> file */
  gms_ReadTINs(&TINs, &nTINs, argv[1]);

  /* use the first TIN as the mask */
  mask_TIN = TINs[0];
  tfree(TINs);

  /* read the <TIN input> file */
  gms_ReadTINs(&TINs, &nTINs, argv[2]);

  /*-----------------------------------------------------------------------
   * Triangulate each of the TINs
   *-----------------------------------------------------------------------*/

  /* Get vertices and triangles from mask_TIN structure */
  mask_vertices = (mask_TIN->vertices);
  mask_nvertices = (mask_TIN->nvertices);
  mask_triangles = (mask_TIN->triangles);
  mask_ntriangles = (mask_TIN->ntriangles);

  /* Sort mask_TIN vertices */
  mask_new_to_old = SortXYVertices(mask_vertices, mask_nvertices, 1);

  /* Allocate space for the mask_old_to_tin array */
  mask_old_to_tin = ctalloc(int, mask_nvertices);

  max_nvertices = 0;
  for (T = 0; T < nTINs; T++)
  {
    /* Get vertices from TIN structure */
    vertices = (TINs[T]->vertices);
    nvertices = (TINs[T]->nvertices);

    max_nvertices = max(max_nvertices, nvertices);

    /* Sort TIN vertices */
    SortXYVertices(vertices, nvertices, 0);

    /* Construct mask_old_to_tin array */
    for (mask_v = 0; mask_v < mask_nvertices; mask_v++)
      mask_old_to_tin[mask_v] = -1;
    mask_v = 0;
    v = 0;
    while ((mask_v < mask_nvertices) && (v < nvertices))
    {
      Compare(compare_result, mask_vertices[mask_v], vertices[v]);
      if (compare_result < 0)
        mask_v++;
      else if (compare_result > 0)
        v++;
      else
      {
        mask_old_to_tin[mask_new_to_old[mask_v]] = v;
        mask_v++;
        v++;
      }
    }

    /* Delete old triangulation */
    tfree(TINs[T]->triangles);

    /* Allocate the triangles array */
    triangles = ctalloc(Triangle *, mask_ntriangles);

    /* Add triangles to the TIN structure */
    t = 0;
    for (mask_t = 0; mask_t < mask_ntriangles; mask_t++)
    {
      v0 = mask_old_to_tin[(mask_triangles[mask_t]->v0)];
      v1 = mask_old_to_tin[(mask_triangles[mask_t]->v1)];
      v2 = mask_old_to_tin[(mask_triangles[mask_t]->v2)];
      if ((v0 > -1) && (v1 > -1) && (v2 > -1))
      {
        triangles[t] = ctalloc(Triangle, 1);
        (triangles[t]->v0) = v0;
        (triangles[t]->v1) = v1;
        (triangles[t]->v2) = v2;
        t++;
      }
    }
    ntriangles = t;
    (TINs[T]->triangles) = triangles;
    (TINs[T]->ntriangles) = ntriangles;
  }

  /* Free up the mask_old_to_tin array */
  tfree(mask_old_to_tin);

  /*-----------------------------------------------------------------------
   * Delete unused vertices
   *-----------------------------------------------------------------------*/

  /* Allocate space for the old_to_new array */
  old_to_new = ctalloc(int, max_nvertices);

  for (T = 0; T < nTINs; T++)
  {
    vertices = (TINs[T]->vertices);
    nvertices = (TINs[T]->nvertices);
    triangles = (TINs[T]->triangles);
    ntriangles = (TINs[T]->ntriangles);

    for (v = 0; v < nvertices; v++)
      old_to_new[v] = -1;
    for (t = 0; t < ntriangles; t++)
    {
      old_to_new[(triangles[t]->v0)] = 1;
      old_to_new[(triangles[t]->v1)] = 1;
      old_to_new[(triangles[t]->v2)] = 1;
    }
    new_v = 0;
    for (v = 0; v < nvertices; v++)
    {
      if (old_to_new[v] > -1)
      {
        vertices[new_v] = vertices[v];
        old_to_new[v] = new_v;
        new_v++;
      }
    }
    nvertices = new_v;
    (TINs[T]->nvertices) = nvertices;
    for (t = 0; t < ntriangles; t++)
    {
      (triangles[t]->v0) = old_to_new[(triangles[t]->v0)];
      (triangles[t]->v1) = old_to_new[(triangles[t]->v1)];
      (triangles[t]->v2) = old_to_new[(triangles[t]->v2)];
    }
  }

  /* Free up the old_to_new array */
  tfree(old_to_new);

  /*-----------------------------------------------------------------------
   * Write out the new TINs
   *-----------------------------------------------------------------------*/

  gms_WriteTINs(TINs, nTINs, argv[3]);

  return(0);
}
