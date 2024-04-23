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
#include <string.h>

/*--------------------------------------------------------------------------
 * gms_ReadTINs
 *--------------------------------------------------------------------------*/

void           gms_ReadTINs(
                            gms_TIN ***TINs_ptr,
                            int *      nTINs_ptr,
                            char *     filename)
{
  FILE         *file;
  gms_CardType card_type;

  gms_TIN     **TINs = NULL;
  gms_TIN     **tmp_TINs;
  int nTINs;

  Vertex      **vertices;
  int nvertices;

  Triangle    **triangles;
  int ntriangles;

  int v0, v1, v2;

  int tmp_index;

  int T, v, t;


  /*-----------------------------------------------------------------------
   * Read in the gms TIN files
   *-----------------------------------------------------------------------*/

  nTINs = 0;

  /* open the input file */
  file = fopen(filename, "r");

  /* Check that the input file is a gms TIN file */
  fscanf(file, "%s", card_type);
  if (strncmp(card_type, "TIN", 3))
  {
    printf("%s is not a gms TIN file\n", filename);
    exit(1);
  }

  while (fscanf(file, "%s", card_type) != EOF)
  {
    /*---------------------------------------------------
     * BEGS card type
     *---------------------------------------------------*/

    if (!strncmp(card_type, "BEGT", 4))
    {
      /* allocate new space */
      tmp_TINs = TINs;
      TINs = ctalloc(gms_TIN *, (nTINs + 1));
      for (T = 0; T < nTINs; T++)
        TINs[T] = tmp_TINs[T];
      TINs[nTINs] = ctalloc(gms_TIN, 1);
      tfree(tmp_TINs);
    }

    /*---------------------------------------------------
     * ENDS card type
     *---------------------------------------------------*/

    else if (!strncmp(card_type, "ENDT", 4))
    {
      nTINs++;
    }

    /*---------------------------------------------------
     * SNAM card type
     *---------------------------------------------------*/

    else if (!strncmp(card_type, "TNAM", 4))
    {
      /* read in the TIN name */
      fscanf(file, "%s", (TINs[nTINs]->TIN_name));
    }

    /*---------------------------------------------------
     * MAT card type
     *---------------------------------------------------*/

    else if (!strncmp(card_type, "MAT", 3))
    {
      /* read in the TIN material id */
      fscanf(file, "%d", &(TINs[nTINs]->mat_id));
    }

    /*---------------------------------------------------
     * VERT card type
     *---------------------------------------------------*/

    else if (!strncmp(card_type, "VERT", 4))
    {
      /* read in the number of vertices */
      fscanf(file, "%d", &nvertices);

      /* read in the vertices */
      vertices = ctalloc(Vertex *, nvertices);
      for (v = 0; v < nvertices; v++)
      {
        vertices[v] = ctalloc(Vertex, 1);
        fscanf(file, "%le%le%le%d",
               &(vertices[v]->x),
               &(vertices[v]->y),
               &(vertices[v]->z),
               &tmp_index);
      }

      (TINs[nTINs]->vertices) = vertices;
      (TINs[nTINs]->nvertices) = nvertices;
    }

    /*---------------------------------------------------
     * TRI card type
     *---------------------------------------------------*/

    else if (!strncmp(card_type, "TRI", 3))
    {
      /* read in the number of triangles */
      fscanf(file, "%d", &ntriangles);

      /* read in the triangles */
      triangles = ctalloc(Triangle *, ntriangles);
      for (t = 0; t < ntriangles; t++)
      {
        triangles[t] = ctalloc(Triangle, 1);
        fscanf(file, "%d%d%d", &v0, &v1, &v2);
        (triangles[t]->v0) = v0 - 1;
        (triangles[t]->v1) = v1 - 1;
        (triangles[t]->v2) = v2 - 1;
      }

      (TINs[nTINs]->triangles) = triangles;
      (TINs[nTINs]->ntriangles) = ntriangles;
    }
  }

  /* close the input file */
  fclose(file);

  *TINs_ptr = TINs;
  *nTINs_ptr = nTINs;
}


