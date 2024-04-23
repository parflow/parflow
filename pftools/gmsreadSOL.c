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
 * gms_ReadSolids
 *--------------------------------------------------------------------------*/

void           gms_ReadSolids(
                              gms_Solid ***solids_ptr,
                              int *        nsolids_ptr,
                              char *       filename)
{
  FILE         *file;
  gms_CardType card_type;

  gms_Solid   **solids = NULL;
  gms_Solid   **tmp_solids;
  int nsolids;

  Vertex      **vertices;
  int nvertices;

  Triangle    **triangles;
  int ntriangles;

  int tmp_index;

  int s, v, t;


  /*-----------------------------------------------------------------------
   * Read in the gms SOLID files
   *-----------------------------------------------------------------------*/

  nsolids = 0;

  /* open the input file */
  file = fopen(filename, "r");

  /* Check that the input file is a gms SOLID file */
  fscanf(file, "%s", card_type);
  if (strncmp(card_type, "SOLID", 5))
  {
    printf("%s is not a gms SOLID file\n", filename);
    exit(1);
  }

  while (fscanf(file, "%s", card_type) != EOF)
  {
    /*---------------------------------------------------
     * BEGS card type
     *---------------------------------------------------*/

    if (!strncmp(card_type, "BEGS", 4))
    {
      /* allocate new space */
      tmp_solids = solids;
      solids = ctalloc(gms_Solid *, (nsolids + 1));
      for (s = 0; s < nsolids; s++)
        solids[s] = tmp_solids[s];
      solids[nsolids] = ctalloc(gms_Solid, 1);
      tfree(tmp_solids);
    }

    /*---------------------------------------------------
     * ENDS card type
     *---------------------------------------------------*/

    else if (!strncmp(card_type, "ENDS", 4))
    {
      nsolids++;
    }

    /*---------------------------------------------------
     * SNAM card type
     *---------------------------------------------------*/

    else if (!strncmp(card_type, "SNAM", 4))
    {
      /* read in the solid name */
      fscanf(file, "%s", (solids[nsolids]->solid_name));
    }

    /*---------------------------------------------------
     * MAT card type
     *---------------------------------------------------*/

    else if (!strncmp(card_type, "MAT", 3))
    {
      /* read in the solid material id */
      fscanf(file, "%d", &(solids[nsolids]->mat_id));
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
        fscanf(file, "%le%le%le",
               &(vertices[v]->x),
               &(vertices[v]->y),
               &(vertices[v]->z));
      }

      (solids[nsolids]->vertices) = vertices;
      (solids[nsolids]->nvertices) = nvertices;
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
        fscanf(file, "%d%d%d%d",
               &(triangles[t]->v0),
               &(triangles[t]->v1),
               &(triangles[t]->v2),
               &tmp_index);
      }

      (solids[nsolids]->triangles) = triangles;
      (solids[nsolids]->ntriangles) = ntriangles;
    }
  }

  /* close the input file */
  fclose(file);

  *solids_ptr = solids;
  *nsolids_ptr = nsolids;
}


