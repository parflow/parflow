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
 * gms_WriteTINs
 *--------------------------------------------------------------------------*/

void           gms_WriteTINs(
                             gms_TIN **TINs,
                             int       nTINs,
                             char *    filename)
{
  FILE         *file;

  Vertex      **vertices;
  int nvertices;

  Triangle    **triangles;
  int ntriangles;

  int T, v, t;


  /*-----------------------------------------------------------------------
   * Print the TIN output file
   *-----------------------------------------------------------------------*/

  /* open the output file */
  file = fopen(filename, "w");

  /* print some heading info */
  fprintf(file, "TIN\n");

  for (T = 0; T < nTINs; T++)
  {
    /* Get vertices and triangles from the TIN structure */
    vertices = (TINs[T]->vertices);
    nvertices = (TINs[T]->nvertices);
    triangles = (TINs[T]->triangles);
    ntriangles = (TINs[T]->ntriangles);

    /* print some TIN heading info */
    fprintf(file, "BEGT\n");
    if (strlen(TINs[T]->TIN_name))
      fprintf(file, "TNAM %s\n", (TINs[T]->TIN_name));
    fprintf(file, "MAT %d\n", (TINs[T]->mat_id));

    /* print out the vertices */
    if (nvertices)
    {
      fprintf(file, "VERT %d\n", nvertices);
      for (v = 0; v < nvertices; v++)
      {
        fprintf(file, "%.15e %.15e %.15e  0\n",
                (vertices[v]->x),
                (vertices[v]->y),
                (vertices[v]->z));
      }
    }

    /* print out the triangles */
    if (ntriangles)
    {
      fprintf(file, "TRI %d\n", ntriangles);
      for (t = 0; t < ntriangles; t++)
      {
        fprintf(file, "%d %d %d\n",
                (triangles[t]->v0) + 1,
                (triangles[t]->v1) + 1,
                (triangles[t]->v2) + 1);
      }
    }

    /* print some TIN closing info */
    fprintf(file, "ENDT\n");
  }

  /* close the output file */
  fclose(file);
}


