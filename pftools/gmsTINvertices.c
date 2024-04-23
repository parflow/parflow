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
 * Main routine
 *--------------------------------------------------------------------------*/

int main(
         int    argc,
         char **argv)
{
  gms_TIN     **TINs = NULL;
  gms_TIN     **new_TINs;
  gms_TIN     **tmp_TINs;
  int nTINs;
  int new_nTINs;

  Vertex      **vertices;
  int nvertices;

  int T, v, i;


  if (argc < 3)
  {
    fprintf(stderr,
            "Usage:  gmsTINvertices <TIN input files> <TIN output file>\n");
    exit(1);
  }

  /*-----------------------------------------------------------------------
   * Read in the gms TIN files
   *-----------------------------------------------------------------------*/

  nTINs = 0;
  for (i = 1; i <= (argc - 2); i++)
  {
    /* read the TINs in next input file */
    gms_ReadTINs(&new_TINs, &new_nTINs, argv[i]);

    /* add the new TINs to the TINs array */
    tmp_TINs = TINs;
    TINs = ctalloc(gms_TIN *, (nTINs + new_nTINs));
    for (T = 0; T < nTINs; T++)
      TINs[T] = tmp_TINs[T];
    for (T = 0; T < new_nTINs; T++)
      TINs[nTINs + T] = new_TINs[T];
    nTINs += new_nTINs;
    tfree(tmp_TINs);
    tfree(new_TINs);
  }

  /*-----------------------------------------------------------------------
   * Concatenate the vertices of the TINs and set the z-component to 0
   *-----------------------------------------------------------------------*/

  nvertices = 0;
  for (T = 0; T < nTINs; T++)
    nvertices += (TINs[T]->nvertices);

  vertices = ctalloc(Vertex *, nvertices);
  v = 0;
  for (T = 0; T < nTINs; T++)
    for (i = 0; i < (TINs[T]->nvertices); i++, v++)
    {
      vertices[v] = (TINs[T]->vertices[i]);
      (vertices[v]->z) = 0.0;
    }

  /*-----------------------------------------------------------------------
   * Sort the vertices (y first, then x; i.e. x varies fastest)
   *-----------------------------------------------------------------------*/

  SortXYVertices(vertices, nvertices, 0);

  /*-----------------------------------------------------------------------
   * Eliminate duplicate xy vertices
   *-----------------------------------------------------------------------*/

  i = 0;
  for (v = 0; v < nvertices; v++)
  {
    if (((vertices[v]->x) != (vertices[i]->x)) ||
        ((vertices[v]->y) != (vertices[i]->y)))
    {
      i++;
      vertices[i] = vertices[v];
    }
  }
  nvertices = (i + 1);

  /*-----------------------------------------------------------------------
   * Create the output TIN structure
   *-----------------------------------------------------------------------*/

  new_TINs = ctalloc(gms_TIN *, 1);
  new_TINs[0] = ctalloc(gms_TIN, 1);

  (new_TINs[0]->vertices) = vertices;
  (new_TINs[0]->nvertices) = nvertices;

  /*-----------------------------------------------------------------------
   * Write the output file
   *-----------------------------------------------------------------------*/

  gms_WriteTINs(new_TINs, 1, argv[argc - 1]);

  return(0);
}
