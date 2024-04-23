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
/*****************************************************************************
*
* Routines for printing vectors to file
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * PrintSubvectorAll
 *--------------------------------------------------------------------------*/

void        PrintSubvectorAll(
                              amps_File  file,
                              Subvector *subvector)
{
  int ix, iy, iz;
  int nx, ny, nz;
  int i, j, k;


  ix = SubvectorIX(subvector);
  iy = SubvectorIY(subvector);
  iz = SubvectorIZ(subvector);

  nx = SubvectorNX(subvector);
  ny = SubvectorNY(subvector);
  nz = SubvectorNZ(subvector);

  amps_Fprintf(file, "\t\tPosition(%d,%d,%d), Size (%d,%d,%d)\n",
               ix, iy, iz, nx, ny, nz);

  for (k = iz; k < iz + nz; k++)
    for (j = iy; j < iy + ny; j++)
      for (i = ix; i < ix + nx; i++)
        amps_Fprintf(file, "\t\t(%d,%d,%d): %f\n", i, j, k,
                     *SubvectorElt(subvector, i, j, k));
}


/*--------------------------------------------------------------------------
 * PrintVectorAll
 *--------------------------------------------------------------------------*/

void     PrintVectorAll(
                        char *  filename,
                        Vector *v)
{
  amps_File file;

  int g;
  Grid *grid;


  if ((file = amps_Fopen(filename, "w")) == NULL)
  {
    amps_Printf("Error: can't open output file %s\n", filename);
    exit(1);
  }

  amps_Fprintf(file, "===================================================\n");

  grid = VectorGrid(v);

  for (g = 0; g < GridNumSubgrids(grid); g++)
  {
    amps_Fprintf(file, "Subvector Number: %d\n", g);

    PrintSubvectorAll(file, VectorSubvector(v, g));
  }

  amps_Fprintf(file, "===================================================\n");

  fflush(file);
  amps_Fclose(file);
}


/*--------------------------------------------------------------------------
 * PrintSubvector
 *--------------------------------------------------------------------------*/

void        PrintSubvector(
                           amps_File  file,
                           Subvector *subvector,
                           Subgrid *  subgrid)
{
  int ix, iy, iz;
  int nx, ny, nz;
  int i, j, k;


  ix = SubgridIX(subgrid);
  iy = SubgridIY(subgrid);
  iz = SubgridIZ(subgrid);

  nx = SubgridNX(subgrid);
  ny = SubgridNY(subgrid);
  nz = SubgridNZ(subgrid);

  amps_Fprintf(file, "\t\tPosition(%d,%d,%d), Size (%d,%d,%d)\n",
               ix, iy, iz, nx, ny, nz);

  for (k = iz; k < iz + nz; k++)
    for (j = iy; j < iy + ny; j++)
      for (i = ix; i < ix + nx; i++)
        amps_Fprintf(file, "\t\t(%d,%d,%d): %f\n", i, j, k,
                     *SubvectorElt(subvector, i, j, k));
}


/*--------------------------------------------------------------------------
 * PrintVector
 *--------------------------------------------------------------------------*/

void     PrintVector(
                     char *  filename,
                     Vector *v)
{
  amps_File file;

  int g;
  Grid *grid;


  if ((file = amps_Fopen(filename, "w")) == NULL)
  {
    amps_Printf("Error: can't open output file %s\n", filename);
    exit(1);
  }

  amps_Fprintf(file, "===================================================\n");

  grid = VectorGrid(v);

  for (g = 0; g < GridNumSubgrids(grid); g++)
  {
    amps_Fprintf(file, "Subvector Number: %d\n", g);

    PrintSubvector(file, VectorSubvector(v, g), GridSubgrid(grid, g));
  }

  amps_Fprintf(file, "===================================================\n");

  fflush(file);
  amps_Fclose(file);
}
