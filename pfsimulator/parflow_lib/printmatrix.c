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
* Routines for printing matrices to file
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * PrintSubmatrixAll
 *--------------------------------------------------------------------------*/

void        PrintSubmatrixAll(
                              amps_File  file,
                              Submatrix *submatrix,
                              Stencil *  stencil)
{
  int ix, iy, iz;
  int nx, ny, nz;
  int sx, sy, sz;
  int i, j, k, s;

  int stencil_sz;


  ix = SubmatrixIX(submatrix);
  iy = SubmatrixIY(submatrix);
  iz = SubmatrixIZ(submatrix);

  nx = SubmatrixNX(submatrix);
  ny = SubmatrixNY(submatrix);
  nz = SubmatrixNZ(submatrix);

  sx = SubmatrixSX(submatrix);
  sy = SubmatrixSY(submatrix);
  sz = SubmatrixSZ(submatrix);

  amps_Fprintf(file, "\t\tPosition(%d,%d,%d), Size (%d,%d,%d)\n",
               ix, iy, iz, nx, ny, nz);

  stencil_sz = StencilSize(stencil);
  amps_Fprintf(file, "\t\tStencilSize (%d)\n", stencil_sz);

  for (k = iz; k < iz + sz * nz; k += sz)
    for (j = iy; j < iy + sy * ny; j += sy)
      for (i = ix; i < ix + sx * nx; i += sx)
      {
        amps_Fprintf(file, "\t\t\t(%d,%d,%d):\n", i, j, k);
        for (s = 0; s < stencil_sz; s++)
          amps_Fprintf(file, "\t\t\t\t%f\n",
                       *SubmatrixElt(submatrix, s, i, j, k));
      }
}


/*--------------------------------------------------------------------------
 * PrintMatrixAll
 *--------------------------------------------------------------------------*/

void     PrintMatrixAll(
                        char *  filename,
                        Matrix *A)
{
  amps_File file;

  int g;
  Grid *grid;


  if ((file = amps_Fopen(filename, "a")) == NULL)
  {
    amps_Printf("Error: can't open output file %s\n", filename);
    exit(1);
  }

  amps_Fprintf(file, "===================================================\n");

  grid = MatrixGrid(A);

  for (g = 0; g < GridNumSubgrids(grid); g++)
  {
    amps_Fprintf(file, "Submatrix Number: %d\n", g);

    PrintSubmatrixAll(file, MatrixSubmatrix(A, g), MatrixStencil(A));
  }

  amps_Fprintf(file, "===================================================\n");

  fflush(file);
  amps_Fclose(file);
}

/*--------------------------------------------------------------------------
 * PrintSubmatrix
 *--------------------------------------------------------------------------*/

void        PrintSubmatrix(
                           amps_File  file,
                           Submatrix *submatrix,
                           Subregion *subregion,
                           Stencil *  stencil)
{
  int ix, iy, iz;
  int nx, ny, nz;
  int sx, sy, sz;
  int i, j, k, s;

  int stencil_sz;


  ix = SubregionIX(subregion);
  iy = SubregionIY(subregion);
  iz = SubregionIZ(subregion);

  nx = SubregionNX(subregion);
  ny = SubregionNY(subregion);
  nz = SubregionNZ(subregion);

  sx = SubregionSX(subregion);
  sy = SubregionSY(subregion);
  sz = SubregionSZ(subregion);

  amps_Fprintf(file, "\t\tPosition(%d,%d,%d), Size (%d,%d,%d)\n",
               ix, iy, iz, nx, ny, nz);

  stencil_sz = StencilSize(stencil);
  amps_Fprintf(file, "\t\tStencilSize (%d)\n", stencil_sz);

  for (k = iz; k < iz + sz * nz; k += sz)
    for (j = iy; j < iy + sy * ny; j += sy)
      for (i = ix; i < ix + sx * nx; i += sx)
      {
        amps_Fprintf(file, "\t\t\t(%d,%d,%d):\n", i, j, k);
        for (s = 0; s < stencil_sz; s++)
          amps_Fprintf(file, "\t\t\t\t%f\n",
                       *SubmatrixElt(submatrix, s, i, j, k));
      }
}


/*--------------------------------------------------------------------------
 * PrintMatrix
 *--------------------------------------------------------------------------*/

void     PrintMatrix(
                     char *  filename,
                     Matrix *A)
{
  amps_File file;

  int g;
  Grid *grid;


  if ((file = amps_Fopen(filename, "a")) == NULL)
  {
    amps_Printf("Error: can't open output file %s\n", filename);
    exit(1);
  }

  amps_Fprintf(file, "===================================================\n");

  grid = MatrixGrid(A);

  for (g = 0; g < GridNumSubgrids(grid); g++)
  {
    amps_Fprintf(file, "Submatrix Number: %d\n", g);

    PrintSubmatrix(file, MatrixSubmatrix(A, g),
                   SubregionArraySubregion(MatrixRange(A), g),
                   MatrixStencil(A));
  }

  amps_Fprintf(file, "===================================================\n");

  fflush(file);
  amps_Fclose(file);
}


/*--------------------------------------------------------------------------
 * PrintSortMatrix
 *--------------------------------------------------------------------------*/

void     PrintSortMatrix(
                         char *  filename,
                         Matrix *A,
                         int     all)
{
  amps_File file;
  amps_Invoice add_invoice;

  Submatrix       *A_sub;
  double          *ap;

  SubregionArray  *domain_sra, *range_sra;
  Subregion       *subregion;

  Stencil         *stencil;

  int ix, iy, iz;
  int nx, ny, nz;
  int sx, sy, sz;
  int level;

  int domain_num_s, range_num_s;
  int is, i, j, k, istenc;


  if ((file = amps_Fopen(filename, "w")) == NULL)
  {
    amps_Printf("Error: can't open output file %s\n", filename);
    exit(1);
  }

  domain_sra = GridSubgrids(MatrixGrid(A));

  if (all)
    range_sra = MatrixDataSpace(A);
  else
    range_sra = MatrixRange(A);

  /*----------------------------------------
   * print header info
   *----------------------------------------*/

  domain_num_s = SubregionArraySize(domain_sra);
  range_num_s = SubregionArraySize(range_sra);
  add_invoice = amps_NewInvoice("%i%i", &domain_num_s, &range_num_s);
  amps_AllReduce(amps_CommWorld, add_invoice, amps_Add);
  amps_FreeInvoice(add_invoice);

  if (amps_Rank(amps_CommWorld) == 0)
  {
    amps_Fprintf(file, "a: %d\n", 0);
    stencil = MatrixStencil(A);
    amps_Fprintf(file, "b:  %d\n", StencilSize(stencil));
    for (istenc = 0; istenc < StencilSize(stencil); istenc++)
      amps_Fprintf(file, "b: %d %d  %d  % d % d % d\n",
                   0, 0, istenc,
                   StencilShape(stencil)[istenc][0],
                   StencilShape(stencil)[istenc][1],
                   StencilShape(stencil)[istenc][2]);

    amps_Fprintf(file, "c: %d\n", domain_num_s);
    amps_Fprintf(file, "e: %d\n", range_num_s);
  }

  /* print out matrix domain information */
  ForSubregionI(is, domain_sra)
  {
    subregion = SubregionArraySubregion(domain_sra, is);
    amps_Fprintf(file, "d: %d  % 3d % 3d % 3d  % 3d % 3d % 3d",
                 SubregionLevel(subregion),
                 SubregionIX(subregion),
                 SubregionIY(subregion),
                 SubregionIZ(subregion),
                 SubregionNX(subregion),
                 SubregionNY(subregion),
                 SubregionNZ(subregion));
    amps_Fprintf(file, "  % 3d % 3d % 3d  % 3d % 3d % 3d\n",
                 SubregionSX(subregion),
                 SubregionSY(subregion),
                 SubregionSZ(subregion),
                 SubregionRX(subregion),
                 SubregionRY(subregion),
                 SubregionRZ(subregion));
  }

  /* print out matrix range information */
  ForSubregionI(is, range_sra)
  {
    subregion = SubregionArraySubregion(range_sra, is);
    amps_Fprintf(file, "f: %d  % 3d % 3d % 3d  % 3d % 3d % 3d",
                 SubregionLevel(subregion),
                 SubregionIX(subregion),
                 SubregionIY(subregion),
                 SubregionIZ(subregion),
                 SubregionNX(subregion),
                 SubregionNY(subregion),
                 SubregionNZ(subregion));
    amps_Fprintf(file, "  % 3d % 3d % 3d  % 3d % 3d % 3d\n",
                 SubregionSX(subregion),
                 SubregionSY(subregion),
                 SubregionSZ(subregion),
                 SubregionRX(subregion),
                 SubregionRY(subregion),
                 SubregionRZ(subregion));
  }

  /*----------------------------------------
   * print coefficients
   *----------------------------------------*/

  ForSubregionI(is, range_sra)
  {
    A_sub = MatrixSubmatrix(A, is);

    subregion = SubregionArraySubregion(range_sra, is);

    ix = SubregionIX(subregion);
    iy = SubregionIY(subregion);
    iz = SubregionIZ(subregion);

    nx = SubregionNX(subregion);
    ny = SubregionNY(subregion);
    nz = SubregionNZ(subregion);

    sx = SubregionSX(subregion);
    sy = SubregionSY(subregion);
    sz = SubregionSZ(subregion);

    level = SubregionLevel(subregion);

    stencil = MatrixStencil(A);

    for (k = iz; k < iz + sz * nz; k += sz)
      for (j = iy; j < iy + sy * ny; j += sy)
        for (i = ix; i < ix + sx * nx; i += sx)
        {
          amps_Fprintf(file, "g: %d %d  %d  % 4d % 4d % 4d",
                       0, 0, level, k, j, i);
          for (istenc = 0; istenc < StencilSize(stencil); istenc++)
          {
            ap = SubmatrixElt(A_sub, istenc, i, j, k);
            amps_Fprintf(file, " % f", *ap);
          }
          amps_Fprintf(file, "\n");
        }
  }

  fflush(file);
  amps_Fclose(file);
}


