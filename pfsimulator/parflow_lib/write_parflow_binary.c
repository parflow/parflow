/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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
* Routines to write a Vector to a file in full or scattered form.
*
*****************************************************************************/

#include "parflow.h"

#include <math.h>

long SizeofPFBinarySubvector(
                             Subvector *subvector,
                             Subgrid *  subgrid)
{
  int ix = SubgridIX(subgrid);
  int iy = SubgridIY(subgrid);
  int iz = SubgridIZ(subgrid);

  int nx = SubgridNX(subgrid);
  int ny = SubgridNY(subgrid);
  int nz = SubgridNZ(subgrid);

  int nx_v = SubvectorNX(subvector);
  int ny_v = SubvectorNY(subvector);
  int nz_v = SubvectorNZ(subvector);

  int i, j, k, ai;
  PF_UNUSED(ai);

  long size;

  size = 9 * amps_SizeofInt;

  ai = 0;
  BoxLoopI1(i, j, k,
            ix, iy, iz, nx, ny, nz,
            ai, nx_v, ny_v, nz_v, 1, 1, 1,
  {
    size += amps_SizeofDouble;
  });

  return size;
}


void       WritePFBinary_Subvector(
                                   amps_File  file,
                                   Subvector *subvector,
                                   Subgrid *  subgrid)
{
  int ix = SubgridIX(subgrid);
  int iy = SubgridIY(subgrid);
  int iz = SubgridIZ(subgrid);

  int nx = SubgridNX(subgrid);
  int ny = SubgridNY(subgrid);
  int nz = SubgridNZ(subgrid);

  int nx_v = SubvectorNX(subvector);
  int ny_v = SubvectorNY(subvector);
  int nz_v = SubvectorNZ(subvector);

  int i, j, k, ai;
  double         *data;

  amps_WriteInt(file, &ix, 1);
  amps_WriteInt(file, &iy, 1);
  amps_WriteInt(file, &iz, 1);

  amps_WriteInt(file, &nx, 1);
  amps_WriteInt(file, &ny, 1);
  amps_WriteInt(file, &nz, 1);

  amps_WriteInt(file, &SubgridRX(subgrid), 1);
  amps_WriteInt(file, &SubgridRY(subgrid), 1);
  amps_WriteInt(file, &SubgridRZ(subgrid), 1);

  data = SubvectorElt(subvector, ix, iy, iz);

  /* SGS Note:
   *  this is way to slow we need better boxloops that operate
   *  on arrays */

  ai = 0;
  BoxLoopI1(i, j, k,
            ix, iy, iz, nx, ny, nz,
            ai, nx_v, ny_v, nz_v, 1, 1, 1,
  {
    amps_WriteDouble(file, &data[ai], 1);
  });
}


void     WritePFBinary(
                       char *  file_prefix,
                       char *  file_suffix,
                       Vector *v)
{
  Grid           *grid = VectorGrid(v);
  SubgridArray   *subgrids = GridSubgrids(grid);
  Subgrid        *subgrid;
  Subvector      *subvector;

  int g;
  int p;

  long size;

  char file_extn[7] = "pfb";
  char filename[255];
  amps_File file;

  BeginTiming(PFBTimingIndex);

  p = amps_Rank(amps_CommWorld);

  if (p == 0)
    size = 6 * amps_SizeofDouble + 4 * amps_SizeofInt;
  else
    size = 0;

  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(v, g);

    size += SizeofPFBinarySubvector(subvector, subgrid);
  }

  /* open file */
  sprintf(filename, "%s.%s.%s", file_prefix, file_suffix, file_extn);

  if ((file = amps_FFopen(amps_CommWorld, filename, "wb", size)) == NULL)
  {
    amps_Printf("Error: can't open output file %s\n", filename);
    exit(1);
  }

  /* Compute number of patches to write */
  int num_subgrids = GridNumSubgrids(grid);
  {
    amps_Invoice invoice = amps_NewInvoice("%i", &num_subgrids);

    amps_AllReduce(amps_CommWorld, invoice, amps_Add);

    amps_FreeInvoice(invoice);
  }


  if (p == 0)
  {
    amps_WriteDouble(file, &BackgroundX(GlobalsBackground), 1);
    amps_WriteDouble(file, &BackgroundY(GlobalsBackground), 1);
    amps_WriteDouble(file, &BackgroundZ(GlobalsBackground), 1);

    amps_WriteInt(file, &SubgridNX(GridBackground(grid)), 1);
    amps_WriteInt(file, &SubgridNY(GridBackground(grid)), 1);
    amps_WriteInt(file, &SubgridNZ(GridBackground(grid)), 1);

    amps_WriteDouble(file, &BackgroundDX(GlobalsBackground), 1);
    amps_WriteDouble(file, &BackgroundDY(GlobalsBackground), 1);
    amps_WriteDouble(file, &BackgroundDZ(GlobalsBackground), 1);

    amps_WriteInt(file, &num_subgrids, 1);
  }

  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(v, g);

    WritePFBinary_Subvector(file, subvector, subgrid);
  }

  amps_FFclose(file);

  EndTiming(PFBTimingIndex);
}

long SizeofPFSBinarySubvector(
                              Subvector *subvector,
                              Subgrid *  subgrid,
                              double     drop_tolerance)
{
  int ix = SubgridIX(subgrid);
  int iy = SubgridIY(subgrid);
  int iz = SubgridIZ(subgrid);

  int nx = SubgridNX(subgrid);
  int ny = SubgridNY(subgrid);
  int nz = SubgridNZ(subgrid);

  int nx_v = SubvectorNX(subvector);
  int ny_v = SubvectorNY(subvector);
  int nz_v = SubvectorNZ(subvector);

  int i, j, k, ai, n;
  double         *data;

  long size;

  size = 9 * amps_SizeofInt;

  data = SubvectorElt(subvector, ix, iy, iz);

  ai = 0; n = 0;
  BoxLoopI1(i, j, k,
            ix, iy, iz, nx, ny, nz,
            ai, nx_v, ny_v, nz_v, 1, 1, 1,
  {
    if (fabs(data[ai]) > drop_tolerance)
    {
      n++;
    }
  });

  size += amps_SizeofInt + (n * (3 * amps_SizeofInt + amps_SizeofDouble));

  return size;
}


void       WritePFSBinary_Subvector(
                                    amps_File  file,
                                    Subvector *subvector,
                                    Subgrid *  subgrid,
                                    double     drop_tolerance)
{
  int ix = SubgridIX(subgrid);
  int iy = SubgridIY(subgrid);
  int iz = SubgridIZ(subgrid);

  int nx = SubgridNX(subgrid);
  int ny = SubgridNY(subgrid);
  int nz = SubgridNZ(subgrid);

  int nx_v = SubvectorNX(subvector);
  int ny_v = SubvectorNY(subvector);
  int nz_v = SubvectorNZ(subvector);

  int i, j, k, ai, n;
  double         *data;

  amps_WriteInt(file, &ix, 1);
  amps_WriteInt(file, &iy, 1);
  amps_WriteInt(file, &iz, 1);

  amps_WriteInt(file, &nx, 1);
  amps_WriteInt(file, &ny, 1);
  amps_WriteInt(file, &nz, 1);

  amps_WriteInt(file, &SubgridRX(subgrid), 1);
  amps_WriteInt(file, &SubgridRY(subgrid), 1);
  amps_WriteInt(file, &SubgridRZ(subgrid), 1);

  data = SubvectorElt(subvector, ix, iy, iz);

  ai = 0; n = 0;
  BoxLoopI1(i, j, k,
            ix, iy, iz, nx, ny, nz,
            ai, nx_v, ny_v, nz_v, 1, 1, 1,
  {
    if (fabs(data[ai]) > drop_tolerance)
    {
      n++;
    }
  });

  amps_WriteInt(file, &n, 1);

  ai = 0;
  BoxLoopI1(i, j, k,
            ix, iy, iz, nx, ny, nz,
            ai, nx_v, ny_v, nz_v, 1, 1, 1,
  {
    if (fabs(data[ai]) > drop_tolerance)
    {
      amps_WriteInt(file, &i, 1);
      amps_WriteInt(file, &j, 1);
      amps_WriteInt(file, &k, 1);
      amps_WriteDouble(file, &data[ai], 1);
    }
  });
}

void     WritePFSBinary(
                        char *  file_prefix,
                        char *  file_suffix,
                        Vector *v,
                        double  drop_tolerance)
{
  Grid           *grid = VectorGrid(v);
  SubgridArray   *subgrids = GridSubgrids(grid);
  Subgrid        *subgrid;
  Subvector      *subvector;

  int g;
  int p, P;

  char file_extn[7] = "pfsb";
  char filename[255];
  amps_File file;

  long size;

  BeginTiming(PFSBTimingIndex);

  p = amps_Rank(amps_CommWorld);
  P = amps_Size(amps_CommWorld);

  if (p == 0)
    size = 6 * amps_SizeofDouble + 4 * amps_SizeofInt;
  else
    size = 0;

  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(v, g);

    size += SizeofPFSBinarySubvector(subvector, subgrid, drop_tolerance);
  }

  /* open file */
  sprintf(filename, "%s.%s.%s", file_prefix, file_suffix, file_extn);

  if ((file = amps_FFopen(amps_CommWorld, filename, "wb", size)) == NULL)
  {
    amps_Printf("Error: can't open output file %s\n", filename);
    exit(1);
  }

  if (p == 0)
  {
    amps_WriteDouble(file, &BackgroundX(GlobalsBackground), 1);
    amps_WriteDouble(file, &BackgroundY(GlobalsBackground), 1);
    amps_WriteDouble(file, &BackgroundZ(GlobalsBackground), 1);

    amps_WriteInt(file, &BackgroundNX(GlobalsBackground), 1);
    amps_WriteInt(file, &BackgroundNY(GlobalsBackground), 1);
    amps_WriteInt(file, &BackgroundNZ(GlobalsBackground), 1);

    amps_WriteDouble(file, &BackgroundDX(GlobalsBackground), 1);
    amps_WriteDouble(file, &BackgroundDY(GlobalsBackground), 1);
    amps_WriteDouble(file, &BackgroundDZ(GlobalsBackground), 1);

    amps_WriteInt(file, &P, 1);
  }

  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(v, g);

    WritePFSBinary_Subvector(file, subvector, subgrid, drop_tolerance);
  }

  amps_FFclose(file);

  EndTiming(PFSBTimingIndex);
}
