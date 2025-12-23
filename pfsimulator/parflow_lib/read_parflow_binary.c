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
* Routines to read a Vector from a distributed file.
*
*****************************************************************************/
#include "parflow.h"

#include <string.h>
#include <math.h>

void ReadPFBinary_Subvector(
                            amps_File  file,
                            Subvector *subvector,
                            Subgrid *  subgrid)
{
  int ix, iy, iz;
  int nx, ny, nz;
  int rx, ry, rz;

  int nx_v = SubvectorNX(subvector);
  int ny_v = SubvectorNY(subvector);
  int nz_v = SubvectorNZ(subvector);

  int i, j, k, ai;
  double         *data;

  (void)subgrid;

  amps_ReadInt(file, &ix, 1);
  amps_ReadInt(file, &iy, 1);
  amps_ReadInt(file, &iz, 1);

  amps_ReadInt(file, &nx, 1);
  amps_ReadInt(file, &ny, 1);
  amps_ReadInt(file, &nz, 1);

  amps_ReadInt(file, &rx, 1);
  amps_ReadInt(file, &ry, 1);
  amps_ReadInt(file, &rz, 1);

  data = SubvectorElt(subvector, ix, iy, iz);

  ai = 0;
  BoxLoopI1(i, j, k,
            ix, iy, iz, nx, ny, nz,
            ai, nx_v, ny_v, nz_v, 1, 1, 1,
  {
    amps_ReadDouble(file, &data[ai], 1);
  });
}


void ReadPFBinary(
                  char *  filename,
                  Vector *v)
{
  Grid           *grid = VectorGrid(v);
  SubgridArray   *subgrids = GridSubgrids(grid);
  Subgrid        *subgrid;
  Subvector      *subvector;

  int num_chars, g;
  int p, P;

  amps_File file;

  double X, Y, Z;
  int NX, NY, NZ;
  double DX, DY, DZ;

  BeginTiming(PFBTimingIndex);

  p = amps_Rank(amps_CommWorld);
  P = amps_Size(amps_CommWorld);

  if (((num_chars = strlen(filename)) < 4) ||
      (strcmp(".pfb", &filename[num_chars - 4])))
  {
    amps_Printf("Error: %s is not in pfb format\n", filename);
    exit(1);
  }

  if ((file = amps_FFopen(amps_CommWorld, filename, "rb", 0)) == NULL)
  {
    amps_Printf("Error: can't open input file %s\n", filename);
    exit(1);
  }

  if (p == 0)
  {
    amps_ReadDouble(file, &X, 1);
    amps_ReadDouble(file, &Y, 1);
    amps_ReadDouble(file, &Z, 1);

    amps_ReadInt(file, &NX, 1);
    amps_ReadInt(file, &NY, 1);
    amps_ReadInt(file, &NZ, 1);

    amps_ReadDouble(file, &DX, 1);
    amps_ReadDouble(file, &DY, 1);
    amps_ReadDouble(file, &DZ, 1);

    amps_ReadInt(file, &P, 1);
  }

  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(v, g);
    ReadPFBinary_Subvector(file, subvector, subgrid);
  }

  amps_FFclose(file);

  EndTiming(PFBTimingIndex);
}
