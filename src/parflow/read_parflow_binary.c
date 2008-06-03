/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Routines to read a Vector from a distributed file.
 *
 *****************************************************************************/

#include <math.h>
#include <string.h>

#include "parflow.h"

void ReadPFBinary_Subvector(file, subvector, subgrid)
amps_File       file;
Subvector      *subvector;
Subgrid        *subgrid;
{
   int             ix, iy, iz;
   int             nx, ny, nz;
   int             rx, ry, rz;

   int             nx_v = SubvectorNX(subvector);
   int             ny_v = SubvectorNY(subvector);

   int             i, j, k, ai;
   double         *data;

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


void ReadPFBinary(filename, v)
char           *filename;
Vector         *v;
{
   Grid           *grid     = VectorGrid(v);
   SubgridArray   *subgrids = GridSubgrids(grid);
   Subgrid        *subgrid;
   Subvector      *subvector;

   int             num_chars, g;
   int             p, P;

   amps_File       file;

   double          X, Y, Z;
   int             NX, NY, NZ;
   double          DX, DY, DZ;

   BeginTiming(PFBTimingIndex);

   p = amps_Rank(amps_CommWorld);
   P = amps_Size(amps_CommWorld);

   if ( ((num_chars = strlen(filename)) < 4) ||
	(strcmp(".pfb", &filename[num_chars - 4])) )
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
      subgrid   = SubgridArraySubgrid(subgrids, g);
      subvector = VectorSubvector(v, g);
      ReadPFBinary_Subvector(file, subvector, subgrid);
   }

   amps_FFclose(file);

   EndTiming(PFBTimingIndex);
}
