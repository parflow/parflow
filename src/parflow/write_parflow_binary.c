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
 * Routines to write a Vector to a file in full or scattered form.
 *
 *****************************************************************************/

#include <math.h>

#include "parflow.h"

long SizeofPFBinarySubvector(subvector, subgrid)
Subvector *subvector;
Subgrid   *subgrid;
{
   int ix = SubgridIX(subgrid);
   int iy = SubgridIY(subgrid);
   int iz = SubgridIZ(subgrid);

   int             nx = SubgridNX(subgrid);
   int             ny = SubgridNY(subgrid);
   int             nz = SubgridNZ(subgrid);

   int             nx_v = SubvectorNX(subvector);
   int             ny_v = SubvectorNY(subvector);

   int            i, j, k, ai;

   long size;

   size = 9*amps_SizeofInt;

   ai = 0;
   BoxLoopI1(i,j,k,
	     ix,iy,iz,nx,ny,nz,
	     ai,nx_v,ny_v,nz_v,1,1,1,
	     {
		size += amps_SizeofDouble;
	     });

   return size;
}


void       WritePFBinary_Subvector(file, subvector, subgrid)
amps_File  file;
Subvector *subvector;
Subgrid   *subgrid;
{
   int             ix = SubgridIX(subgrid);
   int             iy = SubgridIY(subgrid);
   int             iz = SubgridIZ(subgrid);

   int             nx = SubgridNX(subgrid);
   int             ny = SubgridNY(subgrid);
   int             nz = SubgridNZ(subgrid);

   int             nx_v = SubvectorNX(subvector);
   int             ny_v = SubvectorNY(subvector);

   int            i, j, k, ai;
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
      this is way to slow we need better boxloops that operate
      on arrays */

   ai = 0;
   BoxLoopI1(i,j,k,
	     ix,iy,iz,nx,ny,nz,
	     ai,nx_v,ny_v,nz_v,1,1,1,
	     {
		amps_WriteDouble(file, &data[ai], 1);
	     });
}


void     WritePFBinary(file_prefix, file_suffix, v)
char    *file_prefix;
char    *file_suffix;
Vector  *v;
{
   Grid           *grid     = VectorGrid(v);
   SubgridArray   *subgrids = GridSubgrids(grid);
   Subgrid        *subgrid;
   Subvector      *subvector;

   int             g;
   int             p, P;

   long            size;

   char            file_extn[7] = "pfb";
   char            filename[255];
   amps_File       file;

   BeginTiming(PFBTimingIndex);

   p = amps_Rank(amps_CommWorld);
   P = amps_Size(amps_CommWorld);

   if ( p == 0 )
      size = 6*amps_SizeofDouble + 4*amps_SizeofInt;
   else
      size = 0;

   ForSubgridI(g, subgrids)
   {
      subgrid   = SubgridArraySubgrid(subgrids, g);
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

   if ( p == 0 )
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
      subgrid   = SubgridArraySubgrid(subgrids, g);
      subvector = VectorSubvector(v, g);
      
      WritePFBinary_Subvector(file, subvector, subgrid);
   }

   amps_FFclose(file);

   EndTiming(PFBTimingIndex);
}

long SizeofPFSBinarySubvector(subvector, subgrid, drop_tolerance)
Subvector *subvector;
Subgrid   *subgrid;
double     drop_tolerance;
{
   int             ix = SubgridIX(subgrid);
   int             iy = SubgridIY(subgrid);
   int             iz = SubgridIZ(subgrid);

   int             nx = SubgridNX(subgrid);
   int             ny = SubgridNY(subgrid);
   int             nz = SubgridNZ(subgrid);

   int             nx_v = SubvectorNX(subvector);
   int             ny_v = SubvectorNY(subvector);

   int            i, j, k, ai, n;
   double         *data;

   long size;

   size = 9*amps_SizeofInt;

   data = SubvectorElt(subvector, ix, iy, iz);

   ai = 0; n = 0;
   BoxLoopI1(i,j,k,
	     ix,iy,iz,nx,ny,nz,
	     ai,nx_v,ny_v,nz_v,1,1,1,
	     {
		if ( fabs(data[ai]) > drop_tolerance ) {n++;}
	     });
	 
   size += amps_SizeofInt + (n * (3*amps_SizeofInt + amps_SizeofDouble));

   return size;
}


void       WritePFSBinary_Subvector(file, subvector, subgrid, drop_tolerance)
amps_File  file;
Subvector *subvector;
Subgrid   *subgrid;
double     drop_tolerance;
{
   int             ix = SubgridIX(subgrid);
   int             iy = SubgridIY(subgrid);
   int             iz = SubgridIZ(subgrid);

   int             nx = SubgridNX(subgrid);
   int             ny = SubgridNY(subgrid);
   int             nz = SubgridNZ(subgrid);

   int             nx_v = SubvectorNX(subvector);
   int             ny_v = SubvectorNY(subvector);

   int            i, j, k, ai, n;
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
   BoxLoopI1(i,j,k,
	     ix,iy,iz,nx,ny,nz,
	     ai,nx_v,ny_v,nz_v,1,1,1,
	     {
		if ( fabs(data[ai]) > drop_tolerance ) {n++;}
	     });
	 
   amps_WriteInt(file, &n, 1);

   ai = 0;
   BoxLoopI1(i,j,k,
	     ix,iy,iz,nx,ny,nz,
	     ai,nx_v,ny_v,nz_v,1,1,1,
	     {
		if ( fabs(data[ai]) > drop_tolerance )
		{
		   amps_WriteInt(file, &i, 1);
		   amps_WriteInt(file, &j, 1);
		   amps_WriteInt(file, &k, 1);
		   amps_WriteDouble(file, &data[ai], 1);
		}
	     });
}


void     WritePFSBinary(file_prefix, file_suffix, v, drop_tolerance)
char    *file_prefix;
char    *file_suffix;
Vector  *v;
double   drop_tolerance;
{
   Grid           *grid     = VectorGrid(v);
   SubgridArray   *subgrids = GridSubgrids(grid);
   Subgrid        *subgrid;
   Subvector      *subvector;

   int             g;
   int             p, P;

   char            file_extn[7] = "pfsb";
   char            filename[255];
   amps_File       file;
   
   long size;

   BeginTiming(PFSBTimingIndex);

   p = amps_Rank(amps_CommWorld);
   P = amps_Size(amps_CommWorld);

   if ( p == 0 )
      size = 6*amps_SizeofDouble + 4*amps_SizeofInt;
   else
      size = 0;

   ForSubgridI(g, subgrids)
   {
      subgrid   = SubgridArraySubgrid(subgrids, g);
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

   if ( p == 0 )
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
      subgrid   = SubgridArraySubgrid(subgrids, g);
      subvector = VectorSubvector(v, g);
      
      WritePFSBinary_Subvector(file, subvector, subgrid, drop_tolerance);
   }

   amps_FFclose(file);

   EndTiming(PFSBTimingIndex);
}
