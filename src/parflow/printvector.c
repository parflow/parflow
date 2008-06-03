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
 * Routines for printing vectors to file
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * PrintSubvectorAll
 *--------------------------------------------------------------------------*/

void        PrintSubvectorAll(file, subvector)
amps_File   file;
Subvector  *subvector;
{
   int  ix, iy, iz;
   int  nx, ny, nz;
   int  i, j, k;


   ix = SubvectorIX(subvector);
   iy = SubvectorIY(subvector);
   iz = SubvectorIZ(subvector);

   nx = SubvectorNX(subvector);
   ny = SubvectorNY(subvector);
   nz = SubvectorNZ(subvector);

   amps_Fprintf(file, "\t\tPosition(%d,%d,%d), Size (%d,%d,%d)\n",
	       ix, iy, iz, nx, ny, nz);

   for(k = iz; k < iz + nz; k++)
      for(j = iy; j < iy + ny; j++)
	 for(i = ix; i < ix + nx; i++)
	    amps_Fprintf(file, "\t\t(%d,%d,%d): %f\n", i, j, k, 
			 *SubvectorElt(subvector, i, j, k));
}


/*--------------------------------------------------------------------------
 * PrintVectorAll
 *--------------------------------------------------------------------------*/

void     PrintVectorAll(filename, v)
char    *filename;
Vector  *v;
{
   amps_File  file;

   int   g;
   Grid *grid;


   if ((file = amps_Fopen(filename, "a")) == NULL)
   {
      amps_Printf("Error: can't open output file %s\n", filename);
      exit(1);
   }

   amps_Fprintf(file, "===================================================\n");

   grid = VectorGrid(v);

   for(g = 0; g < GridNumSubgrids(grid); g++)
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

void        PrintSubvector(file, subvector, subgrid)
amps_File   file;
Subvector  *subvector;
Subgrid    *subgrid;
{
   int  ix, iy, iz;
   int  nx, ny, nz;
   int  i, j, k;


   ix = SubgridIX(subgrid);
   iy = SubgridIY(subgrid);
   iz = SubgridIZ(subgrid);

   nx = SubgridNX(subgrid);
   ny = SubgridNY(subgrid);
   nz = SubgridNZ(subgrid);

   amps_Fprintf(file, "\t\tPosition(%d,%d,%d), Size (%d,%d,%d)\n",
		ix, iy, iz, nx, ny, nz);

   for(k = iz; k < iz + nz; k++)
      for(j = iy; j < iy + ny; j++)
	 for(i = ix; i < ix + nx; i++)
	    amps_Fprintf(file, "\t\t(%d,%d,%d): %f\n", i, j, k, 
			 *SubvectorElt(subvector, i, j, k));
}


/*--------------------------------------------------------------------------
 * PrintVector
 *--------------------------------------------------------------------------*/

void     PrintVector(filename, v)
char    *filename;
Vector  *v;
{
   amps_File  file;

   int   g;
   Grid *grid;


   if ((file = amps_Fopen(filename, "a")) == NULL)
   {
      amps_Printf("Error: can't open output file %s\n", filename);
      exit(1);
   }

   amps_Fprintf(file, "===================================================\n");

   grid = VectorGrid(v);

   for(g = 0; g < GridNumSubgrids(grid); g++)
   {
      amps_Fprintf(file, "Subvector Number: %d\n", g);

      PrintSubvector(file, VectorSubvector(v, g), GridSubgrid(grid, g));
   }

   amps_Fprintf(file, "===================================================\n");

   fflush(file);
   amps_Fclose(file);
}
