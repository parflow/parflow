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
 * Print the grid structure to a file
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * PrintGrid
 *--------------------------------------------------------------------------*/

void   PrintGrid(filename, grid)
char  *filename;
Grid  *grid;
{
   amps_File  file;

   SubgridArray  *s_a;
   Subgrid       *s;
   int            i;


   if ((file = amps_Fopen(filename, "a")) == NULL)
   {
      amps_Printf("Error: can't open output file %s\n", filename);
      exit(1);
   }

   amps_Fprintf(file, "=================================================\n");

   /* print background grid info */
   amps_Fprintf(file, "Background Grid:\n");
   amps_Fprintf(file, "( X,  Y,  Z) = (%f, %f, %f)\n",
		BackgroundX(GlobalsBackground),
		BackgroundY(GlobalsBackground),
		BackgroundZ(GlobalsBackground));
   amps_Fprintf(file, "(NX, NY, NZ) = (%d, %d, %d)\n",
		BackgroundNX(GlobalsBackground),
		BackgroundNY(GlobalsBackground),
		BackgroundNZ(GlobalsBackground));
   amps_Fprintf(file, "(DX, DY, DZ) = (%f, %f, %f)\n",
		BackgroundDX(GlobalsBackground),
		BackgroundDY(GlobalsBackground),
		BackgroundDZ(GlobalsBackground));

   /* print subgrids */
   ForSubgridI(i, (s_a = GridSubgrids(grid)))
   {
      s = SubgridArraySubgrid(s_a, i);

      amps_Fprintf(file, "Subgrids(%d):\n", i);
      amps_Fprintf(file, "( x,  y,  z) = (%d, %d, %d)\n",
		   SubgridIX(s), SubgridIY(s), SubgridIZ(s));
      amps_Fprintf(file, "(nx, ny, nz) = (%d, %d, %d)\n",
		   SubgridNX(s), SubgridNY(s), SubgridNZ(s));

      amps_Fprintf(file, "(rx, ry, rz) = (%d, %d, %d)\n",
		   SubgridRX(s), SubgridRY(s), SubgridRZ(s));
      amps_Fprintf(file, "process = %d\n", SubgridProcess(s));
   }

   /* print all_subgrids */
   ForSubgridI(i, (s_a = GridAllSubgrids(grid)))
   {
      s = SubgridArraySubgrid(s_a, i);

      amps_Fprintf(file, "AllSubgrids(%d):\n", i);
      amps_Fprintf(file, "( x,  y,  z) = (%d, %d, %d)\n",
		   SubgridIX(s), SubgridIY(s), SubgridIZ(s));
      amps_Fprintf(file, "(nx, ny, nz) = (%d, %d, %d)\n",
		   SubgridNX(s), SubgridNY(s), SubgridNZ(s));
      amps_Fprintf(file, "(rx, ry, rz) = (%d, %d, %d)\n",
		   SubgridRX(s), SubgridRY(s), SubgridRZ(s));
      amps_Fprintf(file, "process = %d\n", SubgridProcess(s));
   }

   amps_Fprintf(file, "=================================================\n");

   fflush(file);
   amps_Fclose(file);
}
