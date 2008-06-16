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
 * The main routine
 *
 *****************************************************************************/

#include "parflow.h"

#ifdef CEGDB

#include <cegdb.h>

#endif

int             main(argc,argv)
char *argv[];
int  argc;
{
   FILE *file;
      
   char filename[MAXPATHLEN];

   amps_Clock_t wall_clock_time;
   /*-----------------------------------------------------------------------
    * Initialize AMPS 
    *-----------------------------------------------------------------------*/

   if (amps_Init(&argc, &argv))
   {
      amps_Printf("Error: initalization failed\n");
      exit(1);
   }

#ifdef CEGDB
   cegdb(&argc, &argv, amps_Rank(MPI_CommWorld));
#endif
   
   wall_clock_time = amps_Clock();

   /*-----------------------------------------------------------------------
    * Set up globals structure
    *-----------------------------------------------------------------------*/

   NewGlobals(argv[1]);

   /*-----------------------------------------------------------------------
    * Read the Users Input Deck
    *-----------------------------------------------------------------------*/

   amps_ThreadLocal(input_database) = IDB_NewDB(GlobalsInFileName);

   /*-----------------------------------------------------------------------
    * Setup log printing
    *-----------------------------------------------------------------------*/

   NewLogging();

   /*-----------------------------------------------------------------------
    * Setup timing table
    *-----------------------------------------------------------------------*/

   NewTiming();

   /*-----------------------------------------------------------------------
    * Solve the problem
    *-----------------------------------------------------------------------*/
   Solve();
   printf("Problem solved \n");
   fflush(NULL);

   /*-----------------------------------------------------------------------
    * Log global information
    *-----------------------------------------------------------------------*/

   LogGlobals();

   /*-----------------------------------------------------------------------
    * Print timing results
    *-----------------------------------------------------------------------*/

   PrintTiming();

   /*-----------------------------------------------------------------------
    * Clean up
    *-----------------------------------------------------------------------*/

   FreeLogging();

   FreeTiming();

   /*-----------------------------------------------------------------------
    * Finalize AMPS and exit
    *-----------------------------------------------------------------------*/

   wall_clock_time = amps_Clock() - wall_clock_time;

   if(!amps_Rank(amps_CommWorld))
      IfLogging(0) 
      {
	 
	 FILE *log_file;

	 log_file = OpenLogFile("ParFlow Total Time");
	 
	 fprintf(log_file, "Total Run Time: %f seconds\n", 
		      (double)wall_clock_time/(double)AMPS_TICKS_PER_SEC);
	 
	 CloseLogFile(log_file);
      }

   if(!amps_Rank(amps_CommWorld))
   {
      sprintf(filename, "%s.%s", GlobalsOutFileName, "pftcl");
      file = fopen(filename, "w" );
      
      IDB_PrintUsage(file, amps_ThreadLocal(input_database));
      
      fclose(file);
   }
      
   IDB_FreeDB(amps_ThreadLocal(input_database));

      
   FreeGlobals();

   amps_Finalize();

   return 0;
}

