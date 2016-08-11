#include <stdio.h>
#include <string.h>
#include <parflow.h>
#ifdef HAVE_P4EST
#include "parflow_p4est.h"
#include <p4est.h>
#include <p8est.h>
#endif

int
main(int argc, char **argv)
{
    char *key_val;
    char filename[MAXPATHLEN];
    FILE *file = NULL;
    FILE *log_file = NULL;
    amps_Clock_t wall_clock_time;

    if (amps_Init(&argc, &argv)) {
        amps_Printf("Error: amps_Init initalization failed\n");
        exit(1);
    }

    wall_clock_time = amps_Clock();
    
    if(argc != 2) {
       amps_Printf("Error: invalid number of arguments.\n");
       amps_Printf("       Invoke as parflow <pfidb file>.\n");
       exit(1);
    }

    /*-----------------------------------------------------------------------
     * Set up globals structure
     *-----------------------------------------------------------------------*/

    NewGlobals(argv[1]);

    /*-----------------------------------------------------------------------
     * Read the Users Input Deck
     *-----------------------------------------------------------------------*/

    amps_ThreadLocal(input_database) = IDB_NewDB(GlobalsInFileName);
    key_val = GetStringDefault("use_pforest","no");
    USE_P4EST = strcmp(key_val, "yes") == 0 ? TRUE : FALSE;

    if(USE_P4EST){
#ifdef HAVE_P4EST
      /*
       * Initialize sc and p{4,8}est library
       */
      sc_init(amps_CommWorld, 1, 1, NULL, SC_LP_SILENT);
      p4est_init(NULL, SC_LP_PRODUCTION);
#else
      PARFLOW_ERROR("ParFlow compiled without p4est");
#endif
    }

    NewLogging();

    /*-----------------------------------------------------------------------
     * Setup timing table
     *-----------------------------------------------------------------------*/

    NewTiming();

    /*-----------------------------------------------------------------------
     * Solve the problem
     *-----------------------------------------------------------------------*/
    Solve();
    if(!amps_Rank(amps_CommWorld))
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

    if(USE_P4EST){
#ifdef HAVE_P4EST
      sc_finalize();
#endif
    }

    wall_clock_time = amps_Clock() - wall_clock_time;

    IfLogging(0) 
    {
       if(!amps_Rank(amps_CommWorld)) {
 	 log_file = OpenLogFile("ParFlow Total Time");
         fprintf(log_file, "Total Run Time: %f seconds\n",
		      (double)wall_clock_time/(double)AMPS_TICKS_PER_SEC);
	 CloseLogFile(log_file);

       }
    }

    log_file = OpenLogFile("ParFlow Memory stats");
    printMaxMemory(log_file);
    CloseLogFile(log_file);

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
