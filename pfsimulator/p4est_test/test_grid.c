#include <stdio.h>
#include <parflow.h>
#ifdef HAVE_P4EST
#include <parflow_p4est.h>
#include <p4est.h>
#include <p8est.h>
#endif

int
main(int argc, char **argv)
{
    char *key_val;

    if (amps_Init(&argc, &argv)) {
        amps_Printf("Error: amps_Init initalization failed\n");
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
      sc_init(amps_CommWorld, 1, 1, NULL, SC_LP_DEFAULT);
      p4est_init(NULL, SC_LP_DEFAULT);
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
#else
      PARFLOW_ERROR("ParFlow compiled without p4est");
#endif
    }

    IDB_FreeDB(amps_ThreadLocal(input_database));
    FreeGlobals();
    amps_Finalize();

    return 0;
}
