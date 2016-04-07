#include <stdio.h>
#include <string.h>
#include <parflow.h>
#ifdef HAVE_P4EST
#include <parflow_p4est.h>
#include <p4est.h>
#include <p8est.h>
#endif


int
main(int argc, char **argv)
{
    Grid         *grid;
    Vector      *vector;
    VectorUpdateCommHandle  *handle;
    char *key_val;
#ifdef HAVE_P4EST
   const char     *Nkey[3];
   const char     *mkey[3];
   int             P[3],l, N, m, p, t, sum;
#endif

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

    /* Read some necesary input */
    if (!USE_P4EST){
       GlobalsNumProcsY = GetIntDefault("Process.Topology.Q", 1);
       GlobalsNumProcsX = GetIntDefault("Process.Topology.P", 1);
       GlobalsNumProcsZ = GetIntDefault("Process.Topology.R", 1);
    }else{
 #ifdef HAVE_P4EST
       Nkey[0] = "ComputationalGrid.NX";
       Nkey[1] = "ComputationalGrid.NY";
       Nkey[2] = "ComputationalGrid.NZ";

       mkey[0] = "ComputationalSubgrid.MX";
       mkey[1] = "ComputationalSubgrid.MY";
       mkey[2] = "ComputationalSubgrid.MZ";

       for (t=0; t<3; ++t){
           N    = GetIntDefault(Nkey[t], 1);
           m    = GetIntDefault(mkey[t], 1);
           P[t] = N / m;
           l    = N % m;;
           sum  = 0;
           for(p = 0; p < P[t]; ++p){
              sum += ( p < l ) ? m + 1 : m;
           }
           if ( sum != N ){

               InputError("Error: invalid combination of <%s> and <%s>\n",
                          Nkey[t], mkey[t]);
           }
       }
       GlobalsNumProcsX = P[0];
       GlobalsNumProcsY = P[1];
       GlobalsNumProcsZ = P[2];
 #else
       PARFLOW_ERROR("ParFlow compiled without p4est");
 #endif
    }

    /* Create a grid from input data*/
    GlobalsBackground = ReadBackground ();

    GlobalsUserGrid = ReadUserGrid ();

    SetBackgroundBounds (GlobalsBackground, GlobalsUserGrid);

    grid = CreateGrid(GlobalsUserGrid);

    /* Create a vector */
    vector =  NewVectorType(grid, 1, 1, vector_cell_centered );

    /* Put some values on it: including ghost points */
    InitVectorAll(vector, 0);

    /* Change the values in the interior nodes */
    InitVector(vector, 1 << amps_Rank(amps_CommWorld));

    /* Try to do a parallel update*/
    handle = InitVectorUpdate(vector, VectorUpdateAll);

    FinalizeVectorUpdate(handle);

    /* Finalize everything */
    FreeVector(vector);

    FreeGrid (grid);

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
