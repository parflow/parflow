#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <parflow.h>
#ifdef HAVE_P4EST
#include "parflow_p4est.h"
#include <p4est.h>
#include <p8est.h>
#endif
#include <assert.h>

int
main(int argc, char **argv)
{
    size_t fret;
    int  len, file_len, num_elem, k;
    char filename[MAXPATHLEN];
    char *file_data, *data_ptr;
    char *saveptr;
    FILE *file = NULL;
    FILE *log_file = NULL;
    int mpiret, rank;
    char *key_len, *key;
    char *value_len, *value;
    IDB *db;
    IDB_Entry *entry;
    amps_Clock_t wall_clock_time;

    if (amps_Init(&argc, &argv)) {
        amps_Printf("Error: amps_Init initalization failed\n");
        exit(1);
    }

    wall_clock_time = amps_Clock();
    rank = amps_Rank(amps_CommWorld);

    sc_init(amps_CommWorld, 1, 1, NULL, SC_LP_SILENT);

    if(argc != 2) {
       amps_Printf("Error: invalid number of arguments.\n");
       amps_Printf("       Invoke as parflow <pfidb file>.\n");
       exit(1);
    }

    /*-----------------------------------------------------------------------
     * Set up globals structure
     *-----------------------------------------------------------------------*/

    NewGlobals(argv[1]);

    /* Initalize the db structure */
    db = (IDB *)HBT_new(IDB_Compare,
                        IDB_Free,
                        IDB_Print,
                        NULL,
                        0);

    /* Rank 0 opens and reads lenght of the file */
    if (!rank){
      if ( ( file = fopen(GlobalsInFileName, "r") )== NULL){
          PARFLOW_ERROR("Failed to open db file");
      }else{
          if ( fseek (file, 0, SEEK_END) )
            PARFLOW_ERROR("fseek failed");

          file_len = (int) ftell (file);

          if (file_len < 0)
            PARFLOW_ERROR("ftell failed");

          if ( fseek (file, 0, SEEK_SET) )
            PARFLOW_ERROR("fseek failed");
      }
    }

    /*Send lenght of file to all cpus and allocate space for its data */
    mpiret = sc_MPI_Bcast(&file_len, 1, sc_MPI_INT, 0 , amps_CommWorld);
    SC_CHECK_MPI (mpiret);

    file_data = talloc(char, file_len);

    /*Rank 0 copies content of db to string */
    if(!rank){
      fret = fread(file_data, sizeof(char), (size_t) file_len, file);
      if ( fret < (size_t)file_len )
        PARFLOW_ERROR("fread failed");
    }

    if(file)
      fclose(file);

    /*Make information on file available to all cpus*/
    mpiret = sc_MPI_Bcast(file_data, file_len, sc_MPI_CHAR, 0 , amps_CommWorld);
    SC_CHECK_MPI (mpiret);

    if ( (data_ptr = strtok_r(file_data, "\n", &saveptr)) ){
        /*Get number of elements*/
        if ( sscanf(data_ptr, "%i", &num_elem ) != 1 ){
            PARFLOW_ERROR("scanf failed");
        }
    }else{
        PARFLOW_ERROR("strtok_r failed");
    }

   for (k=0; k<num_elem; ++k){

        key_len = strtok_r(NULL, "\n", &saveptr);
        if ( sscanf(key_len, "%i", &len ) == 1 && key_len ){
          if ( len ){
              key =  strtok_r(NULL, "\n", &saveptr);
              assert( len == (int) strlen(key) );
          }else{
              key = " ";
          }
        }else{
            PARFLOW_ERROR("Failed reading key");
        }

        value_len = strtok_r(NULL, "\n", &saveptr);
        if ( sscanf(value_len, "%i", &len ) == 1 && value_len ){
          if ( len ){
              value =  strtok_r(NULL, "\n", &saveptr);
              assert( len == (int) strlen(value) );
          }else{
             value = " ";
          }
        }else{
            PARFLOW_ERROR("Failed reading value");
        }

        /* Create an new entry */
        entry = IDB_NewEntry(key, value);

        /* Insert into the height balanced tree */
        HBT_insert(db, entry, 0);
    }

    if (file_data){
       tfree(file_data);
    }

    amps_ThreadLocal(input_database) = db;

    key = GetStringDefault("use_pforest","no");
    USE_P4EST = strcmp(key, "yes") == 0 ? TRUE : FALSE;

    if(USE_P4EST){
#ifdef HAVE_P4EST
      /*
       * Initialize sc and p{4,8}est library
       */
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

    sc_finalize();

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
