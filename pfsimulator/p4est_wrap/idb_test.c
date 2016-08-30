#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <parflow.h>
#ifdef HAVE_P4EST
#include "parflow_p4est.h"
#include <p4est.h>
#include <p8est.h>
#endif

int
main(int argc, char **argv)
{
    size_t fret;
    int  file_len;
    char *file_data, *data_ptr;
    char *saveptr, *saveptr1 ,*ptr;
    FILE *file = NULL;
    int mpiret, rank, count;
    char key[IDB_MAX_KEY_LEN];
    char value[IDB_MAX_VALUE_LEN];

    int key_len;
    int value_len;

    if (amps_Init(&argc, &argv)) {
        amps_Printf("Error: amps_Init initalization failed\n");
        exit(1);
    }

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

    /* Rank 0 opens and reads lenght of the file */
    if (!rank){
      if ( ( file = fopen(argv[1], "r") )== NULL){
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


    ptr = data_ptr = strtok_r(file_data, "\n", &saveptr);
    while (data_ptr){
        saveptr1 = saveptr;
        data_ptr = strtok_r(NULL, "\n", &saveptr);
        printf("%s %p\n", data_ptr, saveptr);
    }

#if 0
    /*Get number of elements*/
    if ( sscanf(file_data, "%i%n", &num_elem, &offset ) != 1 ){
        PARFLOW_ERROR("scanf failed");
    }

    /*loop over the file_data to build IDB*/
    data_ptr = file_data + offset;
    count = 0;


    while ( sscanf(data_ptr, "%i%s%i%s%n", &key_len, key, &value_len, value, &offset ) == 4 ){
        printf("%i %s %i %s\n", key_len, key, value_len, value);
        data_ptr += offset;
        count ++;
    }
#endif

#if 0
    while ( sscanf(data_ptr, "%i%s%n", &key_len, key, &offset ) == 2 ){
        data_ptr += offset;
        if (sscanf(data_ptr, "%i%n", &value_len, &offset ) != 1);

        data_ptr += offset;
        if ( !value_len ){
            value[0] = '\0';
            offset += 1;
        }else{
            sscanf(data_ptr, "%s%n", value, &offset );
        }

        printf("%i %s %i %s\n", key_len, key, value_len, value);
        data_ptr += offset;
        count ++;
    }
#endif

    if (file_data){
        tfree(file_data);
    }


    FreeGlobals();
    sc_finalize();
    amps_Finalize();

    return 0;
}
