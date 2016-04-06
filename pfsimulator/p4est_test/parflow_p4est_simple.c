#include <stdio.h>
#include <parflow.h>
#ifdef HAVE_P4EST
#include <p4est.h>
#include <p4est_extended.h>
#endif

#ifdef HAVE_P4EST
typedef struct {
    sc_MPI_Comm     mpicomm;
    int             mpisize;
    int             mpirank;
} mpi_context_t;
#endif

int
main(int argc, char **argv)
{

#ifdef HAVE_P4EST
    int             mpiret;
    mpi_context_t   mpi_context, *mpi = &mpi_context;
    p4est_t        *p4est;
    p4est_connectivity_t *connectivity;

    /*
     * initialize MPI and p4est internals
     */
    mpiret = sc_MPI_Init(&argc, &argv);
    SC_CHECK_MPI(mpiret);
    mpi->mpicomm = sc_MPI_COMM_WORLD;
    mpiret = sc_MPI_Comm_size(mpi->mpicomm, &mpi->mpisize);
    SC_CHECK_MPI(mpiret);
    mpiret = sc_MPI_Comm_rank(mpi->mpicomm, &mpi->mpirank);
    SC_CHECK_MPI(mpiret);

    sc_init(mpi->mpicomm, 1, 1, NULL, SC_LP_DEFAULT);
    p4est_init(NULL, SC_LP_DEFAULT);

    connectivity = p4est_connectivity_new_unitsquare();
    p4est =
        p4est_new_ext(mpi->mpicomm, connectivity, 0, 3, 1, 0, NULL, NULL);

    // finalize the libraries
    p4est_destroy(p4est);
    p4est_connectivity_destroy(connectivity);

    /*
     * clean up and exit
     */
    sc_finalize();
#else
    printf("Parflow was not compiled with p4est\n");
#endif
    return 0;
}
