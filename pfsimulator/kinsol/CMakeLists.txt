
set (SRC_FILES iterativ.c kinsol.c kinspgmr.c llnlmath.c spgmr.c)

add_library(pfkinsol ${SRC_FILES})

if (${PARFLOW_HAVE_MPI})
  target_include_directories (pfkinsol PUBLIC "${MPI_C_INCLUDE_PATH}")
endif (${PARFLOW_HAVE_MPI})

install (TARGETS pfkinsol DESTINATION lib)

