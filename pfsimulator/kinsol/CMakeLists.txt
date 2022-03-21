
set (SRC_FILES iterativ.c kinsol.c kinspgmr.c llnlmath.c spgmr.c)

add_library(pfkinsol ${SRC_FILES})

if(${PARFLOW_HAVE_CUDA})
  target_include_directories(pfkinsol PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif(${PARFLOW_HAVE_CUDA})

if( ${PARFLOW_HAVE_RMM} )
  target_link_libraries(pfkinsol rmm)
  target_include_directories(pfkinsol PRIVATE ${RMM_INCLUDE})
  target_compile_definitions(pfkinsol PRIVATE PARFLOW_HAVE_RMM)
endif( ${PARFLOW_HAVE_RMM} )

if (${PARFLOW_HAVE_MPI})
  target_include_directories (pfkinsol PUBLIC "${MPI_C_INCLUDE_PATH}")
endif (${PARFLOW_HAVE_MPI})

install (TARGETS pfkinsol DESTINATION lib)
