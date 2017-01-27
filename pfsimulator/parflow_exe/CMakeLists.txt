
add_executable(parflow main.c)
target_link_libraries(parflow pfsimulator pfkinsol amps)

if( ${PARFLOW_HAVE_CLM} )
  target_link_libraries(parflow pfclm)
endif( ${PARFLOW_HAVE_CLM} )

if (${PARFLOW_HAVE_HYPRE})
  target_link_libraries (parflow ${HYPRE_LIBRARIES})
endif (${PARFLOW_HAVE_HYPRE})

if (${PARFLOW_HAVE_MPI})
  target_link_libraries (parflow ${MPI_LIBRARIES})
endif (${PARFLOW_HAVE_MPI})


install(TARGETS parflow DESTINATION bin)







