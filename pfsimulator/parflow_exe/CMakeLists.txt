

if( ${PARFLOW_HAVE_CLM} )
  set(EXTRA_LIBRARIES clm)
endif( ${PARFLOW_HAVE_CLM} )

add_executable(parflow main.c)
target_link_libraries(parflow pfsimulator pfkinsol amps ${MPI_C_LIBRARIES})

install(TARGETS parflow DESTINATION bin)



