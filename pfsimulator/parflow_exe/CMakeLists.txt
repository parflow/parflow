

add_executable(parflow main.c)
target_link_libraries(parflow pfsimulator pfkinsol amps ${MPI_C_LIBRARIES})

install(TARGETS parflow DESTINATION bin)



