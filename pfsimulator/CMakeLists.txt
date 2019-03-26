

if (${PARFLOW_HAVE_MPI})
  set(CMAKE_C_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
  set(CMAKE_C_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
  set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
  set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
  include_directories(${MPI_INCLUDE_PATH})
endif (${PARFLOW_HAVE_MPI})

add_subdirectory (amps)
add_subdirectory (kinsol)
add_subdirectory (parflow_lib)
add_subdirectory (parflow_exe)

if( ${PARFLOW_HAVE_CLM} )
  add_subdirectory (clm)
endif( ${PARFLOW_HAVE_CLM} )

