if (${PARFLOW_HAVE_MPI})
  string(APPEND CMAKE_C_FLAGS " ${MPI_COMPILE_FLAGS}" )
  string(APPEND CMAKE_CXX_FLAGS " ${MPI_COMPILE_FLAGS}")
  include_directories(${MPI_INCLUDE_PATH})
endif (${PARFLOW_HAVE_MPI})

string(APPEND CMAKE_C_FLAGS " ${PARFLOW_ETRACE_COMPILE_FLAGS}")
string(APPEND CMAKE_CXX_FLAGS " ${PARFLOW_ETRACE_COMPILE_FLAGS}")

add_subdirectory (amps)

if (NOT PARFLOW_HAVE_SUNDIALS)
  add_subdirectory (kinsol)
endif ()

add_subdirectory (parflow_lib)
add_subdirectory (parflow_exe)
add_subdirectory (third_party)
