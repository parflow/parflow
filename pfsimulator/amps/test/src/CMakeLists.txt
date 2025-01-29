
include(ParflowTest)

set(SEQUENTIAL_TESTS
  test1
  test4
  test5
  test7
  test8
  test9
  test10
  test17
  )

set(PARALLEL_TESTS
  test1
  test2
  test3
  test4
  test5
  test6
  test7
  test8
  test9
  test10
  test12
  test13
  test14
  test15
  test16
  test17
  )

# The feature tested by test16 is not supported by the amps 'cuda' layer
if(${PARFLOW_AMPS_LAYER} STREQUAL "cuda")
  list(REMOVE_ITEM PARALLEL_TESTS test16)
endif(${PARFLOW_AMPS_LAYER} STREQUAL "cuda")

# Add GPU test case (same as test15 with Unified Memory)
if(((${PARFLOW_HAVE_CUDA}) OR (${PARFLOW_HAVE_KOKKOS})) AND (${PARFLOW_AMPS_LAYER} STREQUAL "mpi1"))
  list(APPEND PARALLEL_TESTS test18)
  if(NOT (${PARFLOW_HAVE_CUDA}))
    list(REMOVE_ITEM PARALLEL_TESTS test15)  
  endif(NOT (${PARFLOW_HAVE_CUDA}))
endif(((${PARFLOW_HAVE_CUDA}) OR (${PARFLOW_HAVE_KOKKOS})) AND (${PARFLOW_AMPS_LAYER} STREQUAL "mpi1"))

set(ALL_TESTS ${SEQUENTIAL_TESTS} ${PARALLEL_TESTS})
list(REMOVE_DUPLICATES ALL_TESTS)

foreach(test ${ALL_TESTS})

  if( ${PARFLOW_BUILD_WITH_CPP} )
    set_source_files_properties(${test}.c
      PROPERTIES LANGUAGE CXX)
  endif()

  add_executable( ${test} ${test}.c)
  target_link_libraries(${test} amps)
  
  if (${PARFLOW_HAVE_MPI})
    # In CMake 3.13 this could be target_link_options
    target_link_libraries(${test} ${MPI_LINK_FLAGS})
    target_link_libraries (${test} ${MPI_LIBRARIES})
  endif (${PARFLOW_HAVE_MPI})
endforeach()

foreach(test ${SEQUENTIAL_TESTS})
  pf_add_amps_sequential_test(${test} 1)
endforeach()

if ( ${PARFLOW_AMPS_LAYER} IN_LIST PARFLOW_AMPS_LAYER_REQUIRE_MPI )
  foreach(test ${PARALLEL_TESTS})
    foreach(rank 2 4 8)
      pf_add_amps_parallel_test(${test} ${rank} 1)
    endforeach()
  endforeach()
endif()


