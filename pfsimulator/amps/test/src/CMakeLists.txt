
include(ParflowTest)

set(TESTS
  test1
  test17
  )

foreach(test ${TESTS})
  add_executable( ${test} ${test}.c)
  target_link_libraries(${test} amps)
endforeach()

foreach(test ${TESTS})
    pf_add_amps_sequential_test(${test} 1)
endforeach()

if ( ${PARFLOW_AMPS_LAYER} IN_LIST PARFLOW_AMPS_LAYER_REQUIRE_MPI )
  foreach(test ${TESTS})
    foreach(rank 1 2 4 8)
      pf_add_amps_parallel_test(${test} ${rank} 1)
    endforeach()
  endforeach()
endif()


