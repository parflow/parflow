set(TESTS "")
if(${PARFLOW_HAVE_CLM})
  if(${PARFLOW_HAVE_HYPRE})
    list(APPEND TESTS
      pfclm_sc.tcl)
  endif()
endif()

set(PARALLEL_TESTS)

if((${PARFLOW_AMPS_LAYER} STREQUAL "mpi1") OR (${PARFLOW_AMPS_LAYER} STREQUAL "cuda"))
  if(${PARFLOW_HAVE_HYPRE})
    list(APPEND PARALLEL_TESTS
      pfclm_sc.tcl)
  endif()
endif()

set(SAMRAI_TESTS)

foreach(inputfile ${TESTS})
  pf_add_sequential_test(${inputfile})
endforeach()

foreach(inputfile ${PARALLEL_TESTS})
  foreach(processor_topology "1 2 1" "2 1 1" "2 2 1" "3 3 1" "1 4 1" "4 1 1")
    pf_add_parallel_test(${inputfile} ${processor_topology})
  endforeach()
endforeach()

foreach(inputfile ${SAMRAI_TESTS})
  foreach(processor_topology "1 1 1 1" "2 1 1 1")
    pf_add_parallel_test(${inputfile} ${processor_topology})
  endforeach()
endforeach()


