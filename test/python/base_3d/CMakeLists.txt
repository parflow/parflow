include(ParflowTest)

set(3D_TESTS
  default_single
)

if(${PARFLOW_HAVE_HYPRE})
  list(APPEND 3D_TESTS
    default_richards
  )
endif()

foreach(inputfile ${3D_TESTS})
  pf_add_py_test(${inputfile})
  foreach(processor_topology "1 1 2" "1 2 1" "2 1 1" "2 2 2" "3 3 3" "1 1 4" "1 4 1" "4 1 1")
    if(((${PARFLOW_HAVE_CUDA}) OR (${PARFLOW_HAVE_OMP})) AND (${processor_topology} STREQUAL "3 3 3"))
      #These tests should pass with CUDA or OMP, but they take a very long time
      continue()
    endif(((${PARFLOW_HAVE_CUDA}) OR (${PARFLOW_HAVE_OMP})) AND (${processor_topology} STREQUAL "3 3 3"))
  endforeach()
endforeach()