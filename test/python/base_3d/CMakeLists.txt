include(ParflowTest)

set(3D_TESTS
  default_single
)

if(PARFLOW_HAVE_HYPRE OR DEFINED ENV{PARFLOW_DIR})
  list(APPEND 3D_TESTS
    default_richards
  )
endif()

foreach(inputfile ${3D_TESTS})
  pf_add_py_parallel_test(${inputfile} 1 1 1)
  pf_add_py_parallel_test(${inputfile} 1 1 2)
  pf_add_py_parallel_test(${inputfile} 1 2 1)
  pf_add_py_parallel_test(${inputfile} 2 1 1)
  pf_add_py_parallel_test(${inputfile} 2 2 2)
  pf_add_py_parallel_test(${inputfile} 3 3 3)
  pf_add_py_parallel_test(${inputfile} 1 1 4)
  pf_add_py_parallel_test(${inputfile} 1 4 1)
  pf_add_py_parallel_test(${inputfile} 4 1 1)
endforeach()
