include(ParflowTest)
include(ProcessorCount)

ProcessorCount(N)

if(PARFLOW_HAVE_CLM OR DEFINED ENV{PARFLOW_DIR})
  set(CLM_TESTS
    clm-reuse
  )
  set(CLM_2D_TESTS
    clm
    clm_varDZ
    clm.jac
    clm_forc_veg
    clm_4levels
    clm_slope
  )
endif()


foreach(inputfile ${CLM_TESTS})
  pf_add_py_test(${inputfile})
endforeach()


foreach(inputfile ${CLM_2D_TESTS})
  pf_add_py_parallel_test(${inputfile} 1 1 1)

  if(N GREATER_EQUAL 2)
    pf_add_py_parallel_test(${inputfile} 1 2 1)
    pf_add_py_parallel_test(${inputfile} 2 1 1)
  endif()

  if(N GREATER_EQUAL 4)
    pf_add_py_parallel_test(${inputfile} 1 4 1)
    pf_add_py_parallel_test(${inputfile} 4 1 1)
    pf_add_py_parallel_test(${inputfile} 2 2 1)
  endif()

  if(N GREATER_EQUAL 9)
    pf_add_py_parallel_test(${inputfile} 3 3 1)
  endif()

  if(N GREATER_EQUAL 16)
    pf_add_py_parallel_test(${inputfile} 4 4 1)
  endif()
endforeach()
