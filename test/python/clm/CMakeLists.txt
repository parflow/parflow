include(ParflowTest)

if(${PARFLOW_HAVE_CLM})
  set(CLM_TESTS
    clm-reuse
  )
  set(CLM_2D_TESTS
    clm
    clm_varDZ
    clm.jac
    clm_forc_veg
  )
endif()


foreach(inputfile ${CLM_TESTS})
  pf_add_py_test(${inputfile})
endforeach()


foreach(inputfile ${CLM_2D_TESTS})
  pf_add_py_parallel_test(${inputfile} 1 1 1)
  pf_add_py_parallel_test(${inputfile} 1 2 1)
  pf_add_py_parallel_test(${inputfile} 2 1 1)
  pf_add_py_parallel_test(${inputfile} 1 4 1)
  pf_add_py_parallel_test(${inputfile} 4 1 1)
endforeach()

