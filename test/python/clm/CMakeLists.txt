include(ParflowTest)

if(${PARFLOW_HAVE_CLM})
  set(CLM_TESTS
    clm
    clm_varDZ
    clm.jac
    clm_forc_veg
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
  foreach(processor_topology "1 2 1" "2 1 1" "1 4 1" "4 1 1")
    continue()
  endforeach()
endforeach()

