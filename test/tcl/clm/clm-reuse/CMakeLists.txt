
set(TESTS "")
if(${PARFLOW_HAVE_CLM})
  if(${PARFLOW_HAVE_HYPRE})
    list(APPEND TESTS
      clm-reuse.tcl)
  endif()
endif()

foreach(inputfile ${TESTS})
  pf_add_sequential_test(${inputfile})
endforeach()



