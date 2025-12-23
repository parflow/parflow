include(ParflowTest)

set(TESTS)

if (${PARFLOW_HAVE_HYPRE})
  list(APPEND TESTS
    washita
    washita_clm_keys)
endif()

foreach(inputfile ${TESTS})
  pf_add_py_test(${inputfile})
endforeach()
