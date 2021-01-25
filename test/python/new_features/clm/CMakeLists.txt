include(ParflowTest)

set(TESTS
  washita
  washita_clm_keys
)

foreach(inputfile ${TESTS})
  pf_add_py_test(${inputfile})
endforeach()
