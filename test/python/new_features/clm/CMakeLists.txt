include(ParflowTest)

set(TESTS
  washita
)

foreach(inputfile ${TESTS})
  pf_add_py_test(${inputfile})
endforeach()
