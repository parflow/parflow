include(ParflowTest)

set(TESTS
  key_dynamic_enum_domain
)

foreach(inputfile ${TESTS})
  pf_add_py_test(${inputfile})
endforeach()
