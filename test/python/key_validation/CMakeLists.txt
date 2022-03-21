include(ParflowTest)

set(TESTS
  key_dynamic_enum_domain
  key_initialization_order
)

foreach(inputfile ${TESTS})
  pf_add_py_test(${inputfile})
endforeach()
