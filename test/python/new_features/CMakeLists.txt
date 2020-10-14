include(ParflowTest)

set(TESTS
  asc_write
  enum_versioning
  full_clone
  hyphen_test
  image-as-mask
  os_function
  pfb_mask
  pfset_test
  prefix_naming
  serial_runs
  simple-mask
  write_check
  tables_LW
)

foreach(inputfile ${TESTS})
  pf_add_py_test(${inputfile})
endforeach()
