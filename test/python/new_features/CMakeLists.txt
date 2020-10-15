include(ParflowTest)

set(TESTS
  asc_write
  default_db
  domain_builder
  enum_versioning
  full_clone
  hyphen_test
  image-as-mask
  LW_table_domain
  os_function
  pfb_mask
  pfidb_pfset
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
