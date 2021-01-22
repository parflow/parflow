include(ParflowTest)

set(TESTS
  asc_write
  clm_file_handling
  clm_build_export
  clm_overwrite
  clm_veg_mapping
  default_db
  domain_builder
  enum_versioning
  from_definition
  full_clone
  hyphen_test
  image-as-mask
  LW_table_domain
  normalize_location
  os_function
  # pfb_mask
  pfidb_pfset
  pfset_test
  prefix_naming
  selection_methods
  serial_runs
  simple-mask
  write_check
  tables_LW
)

foreach(inputfile ${TESTS})
  pf_add_py_test(${inputfile})
endforeach()

add_subdirectory(clm)
