include(ParflowTest)


if(PARFLOW_HAVE_NETCDF OR DEFINED ENV{PARFLOW_DIR})
  set(TESTS
    default_richards_with_netcdf
  )
endif()

foreach(inputfile ${TESTS})
  pf_add_py_test(${inputfile})
endforeach()
