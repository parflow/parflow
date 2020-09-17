include(ParflowTest)

set(SILO_TESTS
  indicator_field
)

if(PARFLOW_HAVE_HYPRE OR DEFINED ENV{PARFLOW_DIR})
  list(APPEND SILO_TESTS
    water_balance_y
    water_balance_x
    water_balance_x.hardflow.nojac
    water_balance_x.hardflow.jac
    default_richards_with_silo
  )
endif()

foreach(inputfile ${SILO_TESTS})
  pf_add_py_test(${inputfile})
endforeach()
