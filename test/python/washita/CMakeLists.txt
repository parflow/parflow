include(ParflowTest)

if(PARFLOW_HAVE_HYPRE OR DEFINED ENV{PARFLOW_DIR})
  if(PARFLOW_HAVE_CLM OR DEFINED ENV{PARFLOW_DIR})
    pf_add_py_test(LW_Test)
  endif()
endif()
