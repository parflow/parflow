include(ParflowTest)

if(PARFLOW_HAVE_HYPRE AND PARFLOW_HAVE_CLM)
  pf_add_py_test(LW_Test)
endif()
