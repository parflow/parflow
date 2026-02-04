include(ParflowTest)

set(CLM_TESTS)
set(CLM_2D_TESTS)

if (${PARFLOW_HAVE_CLM})
  if (${PARFLOW_HAVE_HYPRE})
    list(APPEND CLM_TESTS
      clm-reuse
      clm_rz_water_stress
      clm_snow_defaults
      clm_snow_partition
      clm_snow_albedo
      clm_snow_dai
      clm_snow_jennings
      clm_snow_sza_damping)

    list(APPEND CLM_2D_TESTS
      clm_slope)
    if (${PARFLOW_HAVE_SILO})
      list(APPEND CLM_2D_TESTS
	clm
	clm_varDZ
	clm.jac
	clm_forc_veg
	clm_4levels)
    endif()
  endif()
endif()


foreach(inputfile ${CLM_TESTS})
  pf_add_py_test(${inputfile})
endforeach()


foreach(inputfile ${CLM_2D_TESTS})
  pf_add_py_parallel_test(${inputfile} 1 1 1)
  pf_add_py_parallel_test(${inputfile} 1 2 1)
  pf_add_py_parallel_test(${inputfile} 2 1 1)
  pf_add_py_parallel_test(${inputfile} 1 4 1)
  pf_add_py_parallel_test(${inputfile} 4 1 1)
  pf_add_py_parallel_test(${inputfile} 2 2 1)
  pf_add_py_parallel_test(${inputfile} 3 3 1)
  pf_add_py_parallel_test(${inputfile} 4 4 1)
endforeach()
