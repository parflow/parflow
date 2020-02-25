cmake_minimum_required(VERSION 3.4)

# Execute command with error check
macro(pf_exec_check cmd)

  set( ENV{PF_TEST} "yes" )
  if (${PARFLOW_HAVE_SILO})
    set( ENV{PARFLOW_HAVE_SILO} "yes")
  endif()

  execute_process (COMMAND ${${cmd}} RESULT_VARIABLE cmdResult OUTPUT_VARIABLE stdout ERROR_VARIABLE stdout)
  message(${stdout})
  if (cmdResult)
    message (FATAL_ERROR "Error running ${${cmd}}")
  endif()

  # If FAIL is present test fails
  string(FIND "${stdout}" "FAIL" test)
  if (NOT ${test} EQUAL -1)
    message (FATAL_ERROR "Test Failed: output indicated FAIL")
  endif()

  # Test must say PASSED to pass
  string(FIND "${stdout}" "PASSED" test)
  if (${test} LESS 0)
    message (FATAL_ERROR "Test Failed: output did not indicate PASSED")
  endif()

  string(FIND "${stdout}" "Using Valgrind" test)
  if (NOT ${test} EQUAL -1)
    # Using valgrind
    string(FIND "${stdout}" "ERROR SUMMARY: 0 errors" test)
    if (${test} LESS 0)
      message (FATAL_ERROR "Valgrind Errors Found")
    endif()
  endif()

endmacro()

# Clean a parflow directory
macro(pf_test_clean)
  file(GLOB FILES *.pfb* *.silo* *.pfsb* *.log .hostfile .amps.* *.out.pftcl *.pfidb *.out.txt default_richards.out *.out.wells indicator_field.out)
  if (NOT FILES STREQUAL "")
    file(REMOVE ${FILES})
  endif()

  file(GLOB FILES default_single.out water_balance.out default_overland.out LW_var_dz_spinup.out test.log.* richards_hydrostatic_equalibrium.out core.* samrai_grid.tmp.tcl samrai_grid2D.tmp.tcl CMakeCache.txt)
  if (NOT FILES STREQUAL "")
    file(REMOVE ${FILES})
  endif()
endmacro()

pf_test_clean ()

list(APPEND CMD tclsh)
list(APPEND CMD ${PARFLOW_TEST})

if (${PARFLOW_HAVE_MEMORYCHECK})
  SET(ENV{PARFLOW_MEMORYCHECK_COMMAND} ${PARFLOW_MEMORYCHECK_COMMAND})
  SET(ENV{PARFLOW_MEMORYCHECK_COMMAND_OPTIONS} ${PARFLOW_MEMORYCHECK_COMMAND_OPTIONS})
endif()

pf_exec_check(CMD)

if (${PARFLOW_HAVE_MEMORYCHECK})
  UNSET(ENV{PARFLOW_MEMORYCHECK_COMMAND})
  UNSET(ENV{PARFLOW_MEMORYCHECK_COMMAND_OPTIONS})
endif()


