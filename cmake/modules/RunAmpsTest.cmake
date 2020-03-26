#
# Run AMPS test
#
# Use find_package(MPI) in project to set the MPI variables

cmake_minimum_required(VERSION 3.4)

# Execute command with error check
macro(pf_amps_exec_check cmd ranks args)

  set( ENV{PF_TEST} "yes" )

  message("Running : ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS} ${cmd} ${args}")
  if (${ranks} GREATER 0)
    execute_process (COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS} ${cmd} ${args} RESULT_VARIABLE cmdResult OUTPUT_VARIABLE stdout ERROR_VARIABLE stdout)
  else()
    execute_process (COMMAND ./${cmd} ${args} RESULT_VARIABLE cmdResult OUTPUT_VARIABLE stdout ERROR_VARIABLE stdout)
  endif()

  if (cmdResult)
    message (FATAL_ERROR "Error running ${${cmd}} stdout ${stdout}")
  endif()

  message("${stdout}")

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
macro(pf_amps_test_clean)
  file(GLOB FILES *.pfb*)
  if (NOT FILES STREQUAL "")
    file(REMOVE ${FILES})
  endif()

  file(GLOB FILES default_single.out)
  if (NOT FILES STREQUAL "")
    file(REMOVE ${FILES})
  endif()
endmacro()

pf_amps_test_clean ()

list(APPEND CMD ${PARFLOW_TEST})

if (${PARFLOW_HAVE_MEMORYCHECK})
  SET(ENV{PARFLOW_MEMORYCHECK_COMMAND} ${PARFLOW_MEMORYCHECK_COMMAND})
  SET(ENV{PARFLOW_MEMORYCHECK_COMMAND_OPTIONS} ${PARFLOW_MEMORYCHECK_COMMAND_OPTIONS})
endif()

pf_amps_exec_check(${CMD} ${PARFLOW_RANKS} ${PARFLOW_ARGS})

if (${PARFLOW_HAVE_MEMORYCHECK})
  UNSET(ENV{PARFLOW_MEMORYCHECK_COMMAND})
  UNSET(ENV{PARFLOW_MEMORYCHECK_COMMAND_OPTIONS})
endif()
