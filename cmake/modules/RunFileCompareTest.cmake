cmake_minimum_required(VERSION 3.4)

# Execute command with error check
macro(pf_exec_check cmd files)

  execute_process (COMMAND ${${cmd}} RESULT_VARIABLE cmdResult OUTPUT_VARIABLE stdout ERROR_VARIABLE stdout)
  message(STATUS ${stdout})
  if (cmdResult)
    message (FATAL_ERROR "Error running ${${cmd}}")
  endif()

  # If FAIL is present test fails
  string(FIND "${stdout}" "FAIL" test)
  if (NOT ${test} EQUAL -1)
    message (FATAL_ERROR "Test Failed: output indicated FAIL")
  endif()

  foreach(file ${${files}})
    # cmake -E compare_files --ignore-eol file1 file2
    set(COMPARE_DIR "regression-test-output")
    message(STATUS "Comparing ${file}")
    execute_process (COMMAND ${CMAKE_COMMAND} -E compare_files ${file} "${COMPARE_DIR}/${file}" RESULT_VARIABLE cmdResult OUTPUT_VARIABLE stdout ERROR_VARIABLE stdout)
    if (cmdResult)
      message (FATAL_ERROR "FAIL: file comparison failed ${file}")
    endif()

  endforeach()

endmacro()

pf_exec_check(PFCMD PFFILES PFTEST_ARGS)
