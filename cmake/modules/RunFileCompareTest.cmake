cmake_minimum_required(VERSION 3.4)

# Execute command with error check
# all parameters passed in as reference
macro(pf_exec_check cmd files)
  execute_process (COMMAND ${${cmd}} RESULT_VARIABLE cmd_result OUTPUT_VARIABLE joined_stdout_stderr ERROR_VARIABLE joined_stdout_stderr COMMAND_ECHO STDOUT)
  message(STATUS "Output:\n${joined_stdout_stderr}")
  if (cmd_result)
    message (FATAL_ERROR "Error (${cmd_result}) while running.")
  endif()

  # If FAIL is present test fails
  string(FIND "${joined_stdout_stderr}" "FAIL" test)
  if (NOT ${test} EQUAL -1)
    message (FATAL_ERROR "Test Failed: output indicated FAIL")
  endif()

  foreach(file ${${files}})
    # cmake -E compare_files --ignore-eol file1 file2
    set(COMPARE_DIR "regression-test-output")
    message(STATUS "Comparing ${file}")
    execute_process (COMMAND ${CMAKE_COMMAND} -E compare_files ${file} "${COMPARE_DIR}/${file}" RESULT_VARIABLE cmd_result OUTPUT_VARIABLE joined_stdout_stderr ERROR_VARIABLE joined_stdout_stderr COMMAND_ECHO STDOUT)
    if (cmd_result)
      message (FATAL_ERROR "FAIL: comparison of file ${file} failed (${cmd_result})")
    endif()

  endforeach()

endmacro()

pf_exec_check(PFCMD PFFILES PFTEST_ARGS)
