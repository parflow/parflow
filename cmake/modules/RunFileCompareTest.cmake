cmake_minimum_required(VERSION 3.14)

# Execute command with error check
# all parameters passed in as reference
macro(pf_exec_check cmd files)

  # Note: This method of printing the command is only necessary because the
  # 'COMMAND_ECHO' parameter of execute_process is relatively new, introduced
  # around cmake-3.15, and we'd like to be compatible with older cmake versions.
  # See the cmake_minimum_required above.
  list(JOIN ${cmd} " " cmd_str)
  message(STATUS "Executing: ${cmd_str}")
  execute_process (COMMAND ${${cmd}} RESULT_VARIABLE cmd_result OUTPUT_VARIABLE joined_stdout_stderr ERROR_VARIABLE joined_stdout_stderr)
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
    set(compare_cmd ${CMAKE_COMMAND} -E compare_files ${file} "${COMPARE_DIR}/${file}")
    # Note: This method of printing the command is only necessary because the
    # 'COMMAND_ECHO' parameter of execute_process is relatively new, introduced
    # around cmake-3.15, and we'd like to be compatible with older cmake versions.
    # See the cmake_minimum_required above.
    list(JOIN compare_cmd " " cmd_str)
    message(STATUS "Executing: ${cmd_str}")
    execute_process (COMMAND ${compare_cmd} RESULT_VARIABLE cmd_result OUTPUT_VARIABLE joined_stdout_stderr ERROR_VARIABLE joined_stdout_stderr)
    if (cmd_result)
      message (FATAL_ERROR "FAIL: comparison of file ${file} failed (${cmd_result})")
    endif()

  endforeach()

endmacro()

pf_exec_check(PFCMD PFFILES PFTEST_ARGS)
