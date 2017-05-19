# Parflow Test Module
#
# Functions for testing Parflow.
#

#
# Add parflow test to ctest framework.
#
# Parflow tests must accept a processor topology.  Tests may ignore
# toplogy for sequential tests.
#
# inputfile is the TCL script that defines the test.
# topology is the processor topology, number of processor along each axis NX NY NZ
#
# For sequential tests set topology to 1 1 1
#
function (pf_add_parallel_test inputfile topology)
  string(REGEX REPLACE "/\.tcl" "" testname ${inputfile})
  string(REGEX REPLACE " " "_" postfix ${topology})

  list(APPEND args ${inputfile})
  separate_arguments(targs UNIX_COMMAND ${topology})
  list(APPEND args ${targs})

  add_test (NAME ${testname}_${postfix} COMMAND ${CMAKE_COMMAND} -DPARFLOW_TEST=${args} -P ${CMAKE_SOURCE_DIR}/cmake/modules/RunParallelTest.cmake WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

#
# Add parflow test to ctest framework.
#
# inputfile is the TCL script that defines the test.
#
function (pf_add_sequential_test inputfile)
  pf_add_parallel_test $inputfile "1 1 1")
endfunction()

