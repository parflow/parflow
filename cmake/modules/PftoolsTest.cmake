# Pftools Test Module
#
# Functions for testing Parflow.
#

#
# Add parflow test to ctest framework.
#
# inputfile is the TCL script that defines the test.
#
function (pf_add_filecompare_test testname cmd comparefile)

  add_test (NAME ${testname} COMMAND ${CMAKE_COMMAND} "-DPFCMD=${cmd}" -P ${CMAKE_SOURCE_DIR}/cmake/modules/RunFileCompareTest.cmake WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

endfunction()


function (pf_add_mask_test testname inputfile)
  
  set(multiValueArgs TEST_ARGS)
  cmake_parse_arguments(opt "" "" "${multiValueArgs}" ${ARGN} )

  string(REGEX REPLACE "\.asc" "\.vtk" vtkname ${inputfile})
  string(REGEX REPLACE "\.asc" "\.pfsol" pfsolname ${inputfile})

  list(APPEND cmd "${CMAKE_BINARY_DIR}/pftools/pfmask-to-pfsol")
  list(APPEND cmd "--mask")
  list(APPEND cmd ${inputfile})
  
  list(APPEND cmd "--vtk")
  list(APPEND cmd ${vtkname})

  list(APPEND cmd "--pfsol")
  list(APPEND cmd ${pfsolname})

  list(APPEND cmd ${opt_TEST_ARGS})
  
  # ./mask-to-pfsol --mask $< --vtk $(patsubst %.pfsol,%.vtk,$@) --pfsol $@ --bottom-patch-label 2 --side-patch-label 3 $(TEST_ARGS)  

  list(APPEND files "${pfsolname}" "${vtkname}")

  add_test (NAME ${testname} COMMAND ${CMAKE_COMMAND} "-DPFCMD=${cmd}" "-DPFFILES=${files}" -P ${CMAKE_SOURCE_DIR}/cmake/modules/RunFileCompareTest.cmake WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

endfunction()

function (pf_add_multi_mask_test testname inputfile)
  
  set(multiValueArgs TEST_ARGS)
  cmake_parse_arguments(opt "" "" "${multiValueArgs}" ${ARGN} )

  string(REGEX REPLACE "\.asc" "\.vtk" vtkname ${inputfile})
  string(REGEX REPLACE "\.asc" "\.pfsol" pfsolname ${inputfile})

  list(APPEND cmd "${CMAKE_BINARY_DIR}/pftools/pfmask-to-pfsol")

  foreach(dir "top" "bottom" "left" "right" "front" "back")
    list(APPEND cmd "--mask-${dir}")
    string(REGEX REPLACE "\.asc" "-${dir}\.asc" dirmask ${inputfile})
    list(APPEND cmd ${dirmask})
  endforeach()
  
  list(APPEND cmd "--vtk")
  list(APPEND cmd ${vtkname})

  list(APPEND cmd "--pfsol")
  list(APPEND cmd ${pfsolname})

  list(APPEND cmd ${opt_TEST_ARGS})
  
  list(APPEND files "${pfsolname}" "${vtkname}")

  add_test (NAME ${testname} COMMAND ${CMAKE_COMMAND} "-DPFCMD=${cmd}" "-DPFFILES=${files}" -P ${CMAKE_SOURCE_DIR}/cmake/modules/RunFileCompareTest.cmake WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

endfunction()

