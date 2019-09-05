# Test cases for step 1 in parflow-p4est integration. Please
# check the file step_1_test_{2d,3d}.tcl for further details

add_test(NAME p4est_step1_2d COMMAND tclsh step1_tests_2d.tcl WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME p4est_step1_3d COMMAND tclsh step1_tests_3d.tcl WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
