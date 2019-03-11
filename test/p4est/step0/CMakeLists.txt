# Test cases for step 0 in parflow-p4est integration. Please
# check the files step_0_test_{2d,3d}.tcl for further details

add_test(NAME p4est_step0_2d COMMAND tclsh step0_tests_2d.tcl WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(NAME p4est_step0_3d COMMAND tclsh step0_tests_3d.tcl WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
