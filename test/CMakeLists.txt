
add_test (NAME default_single COMMAND tclsh default_single.tcl 1 1 1 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if (${PARFLOW_HAVE_CLM})
  add_subdirectory (clm)
endif(${PARFLOW_HAVE_CLM})
