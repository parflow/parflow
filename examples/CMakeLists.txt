
include(ParflowTest)

set(TESTS bc.00.tcl cond.tcl default_richards.tcl default_single.tcl forsyth2.tcl forsyth5.tcl impes.internalbc.tcl impes.simple.tcl llnl.tcl llnl-perm.tcl test_X3Y2PlusSinXYPlus1.tcl test_XPlusYPlusZ.tcl test_X.tcl test_XYZTPlus1.tcl well_00.tcl well_01.tcl)

if(${PARFLOW_HAVE_HYPRE})
  list(APPEND TESTS impes.plinear.tcl)
endif()

if(${PARFLOW_AMPS_LAYER} STREQUAL "mpi1")
  set(PARALLEL_TESTS richards.plinear.tcl)
endif()

foreach(inputfile ${TESTS})
  pf_add_sequential_test(${inputfile})
endforeach()

foreach(inputfile ${PARALLEL_TESTS})
  pf_add_parallel_test(${inputfile} "2 1 1")
endforeach()



