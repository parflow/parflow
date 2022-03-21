
include(ParflowTest)

set(TESTS bc.00.tcl cond.tcl default_richards.tcl default_single.tcl forsyth2.tcl forsyth5.tcl impes.internalbc.tcl impes.simple.tcl llnl.tcl llnl-perm.tcl test_X3Y2PlusSinXYPlus1.tcl test_XPlusYPlusZ.tcl test_X.tcl test_XYZTPlus1.tcl well_00.tcl well_01.tcl)

if((${PARFLOW_HAVE_HYPRE}) AND (${PARFLOW_HAVE_SILO}))
  list(APPEND TESTS impes.plinear.tcl)
endif()

#This test is suspected to be too precision sensitive
if((${PARFLOW_HAVE_CUDA}) OR (${PARFLOW_HAVE_KOKKOS}) OR (${PARFLOW_HAVE_OMP}))
  list(REMOVE_ITEM TESTS test_XPlusYPlusZ.tcl)
endif()

if((${PARFLOW_AMPS_LAYER} STREQUAL "mpi1") OR (${PARFLOW_AMPS_LAYER} STREQUAL "cuda"))
  set(PARALLEL_TESTS richards.plinear.tcl)
endif()

foreach(inputfile ${TESTS})
  pf_add_sequential_test(${inputfile})
endforeach()

foreach(inputfile ${PARALLEL_TESTS})
  pf_add_parallel_test(${inputfile} "2 1 1")
endforeach()

install(FILES default_single.tcl pftest.tcl DESTINATION examples)
install(FILES
  correct_output/default_single.out.concen.0.00.00000.pfsb
  correct_output/default_single.out.concen.0.00.00001.pfsb
  correct_output/default_single.out.concen.0.00.00002.pfsb
  correct_output/default_single.out.concen.0.00.00003.pfsb
  correct_output/default_single.out.concen.0.00.00004.pfsb
  correct_output/default_single.out.concen.0.00.00005.pfsb
  correct_output/default_single.out.perm_x.pfb
  correct_output/default_single.out.perm_y.pfb
  correct_output/default_single.out.perm_z.pfb
  correct_output/default_single.out.press.00000.pfb
  DESTINATION examples/correct_output)



