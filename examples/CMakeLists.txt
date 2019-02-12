
include(ParflowTest)

set(TESTS 
bc.00.tcl cond.tcl default_richards.tcl default_single.tcl forsyth2.tcl forsyth5.tcl impes.internalbc.tcl impes.plinear.tcl impes.simple.tcl llnl.tcl richards.plinear.tcl
)

foreach(inputfile ${TESTS})
  pf_add_sequential_test(${inputfile})
endforeach()


