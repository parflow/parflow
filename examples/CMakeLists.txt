
include(ParflowTest)

set(TESTS 
  bc.00.tcl
  default_single.tcl
  cond.tcl
  impes.simple.tcl
  impes.internalbc.tcl
  impes.plinear.tcl
  richards.plinear.tcl
)

foreach(inputfile ${TESTS})
  pf_add_sequential_test(${inputfile})
endforeach()


