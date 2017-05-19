cmake_minimum_required(VERSION 3.4)

# Execute command with error check
macro(pf_exec_check cmd)
  execute_process(COMMAND ${${cmd}} RESULT_VARIABLE cmdResult)
  if(cmdResult)
    message(FATAL_ERROR "Error running ${${cmd}}")
  endif()
endmacro()

# Clean a parflow directory
macro(pf_test_clean)
  file(GLOB FILES *.pfb* *.silo* *.pfsb* *.log .hostfile .amps.* *.out.pftcl *.pfidb *.out.txt default_richards.out *.out.wells indicator_field.out)
  if (NOT FILES STREQUAL "")
    file(REMOVE ${FILES})
  endif()

  file(GLOB FILES default_single.out water_balance.out default_overland.out LW_var_dz_spinup.out test.log.* richards_hydrostatic_equalibrium.out core.* samrai_grid.tmp.tcl samrai_grid2D.tmp.tcl CMakeCache.txt)
  if (NOT FILES STREQUAL "")
    file(REMOVE ${FILES})
  endif()
endmacro()

pf_test_clean ()

set (CMD tclsh ${PARFLOW_TEST})
pf_exec_check(CMD)



