
set (SRC_FILES
  ${CMAKE_CURRENT_BINARY_DIR}/parflow_config.F90
  precision.F90
  infnan.F90
  clm_varpar.F90
  clmtype.F90
  drv_module.F90
  clm_varcon.F90
  drv_tilemodule.F90
  drv_gridmodule.F90
  clm_typini.F90
  pf_readout.F90
  close_files.F90
  open_files.F90
  pf_couple.F90
  clm_balchk.F90
  clm_lake.F90
  clm_subdiv.F90
  drv_2dout.F90
  drv_readvegpf.F90
  clm_combin.F90
  clm_leaftem.F90
  clm_surfalb.F90
  drv_almaout.F90
  drv_readvegtf.F90
  clm_combo.F90
  clm_main.F90
  clm_surfrad.F90
  drv_astp.F90
  drv_restart.F90
  clm_compact.F90
  clm_meltfreeze.F90
  clm_thermal.F90
  drv_clm2g.F90
  drv_t2g.F90
  clm_condch.F90
  clm_obuini.F90
  clm_thermalk.F90
  drv_clmini.F90
  drv_tick.F90
  clm_condcq.F90
  clm_obult.F90
  drv_g2clm.F90
  clm_coszen.F90
  clm_psi.F90
  clm_tridia.F90
  drv_getforce.F90
  endrun.F90
  clm_dynvegpar.F90
  clm_qsadv.F90
  clm_twostream.F90
  drv_gridave.F90
  clm_hydro_canopy.F90
  clm_snowage.F90
  clm_hydro_irrig.F90
  clm_snowalb.F90
  clm_hydro_snow.F90
  clm_soilalb.F90
  clm_hydro_soil.F90
  drv_pout.F90
  clm_hydro_wetice.F90
  clm_stomata.F90
  drv_1dout.F90
  drv_readclmin.F90
  clm.F90
  )

#set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -M${CMAKE_CURRENT_BINARY_DIR}")

add_library(pfclm ${SRC_FILES})

install (TARGETS pfclm DESTINATION lib)
