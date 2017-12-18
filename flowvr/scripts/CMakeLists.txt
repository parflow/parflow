SET(SCRIPT_FILES
  cleanup-flowvr-xml.sh
  CMakeLists.txt
  compare_nc.py
  doMPI.sh
  do.sh
  flowvr_starter.tcl
  logger.py
  parFlowVR_modules.py
  ticker.py
  undist.tcl
  visit_loader.py
  )
SET(PARFLOWVR ${CMAKE_INSTALL_PREFIX}/bin/parflowvr)
INSTALL(FILES ${SCRIPT_FILES}
  DESTINATION ${PARFLOWVR}
  PERMISSIONS WORLD_EXECUTE WORLD_READ WORLD_WRITE OWNER_EXECUTE OWNER_WRITE OWNER_READ)
