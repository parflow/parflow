SET(SCRIPT_FILES
  cleanup-flowvr-xml.sh
  compare_nc.py
  doMPI.sh
  do.sh
  flowvr_starter.tcl
  logger.py
  parFlowVR_modules.py
  ticker.py
  undist.tcl
  visit_loader.py
  abort-on-empty.py
  )

INSTALL(FILES ${SCRIPT_FILES}
  DESTINATION bin/parflowvr
  PERMISSIONS WORLD_EXECUTE WORLD_READ WORLD_WRITE OWNER_EXECUTE OWNER_WRITE OWNER_READ)
