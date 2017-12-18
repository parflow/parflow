add_subdirectory (filters)
add_subdirectory (netcdf-writer)
add_subdirectory (analyzer)
add_subdirectory (scripts)

set (PARFLOW_ENABLE_VISIT_CONNECTOR False CACHE BOOL "Enable building of the Visit Connector (visit libsim is needed!)")

if (${PARFLOW_ENABLE_VISIT_CONNECTOR})
  add_subdirectory (VisIt-connector)
endif()
