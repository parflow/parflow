add_subdirectory (filters)
add_subdirectory (netcdf-writer)
add_subdirectory (scripts)
add_subdirectory (testcases/steering-C)

cmake_dependent_option (PARFLOW_ENABLE_VISIT_CONNECTOR "Enable building of the Visit Connector (visit libsim is needed!)" OFF "PARFLOW_ENABLE_FLOWVR" OFF)

if (${PARFLOW_ENABLE_VISIT_CONNECTOR})
  add_subdirectory (VisIt-connector)
endif()

