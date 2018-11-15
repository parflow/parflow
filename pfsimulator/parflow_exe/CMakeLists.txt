add_executable(parflow main.c)

target_link_libraries(parflow pfsimulator pfkinsol amps)

if( ${PARFLOW_HAVE_CLM} )
  target_link_libraries(parflow pfclm)
endif( ${PARFLOW_HAVE_CLM} )

if (${PARFLOW_HAVE_HYPRE})
  target_link_libraries (parflow ${HYPRE_LIBRARIES})
endif (${PARFLOW_HAVE_HYPRE})

if (${PARFLOW_HAVE_MPI})
  target_link_libraries (parflow ${MPI_LIBRARIES})
endif (${PARFLOW_HAVE_MPI})

if (${PARFLOW_HAVE_SILO})
  target_link_libraries (parflow ${SILO_LIBRARIES})
endif (${PARFLOW_HAVE_SILO})

if (${PARFLOW_HAVE_MELISSA})
  target_link_libraries (parflow ${MELISSA_LIBRARIES})
endif (${PARFLOW_HAVE_MELISSA})

if (${PARFLOW_HAVE_NETCDF})
  target_link_libraries (parflow ${NETCDF_LIBRARIES})
endif (${PARFLOW_HAVE_NETCDF})

if (${PARFLOW_HAVE_HDF5})
  target_link_libraries (parflow ${HDF5_LIBRARIES})
endif (${PARFLOW_HAVE_HDF5})

if (${PARFLOW_HAVE_ZLIB})
  target_link_libraries (parflow ${ZLIB_LIBRARIES})
endif (${PARFLOW_HAVE_ZLIB})

if (${PARFLOW_HAVE_SZLIB})
  target_link_libraries (parflow ${SZLIB_LIBRARIES})
endif (${PARFLOW_HAVE_SZLIB})

if (${PARFLOW_HAVE_SLURM})
  target_link_libraries (parflow ${SLURM_LIBRARIES})
endif (${PARFLOW_HAVE_SLURM})

if( ${PARFLOW_HAVE_FLOWVR} )

  SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}  $ENV{FlowVR_DIR} $ENV{FlowVRVRPN_DIR} $ENV{FlowVRContrib_DIR} ${${CMAKE_PROJECT_NAME}_SOURCE_DIR}/cmake/ ${FlowVR_DIR} ${FlowVRVRPN_DIR} ${FlowVRContrib_DIR})
  INCLUDE(MacroLogFeature)
  FIND_PACKAGE(fca REQUIRED)
  MACRO_LOG_FEATURE(fca_FOUND "FlowVR fca contrib" "fca is needed to run ${CMAKE_PROJECT_NAME} application (compile and install flowvr again making sure to build with the fca lib)" " " TRUE " ")
  target_link_libraries(parflow ${fca_LIBRARY} pfanalyzer)
endif( ${PARFLOW_HAVE_FLOWVR} )

if( ${PARFLOW_ENABLE_PROFILING} )
  set_target_properties(parflow PROPERTIES LINK_FLAGS ${PARFLOW_PROFILE_OPTS})
endif( ${PARFLOW_ENABLE_PROFILING} )

if ( DEFINED PARFLOW_LINKER_FLAGS)
   set_target_properties(parflow PROPERTIES LINK_FLAGS ${PARFLOW_LINKER_FLAGS})
endif ( DEFINED PARFLOW_LINKER_FLAGS)


install(TARGETS parflow DESTINATION bin)
