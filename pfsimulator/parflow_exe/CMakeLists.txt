set(MAIN_SRC main.c)

if( ${PARFLOW_BUILD_WITH_CPP} )
  set_source_files_properties(${MAIN_SRC}
    PROPERTIES LANGUAGE CXX)
endif()

add_executable(parflow ${MAIN_SRC})

target_link_libraries(parflow pfsimulator)

if( ${PARFLOW_HAVE_ETRACE} )
  target_include_directories(parflow PUBLIC "../third_party/etrace")
  target_link_libraries(parflow etrace)
endif( ${PARFLOW_HAVE_ETRACE} )

if( ${PARFLOW_HAVE_RMM} )
  target_link_libraries(parflow rmm::rmm)
endif( ${PARFLOW_HAVE_RMM} )

if( ${PARFLOW_HAVE_UMPIRE} )
  target_link_libraries(parflow umpire)
endif( ${PARFLOW_HAVE_UMPIRE} )

if( ${PARFLOW_HAVE_CLM} )
  target_link_libraries(parflow pfclm)
endif( ${PARFLOW_HAVE_CLM} )

if (${PARFLOW_HAVE_HYPRE})
  target_link_libraries (parflow Hypre::Hypre)
endif (${PARFLOW_HAVE_HYPRE})

if (${PARFLOW_HAVE_SUNDIALS})
  target_link_libraries (parflow ${SUNDIALS_LIBRARIES})
endif (${PARFLOW_HAVE_SUNDIALS})

if (${PARFLOW_HAVE_MPI})
  # In CMake 3.13 this could be target_link_options
  target_link_libraries(parflow ${MPI_LINK_FLAGS})
  target_link_libraries (parflow ${MPI_LIBRARIES})
endif (${PARFLOW_HAVE_MPI})

if (${PARFLOW_HAVE_SILO})
  target_link_libraries (parflow ${SILO_LIBRARIES})
endif (${PARFLOW_HAVE_SILO})

if (${PARFLOW_HAVE_PDI})
  target_link_libraries (parflow ${PDI_LIBRARIES})
  target_link_libraries (parflow ${PARACONF_LIBRARIES})
  target_link_libraries(parflow ${SPDLOG_LIBRARIES})
endif (${PARFLOW_HAVE_PDI})

if (${PARFLOW_HAVE_NETCDF})
  target_link_libraries (parflow NetCDF::NetCDF)
  target_link_libraries (parflow ${CURL_LIBRARIES})
endif (${PARFLOW_HAVE_NETCDF})

if (${PARFLOW_HAVE_HDF5})
  target_link_libraries (parflow ${HDF5_LIBRARIES})

  if (${PARFLOW_HAVE_NETCDF})
    target_link_libraries (parflow ${HDF5_HL_LIBRARIES})

  endif (${PARFLOW_HAVE_NETCDF})
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

if( ${PARFLOW_ENABLE_PROFILING} )
  set_target_properties(parflow PROPERTIES LINK_FLAGS ${PARFLOW_PROFILE_OPTS})
endif( ${PARFLOW_ENABLE_PROFILING} )

if ( DEFINED PARFLOW_LINKER_FLAGS)
   set_target_properties(parflow PROPERTIES LINK_FLAGS ${PARFLOW_LINKER_FLAGS})
endif ( DEFINED PARFLOW_LINKER_FLAGS)

install(TARGETS parflow DESTINATION bin)
