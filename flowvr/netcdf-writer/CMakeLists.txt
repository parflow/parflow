ADD_EXECUTABLE(netcdf-writer main.c)
target_link_libraries(netcdf-writer pfanalyzer pfsimulator amps)

FIND_PACKAGE (NetCDF REQUIRED)  # we are really dependent from netcdf!
FIND_PACKAGE(fca REQUIRED)

TARGET_LINK_LIBRARIES(netcdf-writer ${fca_LIBRARY})
target_link_libraries (netcdf-writer ${NETCDF_LIBRARIES})

TARGET_INCLUDE_DIRECTORIES(netcdf-writer PUBLIC "${fca_INCLUDE_DIRECTORY} ${NETCDF_INCLUDE_DIRS}")

if (${PARFLOW_HAVE_MPI})
  target_link_libraries (netcdf-writer ${MPI_LIBRARIES})
endif (${PARFLOW_HAVE_MPI})

if (${PARFLOW_HAVE_HDF5})
  target_link_libraries (netcdf-writer ${HDF5_LIBRARIES})
endif (${PARFLOW_HAVE_HDF5})

INSTALL(TARGETS netcdf-writer DESTINATION bin)
