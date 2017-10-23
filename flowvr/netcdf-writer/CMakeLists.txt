ADD_EXECUTABLE(netcdf-writer main.c)
target_link_libraries(netcdf-writer pfsimulator amps)

FIND_PACKAGE (netCDF REQUIRED)  # we are really dependent from netcdf!
FIND_LIBRARY(FCA_LIBRARY fca)

TARGET_LINK_LIBRARIES(netcdf-writer ${FCA_LIBRARY})
target_link_libraries (netcdf-writer ${netCDF_LIBRARIES})

if (${PARFLOW_HAVE_MPI})
  target_link_libraries (netcdf-writer ${MPI_LIBRARIES})
endif (${PARFLOW_HAVE_MPI})

if (${PARFLOW_HAVE_HDF5})
  target_link_libraries (netcdf-writer ${HDF5_LIBRARIES})
endif (${PARFLOW_HAVE_HDF5})

INSTALL(TARGETS netcdf-writer DESTINATION bin)
