cmake_minimum_required (VERSION 3.4)
project (PARFLOW C Fortran)

#-----------------------------------------------------------------------------
# Version number
#-----------------------------------------------------------------------------
set (PARFLOW_VERSION_MAJOR 3)
set (PARFLOW_VERSION_MINOR 0)

# enable testing
enable_testing ()

#-----------------------------------------------------------------------------
# General project wide configuration
#-----------------------------------------------------------------------------
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
 

find_package(MPI)
if (${MPI_C_FOUND})
  set(PARFLOW_HAVE_MPI TRUE)
  include_directories ("${MPI_C_INCLUDE_PATH}")
endif (${MPI_C_FOUND})

find_package(TCL)
if (${TCL_FOUND})
  set(PARFLOW_HAVE_TCL TRUE)
endif (${TCL_FOUND})

# TODO need to set from user input

# Set AMPS communication layer

set(PARFLOW_AMPS_LAYER "mpi1" CACHE STRING "Set the Communications layer to use")
set_property(CACHE PARFLOW_AMPS_LAYER PROPERTY STRINGS seq mpi1 oas3 smpi win32)

include_directories ("${CMAKE_SOURCE_DIR}/pfsimulator/parflow_lib")

include_directories ("${PROJECT_SOURCE_DIR}/pfsimulator/amps/${PARFLOW_AMPS_LAYER}")
include_directories ("${PROJECT_SOURCE_DIR}/pfsimulator/amps/common")

include_directories ("${PROJECT_BINARY_DIR}/include")

#-----------------------------------------------------------------------------
# Setup configure.h file for accessing configure options
#-----------------------------------------------------------------------------
configure_file (
  pfsimulator/parflow_config.h.in
  include/parflow_config.h
  )

# TODO need to setup script logic got correctly generate version file.
configure_file (
  pfsimulator/version.h.in
  include/version.h
  )

#-----------------------------------------------------------------------------
# CMAKE Subdirectories 
#-----------------------------------------------------------------------------

# Need to turn on testing so tests in subdirctories are included in test target.
enable_testing ()

add_subdirectory (pfsimulator)
add_subdirectory (pftools)
add_subdirectory (test)

#-----------------------------------------------------------------------------
# Setup CTEST environment
#-----------------------------------------------------------------------------
include (CTest)


