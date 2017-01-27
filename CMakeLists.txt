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

# TODO should get rid of non-prefix versions of flags; preprocessor flags should be in PARFLOW namespace

find_package(MPI)
if (${MPI_C_FOUND})
  set(PARFLOW_HAVE_MPI TRUE)
  set(HAVE_MPI TRUE)
endif (${MPI_C_FOUND})

find_package(TCL)
if (${TCL_FOUND})
  set(PARFLOW_HAVE_TCL TRUE)
endif (${TCL_FOUND})

find_package(Hypre)
if (${HYPRE_FOUND})
  set(PARFLOW_HAVE_HYPRE TRUE)
  set(HAVE_HYPRE TRUE)

endif (${HYPRE_FOUND})

#-----------------------------------------------------------------------------
# Fortran checks
#-----------------------------------------------------------------------------
INCLUDE (CheckFortranSourceCompiles)

# Check if simple fortran 77 compile works
CHECK_Fortran_SOURCE_COMPILES("      program main
      implicit none
      write ( *, '(a)' ) '  Hello, world!'
      stop
      end" FORTRAN_77_WORKS)

# Check if Fortran 90 compile works with free format
set(CMAKE_REQUIRED_FLAGS "${CMAKE_Fortran_FORMAT_FREE_FLAG}")
CHECK_Fortran_SOURCE_COMPILES("program main
  implicit none

  write ( *, '(a)' ) '  Hello, world!'

  stop
end" FORTRAN_F90_WORKS)

# Check whether the Fortran compiler supports the access="stream" open syntax
CHECK_Fortran_SOURCE_COMPILES("program freeform
   open(10, file='test.bin', access='stream', form='unformatted', status='replace')
   write(10) \"first\"
   write(10) \"second\"
   close(UNIT=10)
 end program freeform" HAVE_FC_ACCESS_STREAM)

# Check whether the Fortran compiler supports the access="sequential" open syntax
CHECK_Fortran_SOURCE_COMPILES("program freeform
  open(10, file='test.bin', access='sequential', form='binary', status='replace')
  write(10) \"first\"
  write(10) \"second\"
  close(UNIT=10)
end program freeform" HAVE_FC_ACCESS_SEQUENTIAL)

INCLUDE (CheckCSourceCompiles) 
CHECK_C_SOURCE_COMPILES("int main(int argc, char **argv) {return 0;}"
  C_WORKS)

if (${HAVE_FC_ACCESS_STREAM})
  set (PARFLOW_FC_ACCESS "stream")
  set (PARFLOW_FC_FORM "unformatted")
elseif (${HAVE_FC_ACCESS_SEQUENTIAL})
  set (PARFLOW_FC_ACCESS, "sequential")
  set (PARFLOW_FC_FORM "binary")
else (${HAVE_FC_ACCESS_STREAM})
  message( FATAL_ERROR "Unable to determine syntax to use for Fortran binary files")
endif (${HAVE_FC_ACCESS_STREAM})

# TODO replace CASC macro names with PARFLOW when completed.

# Set AMPS communication layer
set(PARFLOW_AMPS_LAYER "mpi1" CACHE STRING "Set the Communications layer to use")
set_property(CACHE PARFLOW_AMPS_LAYER PROPERTY STRINGS mpi1 smpi seq oas3 win32)

include(CheckSymbolExists)
check_symbol_exists(gettimeofday sys/time.h CASC_HAVE_GETTIMEOFDAY)

option(PARFLOW_HAVE_CLM "Compile with CLM" "OFF")

if ( ${PARFLOW_HAVE_CLM} )
  set(HAVE_CLM ${PARFLOW_HAVE_CLM})
endif ( ${PARFLOW_HAVE_CLM} )

include_directories ("${CMAKE_SOURCE_DIR}/pfsimulator/parflow_lib")

include_directories ("${PROJECT_SOURCE_DIR}/pfsimulator/amps/${PARFLOW_AMPS_LAYER}")
include_directories ("${PROJECT_SOURCE_DIR}/pfsimulator/amps/common")

include_directories ("${PROJECT_BINARY_DIR}/include")

#-----------------------------------------------------------------------------
# Setup configure.h file for accessing configure options
#-----------------------------------------------------------------------------
configure_file (pfsimulator/parflow_config.h.in include/parflow_config.h)

# TODO need to setup script logic got correctly generate version file.
configure_file (pfsimulator/version.h.in include/version.h)
configure_file (pfsimulator/config/Makefile.config.in config/Makefile.config)
install(FILES ${PROJECT_BINARY_DIR}/config/Makefile.config DESTINATION config)

# TODO zz.cmake notes

if ( ${PARFLOW_HAVE_CLM} )
  configure_file (pfsimulator/clm/parflow_config.F90.in ${PROJECT_BINARY_DIR}/pfsimulator/clm/parflow_config.F90)
endif ( ${PARFLOW_HAVE_CLM} )
  

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


