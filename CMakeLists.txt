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
set(SAVE_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${CMAKE_Fortran_FORMAT_FREE_FLAG}")
CHECK_Fortran_SOURCE_COMPILES("program main
  implicit none

  write ( *, '(a)' ) '  Hello, world!'

  stop
end" FORTRAN_F90_WORKS)

#
# Determine syntax for writing binary file under Fortran
#

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

#
# Set implicit none flag on Fortran compiles
#
include(CheckFortranCompilerFlag)
check_fortran_compiler_flag("-g" SGS_DEBUGGER)

set(none_test 0)
foreach(flag "-implicitnone" "-fimplicit-none" "-u" "-Wimplicit none")
  message(STATUS "Checking Fortran implicit none flag : ${flag}")
  check_fortran_compiler_flag("${flag}" PARFLOW_FORTRAN_IMPLICIT_NONE_${none_test})
  if(${PARFLOW_FORTRAN_IMPLICIT_NONE_${none_test}})
    set(PARFLOW_FORTRAN_IMPLICIT_NONE TRUE)
    set(PARFLOW_FORTRAN_IMPLICIT_NONE_FLAG "${flag}")
    break()
  endif(${PARFLOW_FORTRAN_IMPLICIT_NONE_${none_test}})
  math(EXPR none_test "${none_test} + 1")
endforeach(flag)

if(${PARFLOW_FORTRAN_IMPLICIT_NONE})
  message("SGS ${CMAKE_Fortran_FLAGS} ${PARFLOW_FORTRAN_IMPLICIT_NONE_FLAG}")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${PARFLOW_FORTRAN_IMPLICIT_NONE_FLAG}")
endif(${PARFLOW_FORTRAN_IMPLICIT_NONE})

set(CMAKE_REQUIRED_FLAGS ${SAVE_CMAKE_REQUIRED_FLAGS})

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

#
# Check for platform specific features
#

# Check for gettimeofday
include(CheckSymbolExists)
check_symbol_exists(gettimeofday sys/time.h PARFLOW_HAVE_GETTIMEOFDAY)
if ( ${PARFLOW_HAVE_GETTIMEOFDAY} )
  set(CASC_HAVE_GETTIMEOFDAY ${PARFLOW_HAVE_GETTIMEOFDAY})
endif ( ${PARFLOW_HAVE_GETTIMEOFDAY} )


option(PARFLOW_HAVE_CLM "Compile with CLM" "OFF")

if ( ${PARFLOW_HAVE_CLM} )
  set(HAVE_CLM ${PARFLOW_HAVE_CLM})
endif ( ${PARFLOW_HAVE_CLM} )


#
# Parflow specific configuration options
#

# Control timing of Parflow functions.
set (PARFLOW_ENABLE_TIMING False CACHE BOOL "Enable timing of key Parflow functions; may slow down performance")
if( ${PARFLOW_ENABLE_TIMING} )
  set (PF_TIMING ${PARFLOW_ENABLE_TIMING})
endif( ${PARFLOW_ENABLE_TIMING} )
  

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


