cmake_minimum_required (VERSION 3.9)

project (PARFLOW C Fortran CXX)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_C_STANDARD 11)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

#-----------------------------------------------------------------------------
# Version number
#-----------------------------------------------------------------------------
include (Version)
#
# Make a version file containing the current version from git.
#

include(GetGitRevisionDescription)
git_describe(PARFLOW_VERSION --tags)

# If not building with git then get version from file
if (NOT PARFLOW_VERSION)
  file (STRINGS "VERSION" PARFLOW_VERSION)
endif ()
 
message("Configuring version : ${PARFLOW_VERSION}")
version_create_variables (PARFLOW)

# enable testing
enable_testing ()

#-----------------------------------------------------------------------------
# General project wide configuration
#-----------------------------------------------------------------------------
# TODO should get rid of non-prefix versions of flags; preprocessor flags should be in PARFLOW namespace

# TODO replace CASC macro names with PARFLOW when completed.

# Use RPATH in install, many mpicc scripts use RPATH so default
# behavior of CMAKE to remove RPATH from installed executables is not
# so good.
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

# Set AMPS communication layer
set(PARFLOW_AMPS_LAYER "seq" CACHE STRING "Set the Communications layer to use")
set_property(CACHE PARFLOW_AMPS_LAYER PROPERTY STRINGS seq mpi1 smpi oas3 win32)
set(AMPS ${PARFLOW_AMPS_LAYER})

option(PARFLOW_AMPS_SEQUENTIAL_IO "Use AMPS single file I/O model for output of PFB files" "FALSE")

if (${PARFLOW_AMPS_SEQUENTIAL_IO})
  message("Using single file AMPS I/O for PFB output")
else ()
  message("Using multiple file AMPS I/O for PFB output")
  set(AMPS_SPLIT_FILE "yes")
endif ()

# OAS3
option(PARFLOW_HAVE_OAS3 "Build with OAS3" "no")

if (${PARFLOW_HAVE_OAS3})
  set (PARFLOW_HAVE_OAS3 "yes")
  set (HAVE_OAS3 ${PARFLOW_HAVE_OAS3})
endif (${PARFLOW_HAVE_OAS3})

set (PARFLOW_AMPS_LAYER_REQUIRE_MPI mpi1 smpi oas3)

# Check for MPI only if AMPS requires it
if ( ${PARFLOW_AMPS_LAYER} IN_LIST PARFLOW_AMPS_LAYER_REQUIRE_MPI )

  find_package(MPI)

  if (${MPI_C_FOUND})
    set(PARFLOW_HAVE_MPI "yes")
    set(HAVE_MPI ${PARFLOW_HAVE_MPI})
  endif (${MPI_C_FOUND})

endif ( ${PARFLOW_AMPS_LAYER} IN_LIST PARFLOW_AMPS_LAYER_REQUIRE_MPI )

find_package(TCL)
if (${TCL_FOUND})
  set(PARFLOW_HAVE_TCL "yes")
  set(HAVE_TCL ${PARFLOW_HAVE_TCL})
else (${TCL_FOUND})
  if (${PARFLOW_ENABLE_TOOLS})
    message(FATAL_ERROR "TCL is required for building pftools")
  endif (${PARFLOW_ENABLE_TOOLS})
endif (${TCL_FOUND})

#-----------------------------------------------------------------------------
# SILO
#-----------------------------------------------------------------------------
set (PARFLOW_ENABLE_SILO False CACHE BOOL "Build with Silo")
if (${PARFLOW_ENABLE_SILO} OR DEFINED SILO_ROOT)
  find_package(Silo)
  if (${SILO_FOUND})
    set(PARFLOW_HAVE_SILO "yes")
    set(HAVE_SILO ${PARFLOW_HAVE_SILO})
  endif (${SILO_FOUND})
endif (${PARFLOW_ENABLE_SILO} OR DEFINED SILO_ROOT)

#-----------------------------------------------------------------------------
# NetCDF
#-----------------------------------------------------------------------------
set (PARFLOW_ENABLE_NETCDF False CACHE BOOL "Build with NetCDF")
if (${PARFLOW_ENABLE_NETCDF} OR DEFINED NETCDF_DIR OR DEFINED NETCDF_INCLUDE_DIR OR DEFINED NETCDF_LIBRARY)
  find_package (NetCDF)
  if (${NETCDF_FOUND})
    set(PARFLOW_HAVE_NETCDF "yes")
    set(HAVE_NETCDF ${PARFLOW_HAVE_NETCDF})
  endif (${NETCDF_FOUND})
endif (${PARFLOW_ENABLE_NETCDF} OR DEFINED NETCDF_DIR OR DEFINED NETCDF_INCLUDE_DIR OR DEFINED NETCDF_LIBRARY)

#-----------------------------------------------------------------------------
# HDF5
#-----------------------------------------------------------------------------
set (PARFLOW_ENABLE_HDF5 False CACHE BOOL "Build with HDF5")
if (${PARFLOW_ENABLE_HDF5} OR DEFINED HDF5_ROOT)
  set(HDF5_PREFER_PARALLEL True)

  if (${PARFLOW_HAVE_NETCDF})
    set(PARFLOW_HDF5_COMPONENTS C HL)
  else()
    set(PARFLOW_HDF5_COMPONENTS C)
  endif()

  find_package(HDF5 COMPONENTS ${PARFLOW_HDF5_COMPONENTS})
  if (${HDF5_FOUND})
    set(PARFLOW_HAVE_HDF5 "yes")
    set(HAVE_HDF5 ${PARFLOW_HAVE_HDF5})
  endif (${HDF5_FOUND})
endif (${PARFLOW_ENABLE_HDF5} OR DEFINED HDF5_ROOT)

#-----------------------------------------------------------------------------
# Hypre
#-----------------------------------------------------------------------------
set (PARFLOW_ENABLE_HYPRE False CACHE BOOL "Build with Hypre")
if (${PARFLOW_ENABLE_HYPRE} OR DEFINED HYPRE_ROOT)
  find_package(Hypre)
  if (${HYPRE_FOUND})
    set(PARFLOW_HAVE_HYPRE "yes")
    set(HAVE_HYPRE ${PARFLOW_HAVE_HYPRE})

    file(STRINGS ${HYPRE_INCLUDE_DIR}/HYPRE_config.h hypreConfig REGEX HYPRE_RELEASE_VERSION)
    separate_arguments(hypreConfig)

    list(GET hypreConfig 2 PARFLOW_HYPRE_VERSION)

    version_create_variables (PARFLOW_HYPRE)

  endif (${HYPRE_FOUND})
endif (${PARFLOW_ENABLE_HYPRE} OR DEFINED HYPRE_ROOT)

#-----------------------------------------------------------------------------
# ZLIB
#-----------------------------------------------------------------------------
set (PARFLOW_ENABLE_ZLIB False CACHE BOOL "Build with Zlib compression library")
if (${PARFLOW_ENABLE_ZLIB} OR DEFINED ZLIB_ROOT)
  find_package(ZLIB)
  if (${ZLIB_FOUND})
    set(PARFLOW_HAVE_ZLIB "yes")
  endif (${ZLIB_FOUND})
endif (${PARFLOW_ENABLE_ZLIB} OR DEFINED ZLIB_ROOT)

#-----------------------------------------------------------------------------
# SZLIB
#-----------------------------------------------------------------------------
set (PARFLOW_ENABLE_SZLIB False CACHE BOOL "Build with SZlib compression library")
if (${PARFLOW_ENABLE_SZLIB} OR DEFINED SZLIB_ROOT)
  find_package(SZLIB)
  if (${SZLIB_FOUND})
    set(PARFLOW_HAVE_SZLIB "yes")
  endif (${SZLIB_FOUND})
endif (${PARFLOW_ENABLE_SZLIB} OR DEFINED SZLIB_ROOT)

#-----------------------------------------------------------------------------
# Sundials
#-----------------------------------------------------------------------------
set (PARFLOW_ENABLE_SUNDIALS False CACHE BOOL "Build with SZlib compression library")
if (${PARFLOW_ENABLE_SUNDIALS} OR DEFINED SUNDIALS_ROOT)
  find_package(SUNDIALS COMPONENTS sundials_cvode sundials_kinsol)
  if (${SUNDIALS_FOUND})
    set(PARFLOW_HAVE_SUNDIALS "yes")
    set(HAVE_SUNDIALS ${PARFLOW_HAVE_SUNDIALS})
  endif (${SUNDIALS_FOUND})
endif (${PARFLOW_ENABLE_SUNDIALS} OR DEFINED SUNDIALS_ROOT)

#-----------------------------------------------------------------------------
# SLURM
#-----------------------------------------------------------------------------
set (PARFLOW_ENABLE_SLURM False CACHE BOOL "Build with SLURM support")
if (${PARFLOW_ENABLE_SLURM} OR DEFINED SLURM_ROOT)
  find_package(SLURM)
  if (${SLURM_FOUND})
    set(PARFLOW_HAVE_SLURM "yes")
    set(HAVE_SLURM ${PARFLOW_HAVE_SLURM})
  endif (${SLURM_FOUND})
endif (${PARFLOW_ENABLE_SLURM} OR DEFINED SLURM_ROOT)


#-----------------------------------------------------------------------------
# libm
#-----------------------------------------------------------------------------
if (NOT DEFINED PARFLOW_LIBM)
  find_library(PARFLOW_LIBM m)
endif (NOT DEFINED PARFLOW_LIBM)		    

#-----------------------------------------------------------------------------
# Valgrind
#-----------------------------------------------------------------------------

set (PARFLOW_ENABLE_VALGRIND False CACHE BOOL "Build with Valgrind support")
if (${PARFLOW_ENABLE_VALGRIND} )
  find_program( PARFLOW_MEMORYCHECK_COMMAND valgrind)
  set(PARFLOW_HAVE_MEMORYCHECK "yes")
  set(PARFLOW_MEMORYCHECK_SUPPRESSIONS_FILE "${PROJECT_SOURCE_DIR}/bin/valgrind.sup")
  set(PARFLOW_MEMORYCHECK_COMMAND_OPTIONS "--gen-suppressions=all --leak-check=full --suppressions=${PARFLOW_MEMORYCHECK_SUPPRESSIONS_FILE}")
endif (${PARFLOW_ENABLE_VALGRIND})

#-----------------------------------------------------------------------------
# Ptrace
#-----------------------------------------------------------------------------
set (PARFLOW_ENABLE_ETRACE False CACHE BOOL "Build with etrace")
if (${PARFLOW_ENABLE_ETRACE})
  set(PARFLOW_HAVE_ETRACE "yes")
  
  set(PARFLOW_ETRACE_COMPILE_FLAGS "-finstrument-functions")
  set(PARFLOW_ETRACE_LIBRARY "etrace")
else (${PARFLOW_ENABLE_ETRACE})
  set(PARFLOW_ETRACE_COMPILE_FLAGS "")
  set(PARFLOW_ETRACE_LIBRARY "")
endif (${PARFLOW_ENABLE_ETRACE})

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

#
# Check for platform specific features
#

include (TestBigEndian)
include(CheckSymbolExists)
include(CheckIncludeFiles)

test_big_endian(PARFLOW_HAVE_BIG_ENDIAN)
set(CASC_HAVE_BIGENDIAN ${PARFLOW_HAVE_BIG_ENDIAN})

# Check for gettimeofday
check_symbol_exists(gettimeofday sys/time.h PARFLOW_HAVE_GETTIMEOFDAY)
if ( ${PARFLOW_HAVE_GETTIMEOFDAY} )
  set(CASC_HAVE_GETTIMEOFDAY ${PARFLOW_HAVE_GETTIMEOFDAY})
endif ( ${PARFLOW_HAVE_GETTIMEOFDAY} )

check_include_files (malloc.h PARFLOW_HAVE_MALLOC_H)
if ( ${PARFLOW_HAVE_MALLOC_H} )
  set(HAVE_MALLOC_H ${PARFLOW_HAVE_MALLOC_H})
endif ( ${PARFLOW_HAVE_MALLOC_H} )

# Check for mallinfo
check_symbol_exists(mallinfo malloc.h PARFLOW_HAVE_MALLINFO)
if ( ${PARFLOW_HAVE_MALLINFO} )
  set(HAVE_MALLINFO ${PARFLOW_HAVE_MALLINFO})
endif ( ${PARFLOW_HAVE_MALLINFO} )

option(PARFLOW_HAVE_CLM "Compile with CLM" "no")

if ( ${PARFLOW_HAVE_CLM} )
  # Make true value match autoconf value; for some backwards compatiblity
  set(PARFLOW_HAVE_CLM "yes")
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

# Profiling
set (PARFLOW_ENABLE_PROFILING False CACHE BOOL "Enable profiling; will slow down performance")
if( ${PARFLOW_ENABLE_PROFILING} )
  set (PARFLOW_PROFILE_OPTS "-pg")
  set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${PARFLOW_PROFILE_OPTS}" )
  set( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${PARFLOW_PROFILE_OPTS}" )
else( ${PARFLOW_ENABLE_PROFILING} )
  set (PARFLOW_PROFILE_OPTS "")
endif( ${PARFLOW_ENABLE_PROFILING} )

include_directories ("${CMAKE_SOURCE_DIR}/pfsimulator/parflow_lib")

include_directories ("${PROJECT_SOURCE_DIR}/pfsimulator/amps/${PARFLOW_AMPS_LAYER}")
include_directories ("${PROJECT_SOURCE_DIR}/pfsimulator/amps/common")

include_directories ("${PROJECT_BINARY_DIR}/include")

#-----------------------------------------------------------------------------
# Setup configure.h file for accessing configure options
# -----------------------------------------------------------------------------
configure_file (cmake/parflow_config.h.in include/parflow_config.h)

configure_file (cmake/pfversion.h.in include/pfversion.h)

configure_file (cmake/Makefile.config.in config/Makefile.config)
configure_file (cmake/pf-cmake-env.sh.in config/pf-cmake-env.sh)

if ( ${PARFLOW_HAVE_CLM} )
  configure_file (pfsimulator/clm/parflow_config.F90.in ${PROJECT_BINARY_DIR}/pfsimulator/clm/parflow_config.F90)
endif ( ${PARFLOW_HAVE_CLM} )

#-----------------------------------------------------------------------------
# CMAKE Subdirectories
#-----------------------------------------------------------------------------

# Need to turn on testing so tests in subdirctories are included in test target.
enable_testing ()

# Optionally build the simulator and/or tools.
# This is used on architectures where the login node is a different architecture
# than the compute nodes.   The simulator is built for the compute nodes; tools
# is built for the login node.
set (PARFLOW_ENABLE_SIMULATOR True CACHE BOOL "Enable building of the Parflow simulator")
if ( ${PARFLOW_ENABLE_SIMULATOR} )
  add_subdirectory (pfsimulator)
  add_subdirectory (test)
  add_subdirectory (examples)
endif ()

set (PARFLOW_ENABLE_TOOLS True CACHE BOOL "Enable building of the Parflow tools")
if ( ${PARFLOW_ENABLE_TOOLS} )
  add_subdirectory (pftools)
endif ()

#-----------------------------------------------------------------------------
# Setup CTEST environment
#-----------------------------------------------------------------------------
include (CTest)

set (PARFLOW_ENABLE_LATEX False CACHE BOOL "Enable LaTEX and building of documentation")
if ( ${PARFLOW_ENABLE_LATEX} )
  add_subdirectory(docs/manuals)
endif()

install (DIRECTORY ${CMAKE_BINARY_DIR}/config/ DESTINATION config)
# Test script for docker builds
install(PROGRAMS bin/parflow-docker-test DESTINATION bin)

#-----------------------------------------------------------------------------
# Doxygen 
#-----------------------------------------------------------------------------
set (PARFLOW_ENABLE_DOXYGEN False CACHE BOOL "Enable Doxygen and building of code documentation")

if ( ${PARFLOW_ENABLE_DOXYGEN} )
  find_package(Doxygen)
  if (DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs/doxygen)
    set(DOXYGEN_IMAGE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/images)
    
    message("Doxygen build started")

    doxygen_add_docs(doxygen
      COMMENT "Generating API documentation with Doxygen"
      ${PROJECT_SOURCE_DIR})

  else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")
  endif (DOXYGEN_FOUND)
endif()
