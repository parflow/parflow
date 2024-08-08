cmake_minimum_required (VERSION 3.14)

project (PARFLOW LANGUAGES C Fortran CXX)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

#-----------------------------------------------------------------------------
# Set standard compiler flagsx
#-----------------------------------------------------------------------------
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_C_STANDARD 11)

include (CheckCCompilerFlag)
include (CheckCXXCompilerFlag)

# Check if compiler supports extra debugging flags
foreach (DEBUG_FLAG "-ggdb" "-g3")
  check_C_compiler_flag(${DEBUG_FLAG} FOUND)
  if(${FOUND})
    set(CMAKE_C_FLAGS_DEBUG ${DEBUG_FLAG})
    break ()
  endif ()
endforeach ()

foreach (DEBUG_FLAG "-ggdb" "-g3")
  check_CXX_compiler_flag(${DEBUG_FLAG} FOUND)
  if(${FOUND})
    set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAG})
    break ()
  endif ()
endforeach ()

#-----------------------------------------------------------------------------
# Set accelerator backend
#-----------------------------------------------------------------------------
set(PARFLOW_ACCELERATOR_BACKEND "none" CACHE STRING "Set accelerator backend")
set_property(CACHE PARFLOW_ACCELERATOR_BACKEND PROPERTY STRINGS none cuda kokkos omp)

if(PARFLOW_ACCELERATOR_BACKEND STREQUAL "none")
elseif(PARFLOW_ACCELERATOR_BACKEND STREQUAL "cuda")
  set(PARFLOW_HAVE_CUDA "yes")
  if(NOT DEFINED KOKKOS_ROOT)
    # Initialize CUDA
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
      set(CMAKE_CUDA_STANDARD 14)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 --expt-extended-lambda")
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER 11)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80")
      endif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER 11)
      set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G --ptxas-options=-v")
      set(CMAKE_CUDA_HOST_COMPILER mpicxx)
      message(STATUS "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION} at ${CMAKE_CUDA_COMPILER}")
    else(CMAKE_CUDA_COMPILER)
      message(FATAL_ERROR "ERROR: CUDA installation not found")
    endif(CMAKE_CUDA_COMPILER)
  else(NOT DEFINED KOKKOS_ROOT)
    # Allow mixed use of CUDA and Kokkos for development purposes
    find_package(Kokkos REQUIRED HINTS ${KOKKOS_ROOT})
    set(KOKKOS_INCLUDE "${KOKKOS_ROOT}/include")
    set(PARFLOW_HAVE_KOKKOS "yes")
  endif(NOT DEFINED KOKKOS_ROOT)
elseif(PARFLOW_ACCELERATOR_BACKEND STREQUAL "kokkos")
  if(DEFINED KOKKOS_ROOT)
    find_package(Kokkos REQUIRED HINTS ${KOKKOS_ROOT})
    set(KOKKOS_INCLUDE "${KOKKOS_ROOT}/include")
    set(PARFLOW_HAVE_KOKKOS "yes")
  else(DEFINED KOKKOS_ROOT)
    message(FATAL_ERROR "ERROR: KOKKOS_ROOT not specified with PARFLOW_ACCELERATOR_BACKEND=kokkos")
  endif(DEFINED KOKKOS_ROOT)
elseif(PARFLOW_ACCELERATOR_BACKEND STREQUAL "omp")
  message(STATUS "ACCELERATOR: Compiling ParFlow with backend accelerator OpenMP")
  # Enable C and CXX -fopenmp flag, enable ParFlow defines.
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  set(PARFLOW_HAVE_OMP "yes")
else()
  message(FATAL_ERROR "ERROR: Unknown backend type! PARFLOW_ACCELERATOR_BACKEND=${PARFLOW_ACCELERATOR_BACKEND} does not exist!")
endif()

# Include RapidsAI Memory Manager (RMM) for pool allocation
if((${PARFLOW_HAVE_CUDA}) OR (${PARFLOW_HAVE_KOKKOS}))
  if(DEFINED RMM_ROOT)
    find_path(RMM_INCLUDE "rmm" HINTS "${RMM_ROOT}/include")
    find_library(RMM_LIBRARY "rmm" HINTS "${RMM_ROOT}/lib")
    if(RMM_INCLUDE AND RMM_LIBRARY)
      message(STATUS "RMM: RMM_LIBRARY set to ${RMM_LIBRARY}")
      message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")
      add_library(rmm SHARED IMPORTED ${RMM_LIBRARY})
      set_target_properties(rmm PROPERTIES IMPORTED_LOCATION ${RMM_LIBRARY})
    # PARFLOW_HAVE_RMM is not defined in parflow_config.h because because it is
    # only set for the compilation of parflow_exe, parflow_lib, and kinsol sources.
    # RMM must be initialized before it is used, so, for example,
    # the amps tests would fail if used across all compiles.
      set(PARFLOW_HAVE_RMM "yes")
    else(RMM_INCLUDE AND RMM_LIBRARY)
      message(FATAL_ERROR "ERROR: RMM installation not found")
    endif(RMM_INCLUDE AND RMM_LIBRARY)
    if(${PARFLOW_HAVE_CUDA}) 
      message(STATUS "ACCELERATOR: Compiling with CUDA and RMM")
    else(${PARFLOW_HAVE_CUDA}) 
      message(STATUS "ACCELERATOR: Compiling with Kokkos and RMM")
    endif(${PARFLOW_HAVE_CUDA}) 
  else(DEFINED RMM_ROOT)
    if(${PARFLOW_HAVE_CUDA}) 
      message(STATUS "ACCELERATOR: Compiling with CUDA (RMM not used)")
    else(${PARFLOW_HAVE_CUDA}) 
      message(STATUS "ACCELERATOR: Compiling with Kokkos (RMM not used)")
    endif(${PARFLOW_HAVE_CUDA}) 
  endif(DEFINED RMM_ROOT)
endif((${PARFLOW_HAVE_CUDA}) OR (${PARFLOW_HAVE_KOKKOS}))

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

message(STATUS "Configuring version : ${PARFLOW_VERSION}")
version_create_variables (PARFLOW)

#-----------------------------------------------------------------------------
# System information
#-----------------------------------------------------------------------------
cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)

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
set(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
   set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")

# Set AMPS communication layer
set(PARFLOW_AMPS_LAYER "seq" CACHE STRING "Set the Communications layer to use")
set_property(CACHE PARFLOW_AMPS_LAYER PROPERTY STRINGS seq mpi1 cuda smpi oas3 win32)
set(AMPS ${PARFLOW_AMPS_LAYER})

if((NOT PARFLOW_HAVE_CUDA) AND (${PARFLOW_AMPS_LAYER} STREQUAL "cuda"))
  message(FATAL_ERROR "ERROR: Using PARFLOW_AMPS_LAYER=cuda requires building with GPU acceleration!")
endif((NOT PARFLOW_HAVE_CUDA) AND (${PARFLOW_AMPS_LAYER} STREQUAL "cuda"))

option(PARFLOW_AMPS_SEQUENTIAL_IO "Use AMPS single file I/O model for output of PFB files" "TRUE")

if (PARFLOW_AMPS_SEQUENTIAL_IO)
  message(STATUS "Using single file AMPS I/O for PFB output")
else ()
  message(STATUS "Using multiple file AMPS I/O for PFB output")
  set(AMPS_SPLIT_FILE "yes")
endif ()

# OAS3
if (${PARFLOW_AMPS_LAYER} STREQUAL "oas3")
  find_package(OASIS)
  if (OASIS_FOUND)
    find_package(NetCDF-Fortran)
    if (NetCDF-Fortran_FOUND)
      set (PARFLOW_HAVE_OAS3 "yes")
      set (HAVE_OAS3 ${PARFLOW_HAVE_OAS3})
    else (NetCDF-Fortran_FOUND)
      message(FATAL_ERROR "ERROR: NetCDF-Fortran is required when compiling with PARFLOW_AMPS_LAYER=oas3 (check NETCDF_Fortran_ROOT variable)")
    endif (NetCDF-Fortran_FOUND)
  else (OASIS_FOUND)
    message(FATAL_ERROR "ERROR: OASIS is required when compiling with PARFLOW_AMPS_LAYER=oas3 (check OAS3_ROOT variable)")
  endif (OASIS_FOUND)
endif (${PARFLOW_AMPS_LAYER} STREQUAL "oas3")

set (PARFLOW_AMPS_LAYER_REQUIRE_MPI cuda mpi1 smpi oas3)
set (PARFLOW_TEST_PARALLEL cuda mpi1 smpi)
option(PARFLOW_TEST_FORCE_MPIEXEC "Always use MPIEXEC runner for MPI applications in tests" "OFF")

# Check for MPI only if AMPS requires it
if ( ${PARFLOW_AMPS_LAYER} IN_LIST PARFLOW_AMPS_LAYER_REQUIRE_MPI )

  find_package(MPI)

  if (${MPI_C_FOUND})
    set(PARFLOW_HAVE_MPI "yes")
    set(HAVE_MPI ${PARFLOW_HAVE_MPI})
    message(STATUS "Found MPIEXEC: ${MPIEXEC}")    
  endif (${MPI_C_FOUND})

endif ( ${PARFLOW_AMPS_LAYER} IN_LIST PARFLOW_AMPS_LAYER_REQUIRE_MPI )

find_package(TCL QUIET)
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
option(PARFLOW_ENABLE_SILO "Build with Silo" "OFF")
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
option(PARFLOW_ENABLE_NETCDF "Build with NetCDF" "OFF")
if (${PARFLOW_ENABLE_NETCDF} OR DEFINED NETCDF_DIR OR DEFINED NETCDF_INCLUDE_DIR OR DEFINED NETCDF_LIBRARY)
  find_package (NetCDF)
  if (${NetCDF_FOUND})
    set(PARFLOW_HAVE_NETCDF "yes")
    set(HAVE_NETCDF ${PARFLOW_HAVE_NETCDF})
    find_package(CURL)
  endif (${NetCDF_FOUND})
endif (${PARFLOW_ENABLE_NETCDF} OR DEFINED NETCDF_DIR OR DEFINED NETCDF_INCLUDE_DIR OR DEFINED NETCDF_LIBRARY)

#-----------------------------------------------------------------------------
# HDF5
#-----------------------------------------------------------------------------
option(PARFLOW_ENABLE_HDF5 "Build with HDF5" "OFF")
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
option(PARFLOW_ENABLE_HYPRE "Build with Hypre" "OFF")
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
option(PARFLOW_ENABLE_ZLIB "Build with Zlib compression library" "OFF")
if (${PARFLOW_ENABLE_ZLIB} OR DEFINED ZLIB_ROOT)
  find_package(ZLIB)
  if (${ZLIB_FOUND})
    set(PARFLOW_HAVE_ZLIB "yes")
  endif (${ZLIB_FOUND})
endif (${PARFLOW_ENABLE_ZLIB} OR DEFINED ZLIB_ROOT)

#-----------------------------------------------------------------------------
# SZLIB
#-----------------------------------------------------------------------------
option(PARFLOW_ENABLE_SZLIB "Build with SZlib compression library" "OFF")
if (${PARFLOW_ENABLE_SZLIB} OR DEFINED SZLIB_ROOT)
  find_package(SZLIB)
  if (${SZLIB_FOUND})
    set(PARFLOW_HAVE_SZLIB "yes")
  endif (${SZLIB_FOUND})
endif (${PARFLOW_ENABLE_SZLIB} OR DEFINED SZLIB_ROOT)

#-----------------------------------------------------------------------------
# Sundials
#-----------------------------------------------------------------------------
option(PARFLOW_ENABLE_SUNDIALS "Build with SUNDIALS" "OFF")
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
option(PARFLOW_ENABLE_SLURM "Build with SLURM support" "OFF")
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
option(PARFLOW_ENABLE_VALGRIND "Build with Valgrind support" "OFF")
if (${PARFLOW_ENABLE_VALGRIND} )
  find_program( PARFLOW_MEMORYCHECK_COMMAND valgrind)
  set(PARFLOW_HAVE_MEMORYCHECK "yes")
  set(PARFLOW_MEMORYCHECK_SUPPRESSIONS_FILE "${PROJECT_SOURCE_DIR}/bin/valgrind.sup")
  set(PARFLOW_MEMORYCHECK_COMMAND_OPTIONS "--gen-suppressions=all --leak-check=full --suppressions=${PARFLOW_MEMORYCHECK_SUPPRESSIONS_FILE}")
endif (${PARFLOW_ENABLE_VALGRIND})

#-----------------------------------------------------------------------------
# Ptrace
#-----------------------------------------------------------------------------
option(PARFLOW_ENABLE_ETRACE "Build with etrace" "OFF")
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
include(CheckFortranSourceCompiles)

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

include(CheckCSourceCompiles)

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
check_symbol_exists(mallinfo2 malloc.h PARFLOW_HAVE_MALLINFO2)
check_symbol_exists(mallinfo malloc.h PARFLOW_HAVE_MALLINFO)

option(PARFLOW_HAVE_CLM "Compile with CLM" "OFF")

if ( ${PARFLOW_HAVE_CLM} )
  # Make true value match autoconf value; for some backwards compatiblity
  set(PARFLOW_HAVE_CLM "yes")
  set(HAVE_CLM ${PARFLOW_HAVE_CLM})

  if( ${PARFLOW_HAVE_CLM} )
    add_subdirectory (pfsimulator/clm)
  endif( ${PARFLOW_HAVE_CLM} )
endif ( ${PARFLOW_HAVE_CLM} )

option(PARFLOW_HAVE_ECLM "Compile with eCLM (experimental)" "OFF")

if ( ${PARFLOW_HAVE_ECLM} )
  if (HAVE_CLM)
    message(FATAL_ERROR "ERROR: PARFLOW_HAVE_CLM and PARFLOW_HAVE_ECLM cannot be enabled at the same time.")
  endif (HAVE_CLM)
  if (NOT HAVE_OAS3)
    message(FATAL_ERROR "ERROR: OASIS is required when compiling with PARFLOW_HAVE_ECLM=ON (check OAS3_ROOT variable)")
  else (HAVE_OAS3)
    message(WARNING "Enabling components for ParFlow-eCLM coupled model (experimental feature!)")
    set (HAVE_ECLM ${PARFLOW_HAVE_ECLM})
  endif (HAVE_OAS3)
endif ( ${PARFLOW_HAVE_ECLM} )
#
# Parflow specific configuration options
#

# Control timing of Parflow functions.
option(PARFLOW_ENABLE_TIMING "Enable timing of key Parflow functions; may slow down performance" "False")
if(PARFLOW_ENABLE_TIMING)
  set (PF_TIMING ${PARFLOW_ENABLE_TIMING})
endif(PARFLOW_ENABLE_TIMING)

# Profiling
option(PARFLOW_ENABLE_PROFILING "Enable profiling; will slow down performance" "False")
if(PARFLOW_ENABLE_PROFILING)
  set(PARFLOW_PROFILE_OPTS "-pg")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${PARFLOW_PROFILE_OPTS}" )
  set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${PARFLOW_PROFILE_OPTS}" )
else(PARFLOW_ENABLE_PROFILING)
  set (PARFLOW_PROFILE_OPTS "")
endif(PARFLOW_ENABLE_PROFILING)

include_directories ("${CMAKE_SOURCE_DIR}/pfsimulator/parflow_lib")

include_directories ("${PROJECT_SOURCE_DIR}/pfsimulator/amps/${PARFLOW_AMPS_LAYER}")
include_directories ("${PROJECT_SOURCE_DIR}/pfsimulator/amps/common")

include_directories ("${PROJECT_BINARY_DIR}/include")

#-----------------------------------------------------------------------------
# Building Python key library and documentation if required
#-----------------------------------------------------------------------------

option(PARFLOW_ENABLE_PYTHON "Build python module for running ParFlow" "FALSE")
if (PARFLOW_ENABLE_PYTHON)
  find_package(Python3 3.6 QUIET REQUIRED COMPONENTS Interpreter)
  set(PARFLOW_PYTHON ${Python3_EXECUTABLE})
  set(PARFLOW_PYTHON_DEPENDS)
  set(PARFLOW_HAVE_PYTHON)

  option(PARFLOW_PYTHON_VIRTUAL_ENV "Use local Python virtual environment" "FALSE")
  if (PARFLOW_PYTHON_VIRTUAL_ENV)
    set(PARFLOW_PYTHON "${CMAKE_CURRENT_BINARY_DIR}/py-env/bin/python3")
    add_custom_command(
      WORKING_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}"
      OUTPUT
        "${PARFLOW_PYTHON}"
      COMMAND
        ${Python3_EXECUTABLE} -m venv py-env
      COMMAND
        "${CMAKE_CURRENT_BINARY_DIR}/py-env/bin/pip3" install --upgrade pip wheel setuptools
      COMMAND
        "${CMAKE_CURRENT_BINARY_DIR}/py-env/bin/pip3" install -r "${CMAKE_CURRENT_SOURCE_DIR}/pftools/python/requirements_all.txt"
      COMMAND
        "${CMAKE_CURRENT_BINARY_DIR}/py-env/bin/pip3" install -r "${CMAKE_CURRENT_SOURCE_DIR}/docs/user_manual/requirements.txt"
      COMMAND
        "${CMAKE_CURRENT_BINARY_DIR}/py-env/bin/pip3" install -r "${CMAKE_CURRENT_SOURCE_DIR}/pftools/python/requirements_dev.txt"
      COMMENT
        "Generate virtual environment for building Python components"
      VERBATIM
    )
    add_custom_target(
      BuildVirtualEnv
      DEPENDS
        "${PARFLOW_PYTHON}"
    )
    set(PARFLOW_PYTHON_DEPENDS BuildVirtualEnv)
  endif ()
endif ()

#-----------------------------------------------------------------------------
# Setup configure.h file for accessing configure options
# -----------------------------------------------------------------------------
configure_file (cmake/parflow_config.h.in include/parflow_config.h)

configure_file (cmake/pfversion.h.in include/pfversion.h)

configure_file (cmake/Makefile.config.in config/Makefile.config)
configure_file (cmake/pf-cmake-env.sh.in config/pf-cmake-env.sh)
configure_file (cmake/parflow.settings.in config/parflow.settings)

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
option(PARFLOW_ENABLE_SIMULATOR "Enable building of the Parflow simulator" "True")
if ( ${PARFLOW_ENABLE_SIMULATOR} )
  add_subdirectory (pfsimulator)
  add_subdirectory (test)
  add_subdirectory (examples)

  #-----------------------------------------------------------------------------
  # Install general helper scripts
  #-----------------------------------------------------------------------------
  install(PROGRAMS bin/pfrun DESTINATION bin)
endif ()

option(PARFLOW_ENABLE_TOOLS "Enable building of the Parflow tools" "True")
if ( ${PARFLOW_ENABLE_TOOLS} )
  add_subdirectory (pftools)
endif ()

#-----------------------------------------------------------------------------
# Setup CTEST environment
#-----------------------------------------------------------------------------
include (CTest)

option(PARFLOW_ENABLE_LATEX "Enable LaTEX and building of documentation" "OFF")
if (PARFLOW_ENABLE_LATEX)
  add_subdirectory(docs/manuals)
endif()

install (DIRECTORY ${CMAKE_BINARY_DIR}/config/ DESTINATION config)

#-----------------------------------------------------------------------------
# Doxygen
#-----------------------------------------------------------------------------
option(PARFLOW_ENABLE_DOXYGEN "Enable Doxygen and building of code documentation" "OFF")

if ( ${PARFLOW_ENABLE_DOXYGEN} )
  find_package(Doxygen)
  if (DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs/doxygen)
    set(DOXYGEN_IMAGE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/images)

    message(STATUS "Doxygen build started")

    doxygen_add_docs(doxygen
      COMMENT "Generating API documentation with Doxygen"
      ${PROJECT_SOURCE_DIR})

  else (DOXYGEN_FOUND)
    message(WARNING "Doxygens need to be installed to generate the doxygen documentation")
  endif (DOXYGEN_FOUND)
endif()

#-----------------------------------------------------------------------------
# Python directories
#-----------------------------------------------------------------------------

if (PARFLOW_ENABLE_PYTHON)
  add_subdirectory(pf-keys)
  add_subdirectory(pftools/python)
  add_subdirectory(test/python)

  # Only enable Key Documentation if we enable Python
  option(PARFLOW_ENABLE_KEYS_DOC "Build key documentation for ParFlow" "FALSE")
  if (PARFLOW_ENABLE_KEYS_DOC)
    add_subdirectory(docs/user_manual)
  endif()
endif()

#-----------------------------------------------------------------------------
# Building Docker
#-----------------------------------------------------------------------------

option(PARFLOW_ENABLE_DOCKER "Build Docker for running ParFlow" "FALSE")
if (PARFLOW_ENABLE_DOCKER)
  add_subdirectory(docker)
endif()
