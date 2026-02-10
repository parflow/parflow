cmake_minimum_required (VERSION 3.22)

project (PARFLOW VERSION 3.14.1 LANGUAGES C Fortran CXX)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")

#cmake_policy(SET CMP0144 NEW)

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
# The ParFlow C library source will be built as C++ if this flag is set.
# ParFlow is moving to using C++; this flag is a first step towards that goal.
# Does not currently change pftools build.
#-----------------------------------------------------------------------------

set(PARFLOW_BUILD_WITH_CPP FALSE CACHE STRING "Build ParFlow using the C++ compiler.  This flag is principlly for developers.")

#-----------------------------------------------------------------------------
# Set accelerator backend
#-----------------------------------------------------------------------------
set(PARFLOW_ACCELERATOR_BACKEND "none" CACHE STRING "Set accelerator backend")
set_property(CACHE PARFLOW_ACCELERATOR_BACKEND PROPERTY STRINGS none cuda kokkos omp)

if(PARFLOW_ACCELERATOR_BACKEND STREQUAL "none")
elseif(PARFLOW_ACCELERATOR_BACKEND STREQUAL "cuda")
   message(STATUS "ACCELERATOR: CUDA backend selected")
  set(PARFLOW_HAVE_CUDA "yes")

  if(NOT DEFINED KOKKOS_ROOT)
    # Initialize CUDA
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
      set(CMAKE_CUDA_STANDARD 14)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
      if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70")
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER 11)
          set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80")
        endif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER 11)
      endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
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
  # Check Kokkos installation root
  if(DEFINED KOKKOS_ROOT)
     # Find and configure Kokkos
    find_package(Kokkos REQUIRED HINTS ${KOKKOS_ROOT})
    set(KOKKOS_INCLUDE "${KOKKOS_ROOT}/include")
    set(PARFLOW_HAVE_KOKKOS "yes")

    # Select the active Kokkos backend
    if(Kokkos_ENABLE_CUDA)
      message(STATUS "ACCELERATOR: Kokkos-enabled CUDA backend selected")
      set(KOKKOS_ENABLE_CUDA "yes")

    elseif(Kokkos_ENABLE_HIP)
      message(STATUS "ACCELERATOR: Kokkos-enabled HIP backend selected")
      set(KOKKOS_ENABLE_HIP "yes")

    elseif(Kokkos_ENABLE_SYCL)
      message(STATUS "ACCELERATOR: Kokkos-enabled SYCL backend selected")
      set(KOKKOS_ENABLE_SYCL "yes")

    elseif(Kokkos_ENABLE_OPENMP_TARGET)
      message(STATUS "ACCELERATOR: Kokkos-enabled OpenMP target-offload backend selected")
      set(KOKKOS_ENABLE_OPENMP_TARGET "yes")

    else(Kokkos_ENABLE_CUDA)  # host-only backend
      message(STATUS "Kokkos host-only backend selected")
      set(KOKKOS_ENABLE_SERIAL "yes")
    endif(Kokkos_ENABLE_CUDA)  # end Kokkos active backend select

  else(DEFINED KOKKOS_ROOT)  # KOKKOS_ROOT not set
    message(FATAL_ERROR "ERROR: KOKKOS_ROOT not specified with PARFLOW_ACCELERATOR_BACKEND=kokkos")
  endif(DEFINED KOKKOS_ROOT)  # end Kokkos root check

  elseif(PARFLOW_ACCELERATOR_BACKEND STREQUAL "omp")
  message(STATUS "ACCELERATOR: Compiling ParFlow with backend accelerator OpenMP")
  # Enable C and CXX -fopenmp flag, enable ParFlow defines.
  find_package(OpenMP REQUIRED)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
  set(PARFLOW_HAVE_OMP "yes")

else(PARFLOW_ACCELERATOR_BACKEND STREQUAL "none")
  message(FATAL_ERROR "ERROR: Unknown backend type! PARFLOW_ACCELERATOR_BACKEND=${PARFLOW_ACCELERATOR_BACKEND} does not exist!")
endif(PARFLOW_ACCELERATOR_BACKEND STREQUAL "none")

# Include UMPIRE or RMM memory manager for pool allocation
if(PARFLOW_HAVE_CUDA OR PARFLOW_HAVE_KOKKOS)

  if (DEFINED UMPIRE_ROOT AND DEFINED RMM_ROOT)
    message(FATAL_ERROR "ERROR: Cannot have both UMPIRE and RMM enabled at the same time.")
  endif(DEFINED UMPIRE_ROOT AND DEFINED RMM_ROOT)
  
  # Determine memory manager and print single status message
  if(DEFINED UMPIRE_ROOT)
    find_package(umpire REQUIRED HINTS ${UMPIRE_ROOT})
    set(PARFLOW_HAVE_UMPIRE "yes")
    message(STATUS "MEMORY MANAGER: UMPIRE is sleceted")
   
  elseif(DEFINED RMM_ROOT)
    find_package(rmm REQUIRED HINTS ${RMM_ROOT})
    set(PARFLOW_HAVE_RMM "yes")
    message(STATUS "MEMORY MANAGER: RMM is sleceted")
   
  else(DEFINED UMPIRE_ROOT)
    message(STATUS "MEMORY MANAGER: NO memoery manager is sleceted")
  endif(DEFINED UMPIRE_ROOT)
endif(PARFLOW_HAVE_CUDA OR PARFLOW_HAVE_KOKKOS)

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

if (NOT(PARFLOW_AMPS_SEQUENTIAL_IO) AND PARFLOW_ENABLE_PYTHON)
  message(FATAL_ERROR "ERROR: Python does not support the multiple file AMPS I/O model for PFB output")
endif ()

# OAS3
if (${PARFLOW_AMPS_LAYER} STREQUAL "oas3")
  find_package(OASIS REQUIRED)
  if (OASIS_FOUND)
    set (PARFLOW_HAVE_OAS3 "yes")
    set (HAVE_OAS3 ${PARFLOW_HAVE_OAS3})
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

    # Need to pass MPI include header to nvcc to avoid missing <mpi.h> error at amps.h.
    if (${PARFLOW_HAVE_CUDA})
      if (MPI_C_INCLUDE_DIRS)
        # Construct the compiler string for the include directories.
        foreach(dir ${MPI_C_INCLUDE_DIRS})
          string(APPEND CUDA_MPI_INCLUDES "-I${dir} ")
        endforeach()
      else()
        # Guess MPI include dir from MPI_HOME (if it exists)
        # https://gitlab.kitware.com/cmake/cmake/-/blob/master/Modules/FindMPI.cmake?ref_type=heads#L120
        if(DEFINED CACHE{MPI_HOME})
          set(CUDA_MPI_INCLUDES -I${MPI_HOME}/include)
       elseif(DEFINED ENV{MPI_HOME})
          set(CUDA_MPI_INCLUDES -I$ENV{MPI_HOME}/include)
        endif()
      endif()

      if(CUDA_MPI_INCLUDES)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_MPI_INCLUDES}")
      else()
        message(WARNING "MPI include dir not found. This might cause compile problems when building CUDA sources with MPI headers.")
      endif()
    endif()
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
  find_package(SILO)
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
  find_package (NetCDF REQUIRED)
  if (${NetCDF_FOUND})
    set(PARFLOW_HAVE_NETCDF "yes")
    set(HAVE_NETCDF ${PARFLOW_HAVE_NETCDF})
    find_package(CURL)
    if (NOT NETCDF_HAS_PARALLEL)
      # ParFlow requires parallel NetCDF header file (used in parflow_netcdf.h and write_parflow_netcdf.c).
      message(FATAL_ERROR "ParFlow requires NetCDF with parallel support.")
    endif()
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
# PDI
#-----------------------------------------------------------------------------
set (PARFLOW_ENABLE_PDI False CACHE BOOL "Build with PDI")
if (${PARFLOW_ENABLE_PDI} OR DEFINED PDI_ROOT)
  find_package(PDI)
  if (${PDI_FOUND})
    set(PARFLOW_HAVE_PDI "yes")
    # Additional dependencies required when PDI is enabled.
    # These libraries must be found to ensure proper functionality.
    find_package(PARACONF REQUIRED)
    find_package(SPDLOG REQUIRED)
  endif (${PDI_FOUND})
endif (${PARFLOW_ENABLE_PDI} OR DEFINED PDI_ROOT)

#-----------------------------------------------------------------------------
# Hypre
#-----------------------------------------------------------------------------
option(PARFLOW_ENABLE_HYPRE "Build with Hypre" "OFF")
option(PARFLOW_ENABLE_HYPRE_LOGGING "Enable detailed Hypre logging and residual output" "OFF")
if (${PARFLOW_ENABLE_HYPRE} OR DEFINED HYPRE_ROOT)
  find_package(HYPRE)
  if (${HYPRE_FOUND})
    set(PARFLOW_HAVE_HYPRE "yes")
    set(HAVE_HYPRE ${PARFLOW_HAVE_HYPRE})

    file(STRINGS ${HYPRE_INCLUDE_DIR}/HYPRE_config.h hypreConfig REGEX HYPRE_RELEASE_VERSION)
    separate_arguments(hypreConfig)

    list(GET hypreConfig 2 PARFLOW_HYPRE_VERSION)

    version_create_variables (PARFLOW_HYPRE)

    if (PARFLOW_ENABLE_HYPRE_LOGGING)
      set(PARFLOW_ENABLE_HYPRE_LOGGING ${PARFLOW_ENABLE_HYPRE_LOGGING})
    endif()

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
  find_package(SUNDIALS COMPONENTS sundials_cvode sundials_kinsol sundials_core sundials_sunlinsolspgmr)
  if (${SUNDIALS_FOUND})
    set(PARFLOW_HAVE_SUNDIALS "yes")
    set(HAVE_SUNDIALS ${PARFLOW_HAVE_SUNDIALS})
  endif (${SUNDIALS_FOUND})
else ()
    message (STATUS "Found KINSOL: embedded")
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
CHECK_Fortran_SOURCE_COMPILES("program main
  implicit none

  write ( *, '(a)' ) '  Hello, world!'

  stop
end" FORTRAN_F90_WORKS SRC_EXT .F90)

#
# Determine syntax for writing binary file under Fortran
#

# Check whether the Fortran compiler supports the access="stream" open syntax
CHECK_Fortran_SOURCE_COMPILES("program freeform
   open(10, file='test.bin', access='stream', form='unformatted', status='replace')
   write(10) \"first\"
   write(10) \"second\"
   close(UNIT=10)
 end program freeform" HAVE_FC_ACCESS_STREAM SRC_EXT .F90)

# Check whether the Fortran compiler supports the access="sequential" open syntax
CHECK_Fortran_SOURCE_COMPILES("program freeform
  open(10, file='test.bin', access='sequential', form='binary', status='replace')
  write(10) \"first\"
  write(10) \"second\"
  close(UNIT=10)
end program freeform" HAVE_FC_ACCESS_SEQUENTIAL SRC_EXT .F90)

#
# Set implicit none flag on Fortran compiles
#
include(CheckFortranCompilerFlag)

set(SAVE_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})

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
include(CheckFunctionExists)

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

# Check if libm is required

# Check if acos is available without additional libraries
check_function_exists(acos HAVE_ACOS)

if(NOT HAVE_ACOS)
  # Save the current required libraries
  set(CMAKE_REQUIRED_LIBRARIES_SAVE ${CMAKE_REQUIRED_LIBRARIES})
  
  # Try with libm
  set(CMAKE_REQUIRED_LIBRARIES m)
  check_function_exists(acos HAVE_ACOS_WITH_LIBM)
  
  # Restore the original required libraries
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES_SAVE})
  
  if(HAVE_ACOS_WITH_LIBM)
    set(MATH_LIBRARY m)
    message(STATUS "Linking with libm math library")
  else()
    message(STATUS "acos function not found in libm math library")
  endif()
endif()

#-----------------------------------------------------------------------------
# Check compiling with CLM / ECLM
#-----------------------------------------------------------------------------

option(PARFLOW_HAVE_CLM "Compile with CLM" "OFF")

if ( ${PARFLOW_HAVE_CLM} )
  # Make true value match autoconf value; for some backwards compatibility
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
  find_package(Python3 3.7 QUIET REQUIRED COMPONENTS Interpreter)
  set(PARFLOW_PYTHON_DEPENDS)
  set(PARFLOW_HAVE_PYTHON)
  set(PARFLOW_PYTHON "${CMAKE_INSTALL_PREFIX}/py-env/bin/python3")
  message(STATUS "Found Python: ${Python3_EXECUTABLE}")
  add_custom_command(
    WORKING_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}"
    OUTPUT
      "${PARFLOW_PYTHON}"
    COMMAND
      ${Python3_EXECUTABLE} -m venv ${CMAKE_INSTALL_PREFIX}/py-env
    COMMAND
      "${CMAKE_INSTALL_PREFIX}//py-env/bin/pip3" install --upgrade pip wheel setuptools
    COMMAND
      "${CMAKE_INSTALL_PREFIX}/py-env/bin/pip3" install -r "${CMAKE_CURRENT_SOURCE_DIR}/pftools/python/requirements_all.txt"
    COMMAND
      "${CMAKE_INSTALL_PREFIX}//py-env/bin/pip3" install -r "${CMAKE_CURRENT_SOURCE_DIR}/docs/user_manual/requirements.txt"
    COMMAND
      "${CMAKE_INSTALL_PREFIX}//py-env/bin/pip3" install -r "${CMAKE_CURRENT_SOURCE_DIR}/pftools/python/requirements_dev.txt"
    COMMENT
      "Generate ParFlow virtual environment"
    VERBATIM
    )
    add_custom_target(
      BuildVirtualEnv
      DEPENDS
        "${PARFLOW_PYTHON}"
    )
    set(PARFLOW_PYTHON_DEPENDS BuildVirtualEnv)
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

# Optionally build the NUOPC cap and ESMX
option(PARFLOW_ENABLE_NUOPC "Enable building of the Parflow nuopc" "False")
option(PARFLOW_ENABLE_ESMX "Enable building of ESMX" "False")
if ( ${PARFLOW_ENABLE_NUOPC} OR ${PARFLOW_ENABLE_ESMX} )
  add_subdirectory (pfnuopc)
endif ()

#-----------------------------------------------------------------------------
# Setup CTEST environment
#-----------------------------------------------------------------------------
include (CTest)

option(PARFLOW_ENABLE_LATEX "Enable LaTEX and building of documentation" "OFF")
if (PARFLOW_ENABLE_LATEX)
  add_subdirectory(docs/manuals)
endif()

install (DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/config/ DESTINATION config)

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
