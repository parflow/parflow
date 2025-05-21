#.rst:
# FindHypre
# --------
#
# Find Hypre library
#
# This module finds an installed the Hypre library.
#
# This module sets the following variables:
#
# ::
#
#   Hypre_FOUND        - set to true if a HYPRE library is found
#   HYPRE_INCLUDE_DIRS - the HYPRE include directory
#   HYPRE_LIBRARIES    - the HYPRE libraries
#   Hypre::Hypre       - Hypre CMake target

include(FindPackageHandleStandardArgs)

if(NOT HYPRE_ROOT)
  set(HYPRE_ROOT $ENV{HYPRE_ROOT})
endif()

#
# If Hypre root is set then search only in that directory for Hypre
#
if (DEFINED HYPRE_ROOT)

  find_path(HYPRE_INCLUDE_DIR NAMES HYPRE.h
    PATH_SUFFIXES hypre
    PATHS ${HYPRE_ROOT}/include
    NO_DEFAULT_PATH)

  find_library(HYPRE_LIBRARY_LS NAMES HYPRE_struct_ls
    PATHS ${HYPRE_ROOT}/lib
    NO_DEFAULT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH)

  if(HYPRE_LIBRARY_LS)
    set(HYPRE_LIBRARIES ${HYPRE_LIBRARY_LS})

    find_library(HYPRE_LIBRARY_MV NAMES HYPRE_struct_mv
      PATHS ${HYPRE_ROOT}/lib
      NO_DEFAULT_PATH
      NO_SYSTEM_ENVIRONMENT_PATH)
          
    if(HYPRE_LIBRARY_MV)
      list(APPEND HYPRE_LIBRARIES ${HYPRE_LIBRARY_MV})
    endif()
  else()
    find_library(HYPRE_LIBRARY NAMES HYPRE HYPRE-64
      PATHS ${HYPRE_ROOT}/lib
      NO_DEFAULT_PATH
      NO_SYSTEM_ENVIRONMENT_PATH)

    set(HYPRE_LIBRARIES ${HYPRE_LIBRARY})    
  endif()
  
else (DEFINED HYPRE_ROOT)

  find_path(HYPRE_INCLUDE_DIR NAMES HYPRE.h
    PATH_SUFFIXES hypre
    HINTS ${HYPRE_ROOT}/include
    PATHS /usr/include/openmpi-x86_64 /usr/include)
  
  # Search first for specific HYPRE libraries; on ubuntu the HYPRE.so is broken and empty.
  find_library(HYPRE_LIBRARY_LS NAMES HYPRE_struct_ls
    HINTS ${HYPRE_ROOT}/lib
    PATHS /usr/lib64/openmpi/lib /usr/lib64 /lib64 /usr/lib /lib)
  
  if(HYPRE_LIBRARY_LS)
    set(HYPRE_LIBRARIES ${HYPRE_LIBRARY_LS})
    
    find_library(HYPRE_LIBRARY_MV NAMES HYPRE_struct_mv
      HINTS ${HYPRE_ROOT}/lib
      PATHS /usr/lib64/openmpi/lib /usr/lib64 /lib64 /usr/lib /lib)
    
    if(HYPRE_LIBRARY_MV)
      list(APPEND HYPRE_LIBRARIES ${HYPRE_LIBRARY_MV})
    endif()
  else()
    find_library(HYPRE_LIBRARY NAMES HYPRE HYPRE-64
      HINTS ${HYPRE_ROOT}/lib
      PATHS /usr/lib64/openmpi/lib /usr/lib64 /lib64 /usr/lib /lib)
    
    set(HYPRE_LIBRARIES ${HYPRE_LIBRARY})    
  endif()
  
endif (DEFINED HYPRE_ROOT)

set(HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIR})

#
# Check Hypre features from HYPRE_config.h
#
set(HYPRE_CONFIG_H "${HYPRE_INCLUDE_DIRS}/HYPRE_config.h")
if (EXISTS "${HYPRE_CONFIG_H}")
  file(STRINGS ${HYPRE_CONFIG_H} HYPRE_USING_CUDA REGEX "HYPRE_USING_CUDA")
  string(REGEX MATCH "[0-9]+" HYPRE_USING_CUDA "${HYPRE_USING_CUDA}")
else()
  set(HYPRE_USING_CUDA 0)
endif()

# TODO: Remove debugging statements
message(STATUS "PARFLOW_HAVE_CUDA=${PARFLOW_HAVE_CUDA}")
message(STATUS "HYPRE_USING_CUDA=${HYPRE_USING_CUDA}")
message(STATUS "HYPRE_CONFIG_H=${HYPRE_CONFIG_H}")

find_package_handle_standard_args(Hypre DEFAULT_MSG HYPRE_LIBRARIES HYPRE_INCLUDE_DIRS)
mark_as_advanced(HYPRE_INCLUDE_DIRS HYPRE_LIBRARIES)

#
# Create target Hypre::Hypre
#
if (Hypre_FOUND AND NOT TARGET Hypre::Hypre)
  add_library(Hypre::Hypre INTERFACE IMPORTED)
  target_include_directories(Hypre::Hypre INTERFACE ${HYPRE_INCLUDE_DIRS})
  target_link_libraries(Hypre::Hypre INTERFACE ${HYPRE_LIBRARIES})

  if(${PARFLOW_HAVE_CUDA} AND ${HYPRE_USING_CUDA})
    # Hypre-CUDA requires linking to cuBLAS, cuRAND, and cuSPARSE
    # See full list of imported CUDA targets here: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html#imported-targets
    include(FindCUDAToolkit)
    if (CUDAToolkit_FOUND)
      if(TARGET CUDA::cublas_static)
        list(APPEND _cuda_targets CUDA::cublas_static)
      elseif(TARGET CUDA::cublas)
        list(APPEND _cuda_targets CUDA::cublas)
      elseif(TARGET CUDA::cublasLt_static)
        list(APPEND _cuda_targets CUDA::cublasLt_static)
      elseif(TARGET CUDA::cublasLt)
        list(APPEND _cuda_targets CUDA::cublasLt)
      endif()

      if(TARGET CUDA::curand_static)
        list(APPEND _cuda_targets CUDA::curand_static)
      elseif(TARGET CUDA::curand)
        list(APPEND _cuda_targets CUDA::curand)
      endif()

      if(TARGET CUDA::cusparse_static)
        list(APPEND _cuda_targets CUDA::cusparse_static)
      elseif(TARGET CUDA::cusparse)
        list(APPEND _cuda_targets CUDA::cusparse)
      endif()

      string(JOIN ", " _hypre_cuda_targets ${_cuda_targets})
      message(STATUS "Found Hypre built with CUDA. ParFlow will be linked to: ${_hypre_cuda_targets}")
      target_link_libraries(Hypre::Hypre INTERFACE ${_cuda_targets})

      unset(_cuda_targets)
      unset(_hypre_cuda_targets)
    endif()
  endif()
endif()
