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
#   HYPRE_FOUND - set to true if a HYPRE library is found
#   HYPRE_INCLUDE_DIR - the HYPRE include directory
#   HYPRE_LIBRARIES - the HYPRE libraries
#

include(FindPackageHandleStandardArgs)

if(NOT HYPRE_ROOT)
  set(HYPRE_ROOT $ENV{HYPRE_ROOT})
endif()

# If Hypre root is set then search only in that directory for Hypre
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

FIND_PACKAGE_HANDLE_STANDARD_ARGS(HYPRE DEFAULT_MSG HYPRE_LIBRARIES HYPRE_INCLUDE_DIRS)

MARK_AS_ADVANCED(HYPRE_INCLUDE_DIRS HYPRE_LIBRARIES)



