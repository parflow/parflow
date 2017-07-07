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

find_path(HYPRE_INCLUDE_DIR NAMES HYPRE.h
  PATH_SUFFIXES hypre
  HINTS ${HYPRE_ROOT}/include
  PATHS /usr/include)

# Search first for specific HYPRE libraries; on ubuntu the HYPRE.so is broken and empty.
message(STATUS "Looking for HYPRE_struct_ls")
find_library(HYPRE_LIBRARY NAMES HYPRE_struct_ls
  HINTS ${HYPRE_ROOT}/lib
  PATHS /usr/lib64 /lib64 /usr/lib /lib)

message(STATUS "Found ${HYPRE_LIBRARY}")

if(HYPRE_LIBRARY)
  set(HYPRE_LIBRARIES ${HYPRE_LIBRARY})

  message(STATUS "Looking for HYPRE_struct_mv")
  find_library(HYPRE_LIBRARY NAMES HYPRE_struct_mv
    HINTS ${HYPRE_ROOT}/lib
    PATHS /usr/lib64 /lib64 /usr/lib /lib)
          
  message(STATUS "Found ${HYPRE_LIBRARY}")
  if(HYPRE_LIBRARY)
    list(APPEND HYPRE_LIBRARIES ${HYPRE_LIBRARY})
  endif()
else()
  find_library(HYPRE_LIBRARY NAMES HYPRE HYPRE-64
    HINTS ${HYPRE_ROOT}/lib
    PATHS /usr/lib64 /lib64 /usr/lib /lib)

  message(STATUS "Found ${HYPRE_LIBRARY}")
  set(HYPRE_LIBRARIES ${HYPRE_LIBRARY})    
endif()

message(STATUS "Found HYPRE_LIBRARIES ${HYPRE_LIBRARIES}")

set(HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIR})

FIND_PACKAGE_HANDLE_STANDARD_ARGS(HYPRE DEFAULT_MSG HYPRE_LIBRARIES HYPRE_INCLUDE_DIRS)

MARK_AS_ADVANCED(HYPRE_INCLUDE_DIRS HYPRE_LIBRARIES)



