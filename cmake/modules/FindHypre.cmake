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

if(NOT BUILD_SHARED_LIBS)
  set(EXT ".a")
else()
  set(EXT "")
endif()

message(STATUS "Looing for Ext ${EXT}")

find_library(HYPRE_LIBRARY NAMES libHYPRE${EXT} libHYPRE-64${EXT}
  HINTS ${HYPRE_ROOT}/lib
  PATHS /usr/lib)

# Following checks were needed on Ubuntu distributions.   libHYPRE.so is empty.
if(HYPRE_LIBRARY)
  set(HYPRE_LIBRARIES ${HYPRE_LIBRARY})    
else()
  message(STATUS "Looking for libHYPRE_struct_ls${EXT}")
  find_library(HYPRE_LIBRARY NAMES libHYPRE_struct_ls${EXT}
    HINTS ${HYPRE_ROOT}/lib
    PATHS /usr/lib /lib)

  message(STATUS "Found ${HYPRE_LIBRARY}")
  if(HYPRE_LIBRARY)
    set(HYPRE_LIBRARIES ${HYPRE_LIBRARY})

      message(STATUS "Looking for libHYPRE_struct_mv${EXT}")
    find_library(HYPRE_LIBRARY NAMES libHYPRE_struct_mv${EXT}
      HINTS ${HYPRE_ROOT}/lib
      PATHS /usr/lib /lib)
          
    message(STATUS "Found ${HYPRE_LIBRARY}")
    if(HYPRE_LIBRARY)
      list(APPEND HYPRE_LIBRARIES ${HYPRE_LIBRARY})
    endif()
  endif()
endif()

message(STATUS "Found ${HYPRE_LIBRARIES}")

set(HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIR})

FIND_PACKAGE_HANDLE_STANDARD_ARGS(HYPRE DEFAULT_MSG HYPRE_LIBRARIES HYPRE_INCLUDE_DIRS)

MARK_AS_ADVANCED(HYPRE_INCLUDE_DIRS HYPRE_LIBRARIES)



