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
                            HINTS ${HYPRE_ROOT}/include)

if(NOT BUILD_SHARED_LIBS)
  find_library(HYPRE_LIBRARY NAMES libHYPRE.a libHYPRE-64.a HINTS ${HYPRE_ROOT}/lib)
else()
  find_library(HYPRE_LIBRARY NAMES HYPRE HYPRE-64 HINTS ${HYPRE_ROOT}/lib)
endif()

set(HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIR})
set(HYPRE_LIBRARIES ${HYPRE_LIBRARY})

FIND_PACKAGE_HANDLE_STANDARD_ARGS(HYPRE DEFAULT_MSG HYPRE_LIBRARIES HYPRE_INCLUDE_DIRS)

MARK_AS_ADVANCED(HYPRE_INCLUDE_DIRS HYPRE_LIBRARIES)



