#.rst:
# FindP4EST
# --------
#
# Find p4est library
#
# This module finds an installed p4est library.
#
# This module sets the following variables:
#
# ::
#
#   P4EST_FOUND - set to true if a P4EST library is found
#   P4EST_INCLUDE_DIR - the P4EST and SC include directories
#   P4EST_LIBRARIES - the P4EST and SC libraries
#

include(FindPackageHandleStandardArgs)

if(NOT P4EST_ROOT)
    set(P4EST_ROOT $ENV{P4EST_ROOT})
endif()

if (DEFINED P4EST_ROOT)

  find_path(P4EST_INCLUDE_DIR NAMES p4est.h
    PATH_SUFFIXES p4est
    PATHS ${P4EST_ROOT}/include
    NO_DEFAULT_PATH)

  find_library(P4EST_LIBRARY NAMES p4est
    HINTS ${P4EST_ROOT}/lib
    NO_DEFAULT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH)

  find_library(SC_LIBRARY NAMES sc
    PATHS ${P4EST_ROOT}/lib
    NO_DEFAULT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH)

  list(APPEND P4EST_LIBRARY ${SC_LIBRARY})

endif (DEFINED P4EST_ROOT)

set(P4EST_INCLUDE_DIRS ${P4EST_INCLUDE_DIR})
set(P4EST_LIBRARIES ${P4EST_LIBRARY})

FIND_PACKAGE_HANDLE_STANDARD_ARGS(P4EST DEFAULT_MSG P4EST_LIBRARIES P4EST_INCLUDE_DIRS)

MARK_AS_ADVANCED(P4EST_INCLUDE_DIRS P4EST_LIBRARIES)  
