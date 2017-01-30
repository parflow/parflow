#.rst:
# FindSilo
# --------
#
# Find Silo library
#
# This module finds an installed the Silo library.
#
# This module sets the following variables:
#
# ::
#
#   SILO_FOUND - set to true if a SILO library is found
#   SILO_INCLUDE_DIR - the SILO include directory
#   SILO_LIBRARIES - the SILO libraries
#

include(FindPackageHandleStandardArgs)

if(NOT SILO_ROOT)
    set(SILO_ROOT $ENV{SILO_ROOT})
endif()

find_path(SILO_INCLUDE_DIR NAMES silo.h
                            PATH_SUFFIXES silo
                            HINTS ${SILO_ROOT}/include)

if(NOT BUILD_SHARED_LIBS)
  find_library(SILO_LIBRARY NAMES libsiloh5.a HINTS ${SILO_ROOT}/lib)
else()
  find_library(SILO_LIBRARY NAMES siloh5 silo HINTS ${SILO_ROOT}/lib)
endif()

set(SILO_INCLUDE_DIRS ${SILO_INCLUDE_DIR})
set(SILO_LIBRARIES ${SILO_LIBRARY})

FIND_PACKAGE_HANDLE_STANDARD_ARGS(SILO DEFAULT_MSG SILO_LIBRARIES SILO_INCLUDE_DIRS)

MARK_AS_ADVANCED(SILO_INCLUDE_DIRS SILO_LIBRARIES)



