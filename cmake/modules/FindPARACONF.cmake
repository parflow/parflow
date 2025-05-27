# FindPARACONF.cmake
# -------------------
#
# This module locates the Paraconf library and sets the following variables:
#
#   PARACONF_INCLUDE_DIRS - The directory containing paraconf.h
#   PARACONF_LIBRARIES - The Paraconf library file
#
# The module allows the user to specify PARACONF_ROOT (or set the environment variable PARACONF_ROOT)
# to guide the search in non-standard locations.
# If PARACONF_ROOT is not provided, and PDI_ROOT is defined, it defaults to ${PDI_ROOT}.
# 
# Usage:
#   find_package(PARACONF REQUIRED)

include(FindPackageHandleStandardArgs)

if(NOT PARACONF_ROOT)
    set(PARACONF_ROOT $ENV{PARACONF_ROOT})
endif()


# If PARACONF_ROOT is still not set and PDI_ROOT is defined, default it to "${PDI_ROOT}"
if(NOT PARACONF_ROOT AND DEFINED PDI_ROOT)
  set(PARACONF_ROOT "${PDI_ROOT}")
endif()


find_path(PARACONF_INCLUDE_DIR NAMES paraconf.h
  PATH_SUFFIXES pdi
  HINTS ${PARACONF_ROOT}/include
  PATHS /usr/include /usr/local/include)

find_library(PARACONF_LIBRARY NAMES paraconf
  HINTS ${PARACONF_ROOT}/lib
  PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib)

set(PARACONF_INCLUDE_DIRS ${PARACONF_INCLUDE_DIR})
set(PARACONF_LIBRARIES ${PARACONF_LIBRARY})

# Ensure proper package handling
find_package_handle_standard_args(PARACONF DEFAULT_MSG PARACONF_LIBRARIES PARACONF_INCLUDE_DIRS)

mark_as_advanced(PARACONF_INCLUDE_DIRS PARACONF_LIBRARIES)