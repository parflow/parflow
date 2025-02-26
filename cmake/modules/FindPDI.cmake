# FindPDI.cmake
# -------------
#
# This module locates the PDI (Parallel Data Interface) library and sets the following variables:
#
#   PDI_FOUND - Set to TRUE if PDI is found
#   PDI_INCLUDE_DIRS - The directory containing pdi.h
#   PDI_LIBRARIES - The PDI library file
#
# Usage:
#   find_package(PDI REQUIRED)
#


include(FindPackageHandleStandardArgs)

if(NOT PDI_ROOT)
    set(PDI_ROOT $ENV{PDI_ROOT})
endif()

find_path(PDI_INCLUDE_DIR NAMES pdi.h
  PATH_SUFFIXES pdi
  HINTS ${PDI_ROOT}/include
  PATHS /usr/include /usr/local/include)

find_library(PDI_LIBRARY NAMES pdi
  HINTS ${PDI_ROOT}/lib
  PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib)

set(PDI_INCLUDE_DIRS ${PDI_INCLUDE_DIR})
set(PDI_LIBRARIES ${PDI_LIBRARY})

FIND_PACKAGE_HANDLE_STANDARD_ARGS(PDI DEFAULT_MSG PDI_LIBRARIES PDI_INCLUDE_DIRS)

MARK_AS_ADVANCED(PDI_INCLUDE_DIRS PDI_LIBRARIES)