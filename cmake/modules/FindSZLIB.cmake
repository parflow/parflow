#.rst:
# FindZLIB
# --------
#
# Find the native ZLIB includes and library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   ZLIB_INCLUDE_DIRS   - where to find zlib.h, etc.
#   ZLIB_LIBRARIES      - List of libraries when using zlib.
#   ZLIB_FOUND          - True if zlib found.
#
# ::
#
# Hints
# ^^^^^
#
# A user may set ``SZLIB_ROOT`` to a szlib installation root to tell this
# module where to look.

set(_SZLIB_SEARCHES)

# Search ZLIB_ROOT first if it is set.
if(SZLIB_ROOT)
  set(_SZLIB_SEARCH_ROOT PATHS ${SZLIB_ROOT} NO_DEFAULT_PATH)
  list(APPEND _SZLIB_SEARCHES _SZLIB_SEARCH_ROOT)
endif()

set(SZLIB_NAMES sz )

# Try each search configuration.
foreach(search ${_SZLIB_SEARCHES})
  find_path(SZLIB_INCLUDE_DIR NAMES szlib.h ${${search}} PATH_SUFFIXES include)
endforeach()

# Allow SZLIB_LIBRARY to be set manually, as the location of the zlib library
if(NOT SZLIB_LIBRARY)
  foreach(search ${_SZLIB_SEARCHES})
    find_library(SZLIB_LIBRARY NAMES ${SZLIB_NAMES} ${${search}} PATH_SUFFIXES lib)
  endforeach()
endif()

unset(SZLIB_NAMES)

mark_as_advanced(SZLIB_LIBRARY SZLIB_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SZLIB REQUIRED_VARS SZLIB_LIBRARY SZLIB_INCLUDE_DIR)

if(SZLIB_FOUND)
  set(SZLIB_INCLUDE_DIRS ${SZLIB_INCLUDE_DIR})
  
  if(NOT SZLIB_LIBRARIES)
    set(SZLIB_LIBRARIES ${SZLIB_LIBRARY})
  endif(NOT SZLIB_LIBRARIES)
endif(SZLIB_FOUND)
