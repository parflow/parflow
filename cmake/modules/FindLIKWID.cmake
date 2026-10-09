# FindLIKWID.cmake
# -------------
#
# This module locates the LIKWID library and sets the following variables:
#
#   LIKWID_FOUND - Set to TRUE if LIKWID is found
#   LIKWID_INCLUDE_DIRS - The directory containing likwid.h
#   LIKWID_LIBRARIES - The LIKWID library
#
# Usage:
#   find_package(LIKWID REQUIRED)
#

include(FindPackageHandleStandardArgs)

if ( NOT LIKWID_ROOT )
    set ( LIKWID_ROOT $ENV{LIKWID_ROOT} )
endif()

find_path ( LIKWID_INCLUDE_DIR likwid.h
        HINTS $ENV{LIKWID_INCDIR}
        $ENV{LIKWID_ROOT}/include
)

find_library ( LIKWID_LIB likwid
        HINTS $ENV{LIKWID_LIBDIR}
        $ENV{LIKWID_ROOT}/lib
)

FIND_PACKAGE_HANDLE_STANDARD_ARGS ( LIKWID DEFAULT_MSG LIKWID_LIB LIKWID_INCLUDE_DIR )

set ( LIKWID_INCLUDE_DIRS ${LIKWID_INCLUDE_DIR} )
set ( LIKWID_LIBRARIES ${LIKWID_LIB} )

MARK_AS_ADVANCED ( LIKWID_INCLUDE_DIRS LIKWID_LIBRARIES )
