#.rst:
# FindSLURM
# --------
#
# Find SLURM library
#
# This module finds an installed the SLURM library.
#
# This module sets the following variables:
#
# ::
#
#   SLURM_FOUND - set to true if a SLURM library is found
#   SLURM_INCLUDE_DIR - the SLURM include directory
#   SLURM_LIBRARIES - the SLURM libraries
#

include(FindPackageHandleStandardArgs)

if(NOT SLURM_ROOT)
    set(SLURM_ROOT $ENV{SLURM_ROOT})
endif()

find_path(SLURM_INCLUDE_DIR NAMES slurm/slurm.h
                            PATH_SUFFIXES slurm
                            HINTS ${SLURM_ROOT}/include /opt/slurm/default /usr/local)

find_library(SLURM_LIBRARY NAMES slurm
			    HINTS ${SLURM_ROOT}/lib64 ${SLURM_ROOT}/lib 
			    PATHS /usr/lib64 /usr/lib)

set(SLURM_INCLUDE_DIRS ${SLURM_INCLUDE_DIR})
set(SLURM_LIBRARIES ${SLURM_LIBRARY})

FIND_PACKAGE_HANDLE_STANDARD_ARGS(SLURM DEFAULT_MSG SLURM_LIBRARIES SLURM_INCLUDE_DIRS)

MARK_AS_ADVANCED(SLURM_INCLUDE_DIRS SLURM_LIBRARIES)



