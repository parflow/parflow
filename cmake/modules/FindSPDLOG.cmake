# FindSPDLOG.cmake
#
# Find spdlog
# ------------
#
# This module locates the spdlog library
#
# It sets the following variables:
#
#   SPDLOG_LIBRARIES   - The spdlog library (e.g. libspdlog.so)
#
# The module allows the user to specify SPDLOG_ROOT (or set the environment variable SPDLOG_ROOT)
# to guide the search in non-standard locations.
# If SPDLOG_ROOT is not provided, and PDI_ROOT is defined, it defaults to ${PDI_ROOT}.
# 
# Usage:
#   find_package(SPDLOG_ROOT REQUIRED)


include(FindPackageHandleStandardArgs)

# Allow user to specify SPDLOG_ROOT; if not set, try to retrieve from the environment.
if(NOT SPDLOG_ROOT)
  set(SPDLOG_ROOT $ENV{SPDLOG_ROOT})
endif()

# If SPDLOG_ROOT is still not set and PDI_ROOT is defined, default it to "${PDI_ROOT}"
if(NOT SPDLOG_ROOT AND DEFINED PDI_ROOT)
  set(SPDLOG_ROOT "${PDI_ROOT}")
endif()

# Locate the spdlog library (typically libspdlog.so)
find_library(SPDLOG_LIBRARY
  NAMES spdlog
  HINTS ${SPDLOG_ROOT}/lib
  PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
)

# Set output variables
set(SPDLOG_LIBRARIES ${SPDLOG_LIBRARY})

# Handle standard arguments and define SPDLOG_FOUND
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SPDLOG DEFAULT_MSG SPDLOG_LIBRARIES)

# Mark the found variables as advanced to reduce cache clutter.
MARK_AS_ADVANCED(SPDLOG_LIBRARIES)
