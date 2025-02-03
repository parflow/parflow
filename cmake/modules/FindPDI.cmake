include(FindPackageHandleStandardArgs)

if(NOT PDI_ROOT)
    set(PDI_ROOT $ENV{PDI_ROOT})
endif()

find_path(PDI_INCLUDE_DIR NAMES pdi.h paraconf.h
  PATH_SUFFIXES pdi
  HINTS ${PDI_ROOT}/include
  PATHS /usr/include /usr/local/include)

find_library(PDI_LIBRARY NAMES pdi yaml paraconf spdlog
  HINTS ${PDI_ROOT}/lib
  PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib)

set(PDI_INCLUDE_DIRS ${PDI_INCLUDE_DIR})
set(PDI_LIBRARIES ${PDI_LIBRARY})

FIND_PACKAGE_HANDLE_STANDARD_ARGS(PDI DEFAULT_MSG PDI_LIBRARIES PDI_INCLUDE_DIRS)

MARK_AS_ADVANCED(PDI_INCLUDE_DIRS PDI_LIBRARIES)

find_path(PARACONF_INCLUDE_DIR NAMES paraconf.h
  PATH_SUFFIXES pdi
  HINTS ${PDI_ROOT}/include
  PATHS /usr/include /usr/local/include)

find_library(PARACONF_LIBRARY NAMES paraconf
  HINTS ${PDI_ROOT}/lib
  PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib)

set(PARACONF_INCLUDE_DIRS ${PARACONF_INCLUDE_DIR})
set(PARACONF_LIBRARIES ${PARACONF_LIBRARY})

FIND_PACKAGE_HANDLE_STANDARD_ARGS(PARACONF DEFAULT_MSG PARACONF_LIBRARIES PARACONF_INCLUDE_DIRS)

MARK_AS_ADVANCED(PARACONF_INCLUDE_DIRS PARACONF_LIBRARIES)


# Find SPDLOG library (libspdlog.so)
find_library(SPDLOG_LIBRARY NAMES spdlog
  HINTS ${PDI_ROOT}/lib
  PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib)

# Validate that SPDLOG is found
if(NOT SPDLOG_LIBRARY)
    message(FATAL_ERROR "spdlog library not found in the specified paths!")
endif()

# Set SPDLOG variables
set(SPDLOG_LIBRARIES ${SPDLOG_LIBRARY})

# Mark SPDLOG as advanced
MARK_AS_ADVANCED(SPDLOG_LIBRARIES)
