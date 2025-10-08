#[==[
Provides the following variables:

  * `NetCDF_FOUND`: Whether NetCDF was found or not.
  * `NETCDF_INCLUDE_DIRS`: Include directories necessary to use NetCDF.
  * `NETCDF_LIBRARIES`: Libraries necessary to use NetCDF.
  * `NETCDF_VERSION`: The version of NetCDF found.
  * `NETCDF_HAS_PARALLEL`: Whether or not NetCDF was found with parallel IO support.
  * `NetCDF::NetCDF`: A target to use with `target_link_libraries`.
#]==]

if (NOT (NETCDF_DIR OR NETCDF_INCLUDE_DIR OR NetCDF_LIBRARY))
  #
  # Strategy 1: Search NetCDF via CMake's find_package(). Relevant paths are extracted
  #             from <search-paths>/lib64/cmake/netCDF/netCDFConfig.cmake (if this file exists).
  #             Note that <search-paths> are automatically computed by find_package().
  #
  find_package(netCDF CONFIG QUIET)
  if (netCDF_FOUND)
    set(SEARCH_STRATEGY "find_package (Strategy 1)")

    # Forward the variables in a consistent way.
    set(NETCDF_DIR "${netCDF_INSTALL_PREFIX}")
    set(NETCDF_INCLUDE_DIRS "${netCDF_INCLUDE_DIR}") # netCDFConfig.cmake returns netCDF_INCLUDE_DIR (singular; see https://github.com/Unidata/netcdf-c/blob/3a6b1debf1557f07b606ce3653e44f0d711203be/netCDFConfig.cmake.in#L9)
    set(NETCDF_LIBRARIES "${netCDF_LIBRARIES}")
    set(NETCDF_VERSION "${netCDF_VERSION}")
  else()
    #
    # Strategy 2: Search NetCDF via pkgconfig. Relevant paths are extracted
    #             from <search-paths>/lib64/pkgconfig/netcdf.pc (if this file exists).
    #             <search-paths> are deduced from the $PKG_CONFIG_PATH environment variable.
    #
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
      pkg_check_modules(_NetCDF QUIET netcdf IMPORTED_TARGET)
      if (_NetCDF_FOUND)
        set(SEARCH_STRATEGY "pkgconfig (Strategy 2)")

        # Forward the variables in a consistent way.
        set(NETCDF_DIR "${_NetCDF_PREFIX}")
        set(NETCDF_INCLUDE_DIRS "${_NetCDF_INCLUDE_DIRS}") # FindPkgConfig returns <XXX>_INCLUDE_DIRS (plural; see https://cmake.org/cmake/help/latest/module/FindPkgConfig.html#command:pkg_check_modules)
        set(NETCDF_LIBRARIES "${_NetCDF_LINK_LIBRARIES}")
        set(NetCDF_VERSION "${_NetCDF_VERSION}")
      endif ()
    endif ()
  endif ()
else()
  #
  # Strategy 3: Guess NETCDF_INCLUDE_DIRS and NETCDF_LIBRARIES from user-provided NETCDF_DIR
  #
  set(SEARCH_STRATEGY "NETCDF_DIR (Strategy 3)")

  # Do NOT look in system paths if user specifies a NETCDF_DIR location, system
  # paths are searched first so find_path will not find user NETCDF if system
  # one is present using the default search order.
  find_path(NETCDF_INCLUDE_DIRS
    NAMES netcdf.h
    DOC "netcdf include directories"
    NO_CMAKE_SYSTEM_PATH
    PATHS "${NETCDF_DIR}/include")
  mark_as_advanced(NETCDF_INCLUDE_DIRS)

  if(NOT NETCDF_INCLUDE_DIRS)
    message(FATAL_ERROR "NetCDF header netcdf.h not found in ${NETCDF_DIR}/include; could not find NetCDF install")
  endif ()

  find_library(NETCDF_LIBRARIES
    NAMES netcdf
    DOC "netcdf library"
    NO_CMAKE_SYSTEM_PATH
    PATHS "${NETCDF_DIR}/lib"
    HINTS "${NETCDF_INCLUDE_DIRS}/../lib")
  mark_as_advanced(NETCDF_LIBRARIES)

  if(NOT NETCDF_LIBRARIES)
    message(FATAL_ERROR "NetCDF library not found in ${NETCDF_DIR}/lib; could not find NetCDF install")
  endif ()

endif ()

#
# Check if NetCDF is parallel-aware by testing if ${include_dir}/netcdf_par.h exists.
#
find_file(NETCDF_PAR_H netcdf_par.h PATHS ${NETCDF_INCLUDE_DIRS})
if (NETCDF_PAR_H)
  set(NETCDF_HAS_PARALLEL TRUE)
else()
  set(NETCDF_HAS_PARALLEL FALSE)
  message(WARNING "NetCDF parallel header file ${include_dir}/netcdf_par.h not found.")
endif()

#
# If NetCDF version wasn't found, try extracting it from netcdf_meta.h
#
if (NETCDF_INCLUDE_DIRS AND NOT NETCDF_VERSION)
  file(STRINGS "${NETCDF_INCLUDE_DIRS}/netcdf_meta.h" _netcdf_version_lines
    REGEX "#define[ \t]+NC_VERSION_(MAJOR|MINOR|PATCH|NOTE)")
  string(REGEX REPLACE ".*NC_VERSION_MAJOR *\([0-9]*\).*" "\\1" _netcdf_version_major "${_netcdf_version_lines}")
  string(REGEX REPLACE ".*NC_VERSION_MINOR *\([0-9]*\).*" "\\1" _netcdf_version_minor "${_netcdf_version_lines}")
  string(REGEX REPLACE ".*NC_VERSION_PATCH *\([0-9]*\).*" "\\1" _netcdf_version_patch "${_netcdf_version_lines}")
  string(REGEX REPLACE ".*NC_VERSION_NOTE *\"\([^\"]*\)\".*" "\\1" _netcdf_version_note "${_netcdf_version_lines}")
  set(NETCDF_VERSION "${_netcdf_version_major}.${_netcdf_version_minor}.${_netcdf_version_patch}${_netcdf_version_note}")
  unset(_netcdf_version_major)
  unset(_netcdf_version_minor)
  unset(_netcdf_version_patch)
  unset(_netcdf_version_note)
  unset(_netcdf_version_lines)
endif ()

# Run `cmake ... --log-level=DEBUG` to display these debugging information.
message(DEBUG "NETCDF search stragegy: via ${SEARCH_STRATEGY}")
message(DEBUG "NETCDF_DIR=${NETCDF_DIR}")
message(DEBUG "NETCDF_INCLUDE_DIRS=${NETCDF_INCLUDE_DIRS}")
message(DEBUG "NETCDF_LIBRARIES=${NETCDF_LIBRARIES}")
message(DEBUG "NETCDF_HAS_PARALLEL=${NETCDF_HAS_PARALLEL}")
message(DEBUG "NETCDF_VERSION=${NETCDF_VERSION}")

#
# Verify if required NETCDF variables were set
#
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NetCDF
  REQUIRED_VARS NETCDF_LIBRARIES NETCDF_INCLUDE_DIRS
  VERSION_VAR NETCDF_VERSION)

#
# Create target NetCDF::NetCDF
#
if (NetCDF_FOUND AND NOT TARGET NetCDF::NetCDF)
  add_library(NetCDF::NetCDF INTERFACE IMPORTED)
  target_include_directories(NetCDF::NetCDF INTERFACE ${NETCDF_INCLUDE_DIRS})
  target_link_libraries(NetCDF::NetCDF INTERFACE ${NETCDF_LIBRARIES})
endif ()
