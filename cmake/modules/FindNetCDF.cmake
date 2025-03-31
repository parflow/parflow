#[==[
Provides the following variables:

  * `NetCDF_FOUND`: Whether NetCDF was found or not.
  * `NetCDF_INCLUDE_DIRS`: Include directories necessary to use NetCDF.
  * `NetCDF_LIBRARIES`: Libraries necessary to use NetCDF.
  * `NetCDF_VERSION`: The version of NetCDF found.
  * `NetCDF::NetCDF`: A target to use with `target_link_libraries`.
  * `NetCDF_HAS_PARALLEL`: Whether or not NetCDF was found with parallel IO support.
#]==]

function(FindNetCDF_get_is_parallel_aware include_dir)
  # NetCDF is parallel-aware if ${include_dir}/netcdf_par.h exists.
  find_file(NETCDF_PAR_H netcdf_par.h PATHS include_dir)
  if (NETCDF_PAR_H)
    set(NetCDF_HAS_PARALLEL TRUE PARENT_SCOPE)
  else()
    set(NetCDF_HAS_PARALLEL FALSE PARENT_SCOPE)
    message(WARNING "NetCDF parallel header file ${include_dir}/netcdf_par.h not found.")
  endif()
endfunction()

if (NOT (NETCDF_DIR OR NETCDF_INCLUDE_DIR OR NetCDF_LIBRARY))
  # Try to find a CMake-built NetCDF.
  find_package(netCDF CONFIG QUIET)
  if (netCDF_FOUND)
    # Forward the variables in a consistent way.
    set(NetCDF_FOUND "${netCDF_FOUND}")
    set(NetCDF_INCLUDE_DIRS "${netCDF_INCLUDE_DIR}") # netCDFConfig.cmake returns netCDF_INCLUDE_DIR (singular; see https://github.com/Unidata/netcdf-c/blob/3a6b1debf1557f07b606ce3653e44f0d711203be/netCDFConfig.cmake.in#L9)
    set(NetCDF_LIBRARIES "${netCDF_LIBRARIES}")
    set(NetCDF_VERSION "${NetCDFVersion}")

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(NetCDF
      REQUIRED_VARS NetCDF_INCLUDE_DIRS NetCDF_LIBRARIES
      VERSION_VAR NetCDF_VERSION)

    if (NOT TARGET NetCDF::NetCDF)
      add_library(NetCDF::NetCDF INTERFACE IMPORTED)
      if (TARGET "netCDF::netcdf")
        # 4.7.3
        set_target_properties(NetCDF::NetCDF PROPERTIES INTERFACE_LINK_LIBRARIES "netCDF::netcdf")
      elseif (TARGET "netcdf")
        set_target_properties(NetCDF::NetCDF PROPERTIES INTERFACE_LINK_LIBRARIES "netcdf")
      else ()
        set_target_properties(NetCDF::NetCDF PROPERTIES INTERFACE_LINK_LIBRARIES "${netCDF_LIBRARIES}")
      endif ()
      set_target_properties(NetCDF::NetCDF PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NetCDF_INCLUDE_DIRS}")
    endif ()
    FindNetCDF_get_is_parallel_aware("${NetCDF_INCLUDE_DIRS}")
    # Skip the rest of the logic in this file.
    return ()
  endif ()

  find_package(PkgConfig QUIET)
  if (PkgConfig_FOUND)
    pkg_check_modules(_NetCDF QUIET netcdf IMPORTED_TARGET)
    if (_NetCDF_FOUND)
      # Forward the variables in a consistent way.
      set(NetCDF_FOUND "${_NetCDF_FOUND}")
      set(NetCDF_INCLUDE_DIRS "${_NetCDF_INCLUDE_DIRS}") # FindPkgConfig returns <XXX>_INCLUDE_DIRS (plural; see https://cmake.org/cmake/help/latest/module/FindPkgConfig.html#command:pkg_check_modules)
      set(NetCDF_LIBRARIES "${_NetCDF_LINK_LIBRARIES}")
      set(NetCDF_VERSION "${_NetCDF_VERSION}")

      include(FindPackageHandleStandardArgs)
      find_package_handle_standard_args(NetCDF
        REQUIRED_VARS NetCDF_LIBRARIES
        # This is not required because system-default include paths are not
        # reported by `FindPkgConfig`, so this might be empty. Assume that if we
        # have a library, the include directories are fine (if any) since
        # PkgConfig reported that the package was found.
        # NetCDF_INCLUDE_DIRS
        VERSION_VAR NetCDF_VERSION)

      if (NOT TARGET NetCDF::NetCDF)
        add_library(NetCDF::NetCDF INTERFACE IMPORTED)
        set_target_properties(NetCDF::NetCDF PROPERTIES INTERFACE_LINK_LIBRARIES "PkgConfig::_NetCDF")
        set_target_properties(NetCDF::NetCDF PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NetCDF_INCLUDE_DIRS}")
      endif ()
      FindNetCDF_get_is_parallel_aware("${NetCDF_INCLUDE_DIRS}")
      # Skip the rest of the logic in this file.
      return ()
    endif ()
  endif ()
endif ()

find_path(NETCDF_INCLUDE_DIR
  NAMES netcdf.h
  DOC "netcdf include directories"
  PATHS "${NETCDF_DIR}/include")
mark_as_advanced(NETCDF_INCLUDE_DIR)

find_library(NETCDF_LIBRARY
  NAMES netcdf
  DOC "netcdf library"
  PATHS "${NETCDF_DIR}/lib"
  HINTS "${NETCDF_INCLUDE_DIR}/../lib")
mark_as_advanced(NETCDF_LIBRARY)

if (NETCDF_INCLUDE_DIR)
  file(STRINGS "${NETCDF_INCLUDE_DIR}/netcdf_meta.h" _netcdf_version_lines
    REGEX "#define[ \t]+NC_VERSION_(MAJOR|MINOR|PATCH|NOTE)")
  string(REGEX REPLACE ".*NC_VERSION_MAJOR *\([0-9]*\).*" "\\1" _netcdf_version_major "${_netcdf_version_lines}")
  string(REGEX REPLACE ".*NC_VERSION_MINOR *\([0-9]*\).*" "\\1" _netcdf_version_minor "${_netcdf_version_lines}")
  string(REGEX REPLACE ".*NC_VERSION_PATCH *\([0-9]*\).*" "\\1" _netcdf_version_patch "${_netcdf_version_lines}")
  string(REGEX REPLACE ".*NC_VERSION_NOTE *\"\([^\"]*\)\".*" "\\1" _netcdf_version_note "${_netcdf_version_lines}")
  set(NetCDF_VERSION "${_netcdf_version_major}.${_netcdf_version_minor}.${_netcdf_version_patch}${_netcdf_version_note}")
  unset(_netcdf_version_major)
  unset(_netcdf_version_minor)
  unset(_netcdf_version_patch)
  unset(_netcdf_version_note)
  unset(_netcdf_version_lines)

  FindNetCDF_get_is_parallel_aware("${NETCDF_INCLUDE_DIR}")
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NetCDF
  REQUIRED_VARS NETCDF_LIBRARY NETCDF_INCLUDE_DIR
  VERSION_VAR NetCDF_VERSION)

if (NetCDF_FOUND)
  set(NETCDF_INCLUDE_DIRS "${NETCDF_INCLUDE_DIR}")
  set(NetCDF_LIBRARIES "${NETCDF_LIBRARY}")

  if (NOT TARGET NetCDF::NetCDF)
    add_library(NetCDF::NetCDF UNKNOWN IMPORTED)
    set_target_properties(NetCDF::NetCDF PROPERTIES
      IMPORTED_LOCATION "${NETCDF_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NETCDF_INCLUDE_DIR}")
  endif ()
endif ()
