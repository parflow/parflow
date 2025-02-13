
# Macros for helping with version numbers

#
# Split a version number "vX.Y.Z-tweak-gHASH" into separate variables.
# Variable ${packageName}_VERSION must exist and contain the version number.
#
macro (version_create_variables packageName)
    if (DEFINED ${packageName}_VERSION)
        string (REGEX MATCHALL "[0-9a-zA-Z]+" _versionComponents "${${packageName}_VERSION}")
        list (LENGTH _versionComponents _len)
        if (${_len} GREATER 0)
	  list(GET _versionComponents 0 _major)
	  string (REGEX MATCHALL "[0-9]+" _major ${_major})
          list(GET _major 0 ${packageName}_VERSION_MAJOR)
        endif()
        if (${_len} GREATER 1)
            list(GET _versionComponents 1 ${packageName}_VERSION_MINOR)
        endif()
        if (${_len} GREATER 2)
            list(GET _versionComponents 2 ${packageName}_VERSION_PATCH)
        endif()
        if (${_len} GREATER 3)
            list(GET _versionComponents 3 ${packageName}_VERSION_TWEAK)
        endif()
        if (${_len} GREATER 3)
          list(GET _versionComponents -1 _minor)
	  string(SUBSTRING ${_minor} 1 -1 _temp)
	  set(${packageName}_VERSION_HASH ${_temp})
        endif()
        set (${packageName}_VERSION_COUNT ${_len})
    else()
        set (${packageName}_VERSION_COUNT 0)
        set (${packageName}_VERSION "")
    endif()
endmacro()
