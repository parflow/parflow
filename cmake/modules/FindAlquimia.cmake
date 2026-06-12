#.rst:
# FindAlquimia
# ------------
#
# Find the Alquimia biogeochemistry interface library.
#
# Interface version assumption (react_trans port, v1): this module and the
# chemistry coupling target the Alquimia v1.0 API as published with
# Molins et al. 2025 (GMD 18:3241). In particular the per-cell solve is
# invoked as
#
#   chem.ReactionStepOperatorSplit(engine, delta_t, properties, state,
#                                  aux_data, status)
#
# Phase 4 must re-verify that argument order against the headers of the
# Alquimia actually found here before enabling chemistry runs.
#
# Alquimia is linked against one or more geochemistry engines (CrunchFlow
# and/or PFLOTRAN) and PETSc. Those are Alquimia's dependencies, not
# ParFlow's: a shared-library Alquimia carries them itself; a static
# Alquimia needs them on the link line, so this module looks for them
# relative to the Alquimia installation (and PETSc via its pkg-config)
# rather than as independently configured packages.
#
# Hints:
#
# ::
#
#   PARFLOW_ALQUIMIA_ROOT - root of an *installed* Alquimia (the prefix
#                           given to Alquimia's CMAKE_INSTALL_PREFIX).
#                           ALQUIMIA_ROOT (variable or environment) is
#                           accepted as a fallback hint.
#
# This module sets the following variables:
#
# ::
#
#   ALQUIMIA_FOUND        - true if Alquimia was found
#   ALQUIMIA_INCLUDE_DIRS - Alquimia include directories
#   ALQUIMIA_LIBRARIES    - Alquimia library plus its engine/PETSc
#                           dependencies when they are required and found

include(FindPackageHandleStandardArgs)

if(NOT PARFLOW_ALQUIMIA_ROOT)
    if(ALQUIMIA_ROOT)
        set(PARFLOW_ALQUIMIA_ROOT ${ALQUIMIA_ROOT})
    elseif(DEFINED ENV{ALQUIMIA_ROOT})
        set(PARFLOW_ALQUIMIA_ROOT $ENV{ALQUIMIA_ROOT})
    endif()
endif()

#
# Prefer an Alquimia CMake package configuration when the installation
# provides one; fall back to a header/library search.
#
find_package(Alquimia CONFIG QUIET HINTS ${PARFLOW_ALQUIMIA_ROOT})

if(TARGET alquimia)
    set(ALQUIMIA_LIBRARIES alquimia)
    get_target_property(
        ALQUIMIA_INCLUDE_DIRS
        alquimia
        INTERFACE_INCLUDE_DIRECTORIES
    )
    find_package_handle_standard_args(
        Alquimia
        REQUIRED_VARS ALQUIMIA_LIBRARIES ALQUIMIA_INCLUDE_DIRS
    )
    return()
endif()

if(DEFINED PARFLOW_ALQUIMIA_ROOT)
    find_path(
        ALQUIMIA_INCLUDE_DIR
        NAMES alquimia/alquimia.h
        PATHS ${PARFLOW_ALQUIMIA_ROOT}/include
        NO_DEFAULT_PATH
    )
    find_library(
        ALQUIMIA_LIBRARY
        NAMES alquimia
        PATHS ${PARFLOW_ALQUIMIA_ROOT}/lib ${PARFLOW_ALQUIMIA_ROOT}/lib64
        NO_DEFAULT_PATH
    )
else()
    find_path(ALQUIMIA_INCLUDE_DIR NAMES alquimia/alquimia.h)
    find_library(ALQUIMIA_LIBRARY NAMES alquimia)
endif()

set(ALQUIMIA_INCLUDE_DIRS ${ALQUIMIA_INCLUDE_DIR})
set(ALQUIMIA_LIBRARIES ${ALQUIMIA_LIBRARY})

#
# Engine and PETSc dependencies of a static Alquimia: search next to the
# Alquimia library itself (engines are co-installed by alquimia-dev /
# spack builds), append whichever are present.
#
if(ALQUIMIA_LIBRARY)
    get_filename_component(ALQUIMIA_LIBRARY_DIR ${ALQUIMIA_LIBRARY} DIRECTORY)

    foreach(_engine crunchchem pflotranchem)
        find_library(
            ALQUIMIA_${_engine}_LIBRARY
            NAMES ${_engine}
            PATHS ${ALQUIMIA_LIBRARY_DIR}
            NO_DEFAULT_PATH
        )
        if(ALQUIMIA_${_engine}_LIBRARY)
            message(
                STATUS
                "Alquimia engine found: ${ALQUIMIA_${_engine}_LIBRARY}"
            )
            list(APPEND ALQUIMIA_LIBRARIES ${ALQUIMIA_${_engine}_LIBRARY})
        endif()
    endforeach()

    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(ALQUIMIA_PETSC QUIET PETSc)
        if(ALQUIMIA_PETSC_FOUND)
            list(APPEND ALQUIMIA_LIBRARIES ${ALQUIMIA_PETSC_LINK_LIBRARIES})
        endif()
    endif()
endif()

find_package_handle_standard_args(
    Alquimia
    REQUIRED_VARS ALQUIMIA_LIBRARY ALQUIMIA_INCLUDE_DIR
    FAIL_MESSAGE
        "Alquimia not found, set PARFLOW_ALQUIMIA_ROOT to the install prefix of an Alquimia built with your geochemistry engine (CrunchFlow and/or PFLOTRAN)"
)
