cmake_minimum_required (VERSION 3.5.2)

### Build Flags
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "${CMAKE_Fortran_FLAGS_RELWITHDEBINFO} -fbacktrace")
  set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -O0 -fbacktrace -fcheck=all" )
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "${CMAKE_Fortran_FLAGS_RELWITHDEBINFO} -traceback")
  set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -O0 -traceback -check all")
endif()

### Find and Include ESMF
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")
find_package(ESMF MODULE REQUIRED)

### Include ESMF Compiler Options
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${ESMF_F90COMPILEOPTS}")

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include)

### Library Files
list(APPEND pf_nuopc_files
  src/pf_nuopc_cap.F90
  src/pf_nuopc_fields.F90
  src/pf_nuopc_grid.F90
  src/pf_nuopc_flags.F90
)

### New Library: parflow_nuopc
add_library(parflow_nuopc STATIC ${pf_nuopc_files})
add_dependencies(parflow_nuopc pfsimulator)
include_directories(parflow_nuopc include)
target_link_libraries(parflow_nuopc PUBLIC ESMF)
target_link_libraries(parflow_nuopc PUBLIC pfsimulator)

# configure fire behavior esmx executable
if (PARFLOW_ENABLE_ESMX)
  if(ESMF_VERSION VERSION_LESS "8.5.0")
    message(FATAL_ERROR "ESMX requires ESMF version 8.5.0 or greater")
  endif (ESMF_VERSION VERSION_LESS "8.5.0")
  include(ExternalProject)
  ExternalProject_Add(esmx
    BINARY_DIR ${CMAKE_BINARY_DIR}/esmx
    SOURCE_DIR ${ESMF_ESMXDIR}
    CMAKE_ARGS -DESMX_BUILD_FILE=${CMAKE_CURRENT_SOURCE_DIR}/esmxBuild.yaml
               -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    BUILD_ALWAYS TRUE
    DEPENDS parflow_nuopc
    EXCLUDE_FROM_ALL TRUE
  )
endif (PARFLOW_ENABLE_ESMX)

### Install Library
install(TARGETS parflow_nuopc
  EXPORT ParFlow
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include
)
install(FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/parflow_nuopc.mod
  DESTINATION include
)

