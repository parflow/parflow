
# Common AMPS files
set(COMMON_SRC_FILES amps_abort.c amps_clear.c amps_clock.c
  amps_exchange.c amps_ffopen.c amps_find_powers.c amps_fopen.c
  amps_invoice.c amps_io.c amps_newhandle.c amps_newpackage.c
  amps_sfbcast.c amps_sfclose.c amps_sfopen.c amps_wait.c signal.c amps_print.c)
string(REGEX REPLACE "([^;]+)" "common/\\1" COMMON_SRC_FILES "${COMMON_SRC_FILES}")

# Sets AMPS_SRC_FILES for AMPS transport layer being used in build
include("${PARFLOW_AMPS_LAYER}/CMakeLists.cmake")
string(REGEX REPLACE "([^;]+)" "${PARFLOW_AMPS_LAYER}/\\1" AMPS_SRC_FILES "${AMPS_SRC_FILES}")

if( ${PARFLOW_BUILD_WITH_CPP} )
  set_source_files_properties(${COMMON_SRC_FILES} ${AMPS_SRC_FILES}
    PROPERTIES LANGUAGE CXX)
endif()

add_library(amps STATIC ${COMMON_SRC_FILES} ${AMPS_SRC_FILES})

if(${PARFLOW_HAVE_CUDA})
  target_include_directories(amps PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif(${PARFLOW_HAVE_CUDA})

if( ${PARFLOW_HAVE_RMM} )
  target_link_libraries(amps rmm)
  target_include_directories(amps PUBLIC ${RMM_INCLUDE})
endif( ${PARFLOW_HAVE_RMM} )

if( ${PARFLOW_HAVE_KOKKOS} )
  target_link_libraries(amps Kokkos::kokkos)
  target_include_directories(amps PUBLIC ${KOKKOS_INCLUDE})
endif( ${PARFLOW_HAVE_KOKKOS} )

if (${PARFLOW_HAVE_MPI})
  target_include_directories (amps PUBLIC "${MPI_C_INCLUDE_PATH}")
endif (${PARFLOW_HAVE_MPI})

if(${PARFLOW_HAVE_OAS3})
  target_link_libraries(amps ${OASIS_Fortran_LIBRARY} ${MCT_Fortran_LIBRARY} ${MPEU_Fortran_LIBRARY} ${SCRIP_Fortran_LIBRARY})
  target_include_directories(amps PUBLIC "${OASIS_Fortran_INCLUDES}")
  
  target_link_libraries(amps ${NETCDF_Fortran_LIBRARY})
  target_include_directories(amps PUBLIC "${NETCDF_Fortran_INCLUDES}")
endif(${PARFLOW_HAVE_OAS3})


install(TARGETS amps DESTINATION lib)

# Install the AMPS transport layer specific scripts
set(AMPS_SCRIPTS run)
string(REGEX REPLACE "([^;]+)" "${PARFLOW_AMPS_LAYER}/\\1" AMPS_SCRIPTS "${AMPS_SCRIPTS}")
install(FILES ${AMPS_SCRIPTS} DESTINATION bin)

add_subdirectory (test/src)

