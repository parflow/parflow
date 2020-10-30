
# Common AMPS files
set(COMMON_SRC_FILES amps_abort.c amps_clear.c amps_clock.c
  amps_exchange.c amps_ffopen.c amps_find_powers.c amps_fopen.c
  amps_invoice.c amps_io.c amps_newhandle.c amps_newpackage.c
  amps_sfbcast.c amps_sfclose.c amps_sfopen.c amps_wait.c signal.c)
string(REGEX REPLACE "([^;]+)" "common/\\1" COMMON_SRC_FILES "${COMMON_SRC_FILES}")

# Sets AMPS_SRC_FILES for AMPS transport layer being used in build
include("${PARFLOW_AMPS_LAYER}/CMakeLists.cmake")
string(REGEX REPLACE "([^;]+)" "${PARFLOW_AMPS_LAYER}/\\1" AMPS_SRC_FILES "${AMPS_SRC_FILES}")

add_library(amps STATIC ${COMMON_SRC_FILES} ${AMPS_SRC_FILES})

# The order for the below libs is important
target_link_libraries(amps /p/project/cesmtst/hokkanen1/TSMP/oasis3-mct_JUWELS_3.1.0MCT_clm-pfl/JUWELS/lib/libpsmile.MPI1.a)
target_link_libraries(amps /p/project/cesmtst/hokkanen1/TSMP/oasis3-mct_JUWELS_3.1.0MCT_clm-pfl/JUWELS/lib/libmct.a)
target_link_libraries(amps /p/project/cesmtst/hokkanen1/TSMP/oasis3-mct_JUWELS_3.1.0MCT_clm-pfl/JUWELS/lib/libmpeu.a)
target_link_libraries(amps /p/project/cesmtst/hokkanen1/TSMP/oasis3-mct_JUWELS_3.1.0MCT_clm-pfl/JUWELS/lib/libscrip.a)
target_include_directories(amps PUBLIC "/p/project/cesmtst/hokkanen1/TSMP/oasis3-mct_JUWELS_3.1.0MCT_clm-pfl/JUWELS/build/lib/psmile.MPI1")

if(${PARFLOW_HAVE_CUDA})
  target_include_directories(amps PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  if( ${PARFLOW_HAVE_RMM} )
    target_link_libraries(amps rmm)
    target_include_directories(amps PUBLIC ${RMM_INCLUDE})
  endif( ${PARFLOW_HAVE_RMM} )
endif(${PARFLOW_HAVE_CUDA})

if (${PARFLOW_HAVE_MPI})
  target_include_directories (amps PUBLIC "${MPI_C_INCLUDE_PATH}")
endif (${PARFLOW_HAVE_MPI})

install(TARGETS amps DESTINATION lib)

# Install the AMPS transport layer specific scripts
set(AMPS_SCRIPTS run)
string(REGEX REPLACE "([^;]+)" "${PARFLOW_AMPS_LAYER}/\\1" AMPS_SCRIPTS "${AMPS_SCRIPTS}")
install(FILES ${AMPS_SCRIPTS} DESTINATION bin)

add_subdirectory (test/src)

