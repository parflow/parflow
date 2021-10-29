set(AMPS_SRC_FILES
  amps_allreduce.c
  amps_bcast.c
  amps_clear.c
  amps_createinvoice.c
  amps_exchange.c
  amps_finalize.c
  amps_init.c
  amps_invoice.c
  amps_irecv.c
  amps_newpackage.c
  amps_pack.c
  amps_print.c
  amps_recv.c
  amps_send.c
  amps_sizeofinvoice.c
  amps_test.c
  amps_unpack.c
  amps_vector.c
  )

if (${OAS3_API_VERSION} STREQUAL "2.0")
  message(STATUS "Using legacy OASIS3-MCT API")
  list(APPEND AMPS_SRC_FILES oas_pfl_define.F90
                             oas_pfl_finalize.F90
                             oas_pfl_init.F90
                             oas_pfl_rcv.F90
                             oas_pfl_snd.F90
                             oas_pfl_vardef.F90
                             receive_fld2_clm.F90
                             send_fld2_clm.F90)
elseif (${OAS3_API_VERSION} STREQUAL "4.0")
  message(STATUS "Using OASIS3-MCT 4.0 APIs")
  list(APPEND AMPS_SRC_FILES oas_pfl_mod.F90)
else()
  message(FATAL_ERROR "OAS3_API_VERSION=${OAS3_API_VERSION} is unsupported.")
endif(${OAS3_API_VERSION})

if((${PARFLOW_HAVE_CUDA}) AND (NOT (${PARFLOW_HAVE_KOKKOS})))
  list(APPEND AMPS_SRC_FILES amps_gpupacking.cu)
endif((${PARFLOW_HAVE_CUDA}) AND (NOT (${PARFLOW_HAVE_KOKKOS})))

if(${PARFLOW_HAVE_KOKKOS})
  list(APPEND AMPS_SRC_FILES amps_gpupacking.cpp)
endif(${PARFLOW_HAVE_KOKKOS})
