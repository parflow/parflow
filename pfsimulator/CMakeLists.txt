

add_subdirectory (amps)
add_subdirectory (kinsol)
add_subdirectory (parflow_lib)
add_subdirectory (parflow_exe)

if( ${PARFLOW_HAVE_CLM} )
  add_subdirectory (clm)
endif( ${PARFLOW_HAVE_CLM} )

