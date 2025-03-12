
add_subdirectory (cjson)
add_subdirectory (etrace)
if(${PARFLOW_HAVE_PYSTENCILS})
    add_subdirectory (pystencils)
endif()

