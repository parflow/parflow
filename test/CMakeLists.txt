# -----------------------------------------------------------------------------
# Adding test files
# -----------------------------------------------------------------------------

add_subdirectory(tcl)

if(${PARFLOW_HAVE_ALQUIMIA})
    add_subdirectory(chemistry)
endif()
