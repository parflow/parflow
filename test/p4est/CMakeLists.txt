# Test ParFlow+p4est integration if p4est was enabled

if(${PARFLOW_HAVE_P4EST})
    add_subdirectory (step0)
    add_subdirectory (step1)
endif()
