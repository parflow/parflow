if( ${PARFLOW_HAVE_CUDA} )
    set (SRC_FILES hello_test.cu cublas_test.cu)
    add_library(pfcuda ${SRC_FILES})

    set_target_properties(pfcuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    install (TARGETS pfcuda DESTINATION lib)
endif( ${PARFLOW_HAVE_CUDA} )