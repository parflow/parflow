cmake_minimum_required ( VERSION 3.24 )

# determine config file depending on build flags

if ( NOT DEFINED PystencilsSfg_CONFIG_MODULE )
        if ( ${PARFLOW_HAVE_CUDA} )
                set ( PystencilsSfg_CONFIG_MODULE cuda_config.py )
        else ()
                set ( PystencilsSfg_CONFIG_MODULE cpu_config.py )
        endif ()
endif ()

# run source file generator

add_library ( PystencilsGeneratedKernels )
pystencilssfg_generate_target_sources ( PystencilsGeneratedKernels
        SCRIPTS pystencils_vector_utilities.py pystencils_nonlinear_eval.py
        OUTPUT_DIRECTORY generated_kernels
)

# TODO: only do on-demand
target_compile_options ( PystencilsGeneratedKernels PRIVATE -mavx )