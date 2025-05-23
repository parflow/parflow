cmake_minimum_required ( VERSION 3.24 )

# determine config file depending on build flags

if ( DEFINED PYSTENCILS_CONFIG_MODULE )
        set ( PystencilsSfg_CONFIG_MODULE ${PYSTENCILS_CONFIG_MODULE} )
else ()
        if ( ${PARFLOW_HAVE_CUDA} )
                set ( PystencilsSfg_CONFIG_MODULE cuda_config.py )
        else ()
                set ( PystencilsSfg_CONFIG_MODULE cpu_config.py )
        endif ()
endif ()

# run source file generator

add_library ( VectorUtilitiesPystencils )
pystencilssfg_generate_target_sources ( VectorUtilitiesPystencils
        SCRIPTS pystencils_vector_utilities.py
        OUTPUT_DIRECTORY VectorUtilitiesPystencils
)

# TODO: only do on-demand
target_compile_options ( VectorUtilitiesPystencils PRIVATE -mavx )