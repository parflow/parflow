cmake_minimum_required ( VERSION 3.24 )

# find package for CMake integration

find_package ( PystencilsSfg REQUIRED )

# determine config file depending on build flags

if ( ${PARFLOW_HAVE_CUDA} )
        set ( PystencilsSfg_CONFIG_MODULE cuda_config.py )
else ()
        set ( PystencilsSfg_CONFIG_MODULE cpu_config.py )
endif ()

# run source file generator

add_library ( VectorUtilitiesPystencils )
pystencilssfg_generate_target_sources ( VectorUtilitiesPystencils
        SCRIPTS pystencils_vector_utilities.py
        OUTPUT_DIRECTORY VectorUtilitiesPystencils
)