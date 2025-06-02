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

set ( PystencilsTargets VectorUtilitiesPystencils NonlinearEvalPystencils )
set ( PystencilsSources pystencils_vector_utilities.py pystencils_nonlinear_eval.py )

foreach(tgt src IN ZIP_LISTS PystencilsTargets PystencilsSources)

        add_library ( ${tgt} )
        pystencilssfg_generate_target_sources ( ${tgt}
                SCRIPTS ${src}
                OUTPUT_DIRECTORY ${tgt}
        )

        # TODO: only do on-demand
        target_compile_options ( ${tgt} PRIVATE -mavx )
endforeach ()