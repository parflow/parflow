cmake_minimum_required ( VERSION 3.24 )

# determine config file depending on build flags

if ( NOT DEFINED PystencilsSfg_CONFIG_MODULE )
        if ( ${PARFLOW_HAVE_CUDA} )
                set ( PystencilsSfg_CONFIG_MODULE sfg_modules/gpu_config_cuda.py)
        else ()
                set ( PystencilsSfg_CONFIG_MODULE sfg_modules/cpu_config.py)
        endif ()
endif ()

# run source file generator

set ( PystencilsSfg_SCRIPTS pystencils_vector_utilities.py pystencils_nonlinear_eval.py pystencils_rbgs.py )

if ( ${PARFLOW_HAVE_PYSTENCILS_FUSED_KERNELS} )
        list ( APPEND PystencilsSfg_SCRIPTS pystencils_kinsol_atimesdq.py )
endif ()

add_library ( PystencilsGeneratedKernels )
pystencilssfg_generate_target_sources ( PystencilsGeneratedKernels
        SCRIPTS ${PystencilsSfg_SCRIPTS}
        OUTPUT_DIRECTORY generated_kernels
)

target_link_libraries ( PystencilsGeneratedKernels PRIVATE pfsimulator amps )

target_compile_definitions ( PystencilsGeneratedKernels PRIVATE PF_COMP_UNIT_TYPE=1 )

target_compile_options( PystencilsGeneratedKernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA,CXX>:-include;${CMAKE_CURRENT_SOURCE_DIR}/pf_amps_shim.h> )

if ( PystencilsSfg_OPTIMIZE_FOR_LOCALHOST )
        target_compile_options ( PystencilsGeneratedKernels PRIVATE "-march=native" )
endif ()