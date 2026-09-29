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

# time codegen and compilation of each kernel

set ( PystencilsSfg_TIMING_LOG ${CMAKE_BINARY_DIR}/pystencils_kernel_times.log )
set ( PystencilsSfg_TIMING_CMD ${PystencilsSfg_PYTHON_INTERPRETER} ${CMAKE_CURRENT_SOURCE_DIR}/pystencils_timing.py ${PystencilsSfg_TIMING_LOG} )
list ( JOIN PystencilsSfg_TIMING_CMD "\" \"" PystencilsSfg_TIMING_LAUNCHER )

set_property ( DIRECTORY PROPERTY RULE_LAUNCH_CUSTOM "\"${PystencilsSfg_TIMING_LAUNCHER}\" codegen=<OUTPUT>" )
set_property ( DIRECTORY PROPERTY RULE_LAUNCH_COMPILE "\"${PystencilsSfg_TIMING_LAUNCHER}\" compile=<SOURCE>" )

add_library ( PystencilsGeneratedKernels )
pystencilssfg_generate_target_sources ( PystencilsGeneratedKernels
        SCRIPTS ${PystencilsSfg_SCRIPTS}
        OUTPUT_DIRECTORY generated_kernels
)

list ( TRANSFORM PystencilsSfg_SCRIPTS REPLACE "\\.py$" "" OUTPUT_VARIABLE PystencilsSfg_KERNELS )
add_custom_command ( TARGET PystencilsGeneratedKernels POST_BUILD
        COMMAND ${PystencilsSfg_TIMING_CMD} summary ${PystencilsSfg_KERNELS} )

target_link_libraries ( PystencilsGeneratedKernels PRIVATE pfsimulator amps )

target_compile_definitions ( PystencilsGeneratedKernels PRIVATE PF_COMP_UNIT_TYPE=1 )

target_compile_options( PystencilsGeneratedKernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA,CXX>:-include;${CMAKE_CURRENT_SOURCE_DIR}/pf_amps_shim.h> )

if ( PystencilsSfg_OPTIMIZE_FOR_LOCALHOST )
        target_compile_options ( PystencilsGeneratedKernels PRIVATE "-march=native" )
endif ()