cmake_minimum_required ( VERSION 3.24 )

project( KINSpgmr )

# extend CMake module path with source file generator module

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${KINSpgmr_SOURCE_DIR}/.cmake )
file(MAKE_DIRECTORY .cmake)
execute_process( COMMAND sfg-cli cmake make-find-module
        WORKING_DIRECTORY ${KINSpgmr_SOURCE_DIR}/.cmake )

# find package for CMake integration

find_package ( PystencilsSfg REQUIRED )

# run source file generator

add_library ( KINSpgmrPystencils )
pystencilssfg_generate_target_sources ( KINSpgmrPystencils
        SCRIPTS KINSpgmrAtimesDQ.py
        FILE_EXTENSIONS .h .cpp
        OUTPUT_DIRECTORY KINSpgmrPystencils
        )