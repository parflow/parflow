#[[
Find-Module for pystencils-sfg.

# Setting the Python interpreter

If the cache entry PystencilsSfg_PYTHON_INTERPRETER is set, e.g. via the commandline
(`-DPystencilsSfg_PYTHON_INTERPRETER=<...>`), its value be taken as the Python interpreter
used to find and run pystencils-sfg.

If the cache entry is unset, but the hint PystencilsSfg_PYTHON_PATH is set, its value will
be used as the Python interpreter.

If none of these is set, a Python interpreter will be selected using the `FindPython` module.

#]]

if(NOT DEFINED CACHE{PystencilsSfg_PYTHON_INTERPRETER})
    #   The Python interpreter cache variable is not set externally, so...
    if(DEFINED PystencilsSfg_PYTHON_PATH)
        #   ... either initialize it from the hint variable ...
        set( _sfg_cache_python_init ${PystencilsSfg_PYTHON_PATH} )
    else()
        #   ... or, if that is also unset, use the system Python
        find_package( Python COMPONENTS Interpreter REQUIRED )
        set( _sfg_cache_python_init ${Python_EXECUTABLE} )
    endif()
endif()

set(PystencilsSfg_PYTHON_INTERPRETER ${_sfg_cache_python_init} CACHE PATH "Path to the Python executable used to run pystencils-sfg")

#   Try to find pystencils-sfg in the python environment

execute_process(COMMAND ${PystencilsSfg_PYTHON_INTERPRETER} -m pystencilssfg version --no-newline
        RESULT_VARIABLE _PystencilsSfgFindResult OUTPUT_VARIABLE PystencilsSfg_VERSION )

if(${_PystencilsSfgFindResult} EQUAL 0)
    set( PystencilsSfg_FOUND ON )
else()
    set( PystencilsSfg_FOUND OFF )
endif()

if(DEFINED PystencilsSfg_FIND_REQUIRED)
    if(NOT ${PystencilsSfg_FOUND})
        message( FATAL_ERROR "Could not find pystencils-sfg in current Python environment." )
    endif()
endif()

if(${PystencilsSfg_FOUND})
    message( STATUS "Found pystencils Source File Generator (Version ${PystencilsSfg_VERSION})")
    message( STATUS "Using Python interpreter ${PystencilsSfg_PYTHON_INTERPRETER} for SFG generator scripts.")

    execute_process(COMMAND ${PystencilsSfg_PYTHON_INTERPRETER} -m pystencilssfg cmake modulepath --no-newline
            OUTPUT_VARIABLE _PystencilsSfg_CMAKE_MODULE_PATH)

    set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${_PystencilsSfg_CMAKE_MODULE_PATH})
    include( PystencilsSfg )
endif()

