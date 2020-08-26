
# -----------------------------------------------------------------------------
# Make sure our Python has the module that we need
# -----------------------------------------------------------------------------

include(FindPythonModules)
find_python_module("yaml" yaml_found)
if (NOT yaml_found AND NOT PARFLOW_PYTHON_VIRTUAL_ENV)
  message(STATUS
    "Please install yaml to generate ParFlow keys for Python module or use the VirtualEnv option.")
  return()
endif()

# -----------------------------------------------------------------------------
# Build generated python file for key definitions
# -----------------------------------------------------------------------------

set(_generated_output
  "${CMAKE_CURRENT_BINARY_DIR}/../pftools/python/parflow/tools/database/generated.py"
)

add_custom_command(
  OUTPUT
    "${_generated_output}"
  COMMAND
    "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/../pftools/python/parflow/tools/database/"
  COMMAND
    ${PARFLOW_PYTHON} pf-python.py
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/generators/pf-python.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/definitions/wells.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/definitions/timing.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/definitions/bconditions.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/definitions/core.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/definitions/geom.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/definitions/netcdf.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/definitions/phase.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/definitions/run.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/definitions/solver.yaml"
  WORKING_DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/generators/"
  COMMENT
    "Generate ParFlow keys for Python"
  VERBATIM
)

# -----------------------------------------------------------------------------

add_custom_target(
  GeneratePythonKeys
  DEPENDS
    ${PARFLOW_PYTHON_DEPENDS}
    "${_generated_output}"
)
