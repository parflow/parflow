
# -----------------------------------------------------------------------------
# Build generated python file for key definitions
# -----------------------------------------------------------------------------

add_custom_command(
  OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/../pftools/python/parflow/tools/database/generated.py"
  COMMAND
    "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/../pftools/python/parflow/tools/database/"
  COMMAND
    ${PARFLOW_PYTHON} pf-python.py
    "${CMAKE_CURRENT_BINARY_DIR}/../pftools/python/parflow/tools/database/generated.py"
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
    "${CMAKE_CURRENT_SOURCE_DIR}/definitions/metadata.yaml"
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
    "${CMAKE_CURRENT_BINARY_DIR}/../pftools/python/parflow/tools/database/generated.py"
)
