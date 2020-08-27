# -----------------------------------------------------------------------------
# Make sure we update the generated.py file
# -----------------------------------------------------------------------------

# copy the source tree into the build tree
add_custom_command(
  OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/parflow/tools/core.py"
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/builders.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/core.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/terminal.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/database/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/database/core.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/database/domains.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/database/handlers.py"
  COMMENT
    "Copying Python pftools to the build directory"
)

# Install Python package
install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/parflow"
  DESTINATION "python"
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc"       EXCLUDE
)
install(
  FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
  DESTINATION
    "python"
)

add_custom_target(
  pf-python ALL
  DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/parflow/tools/core.py"
    GeneratePythonKeys
)

# -----------------------------------------------------------------------------
# Generate Pypy package
# -----------------------------------------------------------------------------

# TODO
