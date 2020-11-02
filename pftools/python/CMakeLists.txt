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
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/cli/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/cli/pfdist_sort.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/cli/tcl2py.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/builders.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/core.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/export.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/fs.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/helper.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/io.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/settings.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/terminal.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/database/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/database/core.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/database/domains.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/database/handlers.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/ref/table_keys.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/ref/vegp_keys.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/ref/subsurface_conus_1.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/ref/subsurface_freeze_cherry.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/ref/subsurface_washita.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/ref/vegp_igbp.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/ref/vegp_header.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/parflow/tools/ref/default_vegp_values.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/MANIFEST.in"
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
    "${CMAKE_CURRENT_SOURCE_DIR}/requirements_all.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/requirements_pfsol.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/requirements_pfb.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
  DESTINATION
    "python"
)
install(
  PROGRAMS
    "${CMAKE_CURRENT_SOURCE_DIR}/py-setup"
    "${CMAKE_CURRENT_SOURCE_DIR}/pfpython"
  DESTINATION
    "bin"
)

install(SCRIPT "${CMAKE_SOURCE_DIR}/pftools/python/PostInstall.cmake")

add_custom_target(
  pf-python ALL
  DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/parflow/tools/core.py"
    GeneratePythonKeys
)

# -----------------------------------------------------------------------------
# Generate PyPI package
# -----------------------------------------------------------------------------

add_custom_target(PythonCreatePackage
  WORKING_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
      "${CMAKE_COMMAND}" -E remove_directory "${CMAKE_CURRENT_BINARY_DIR}/dist/"
  COMMAND
      ${PARFLOW_PYTHON} setup.py sdist bdist_wheel
  COMMAND
      twine check "${CMAKE_CURRENT_BINARY_DIR}/dist/*"
  COMMENT
      "Updating the pftools Python package"
  VERBATIM
  DEPENDS
    pf-python
)

add_custom_target(PythonPublishPackage
  WORKING_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
      twine upload dist/*
  COMMENT
      "Publishing the pftools Python package"
  VERBATIM
  DEPENDS
    PythonCreatePackage
)
