# -----------------------------------------------------------------------------
# Build rst files from key definitions
# -----------------------------------------------------------------------------

add_custom_command(
  OUTPUT
    "${CMAKE_CURRENT_SOURCE_DIR}/python/keys.rst"
  COMMAND
    ${PARFLOW_PYTHON} read_the_doc_rst.py
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pf-keys/generators/read_the_doc_rst.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pf-keys/definitions/wells.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pf-keys/definitions/timing.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pf-keys/definitions/bconditions.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pf-keys/definitions/core.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pf-keys/definitions/geom.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pf-keys/definitions/netcdf.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pf-keys/definitions/phase.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pf-keys/definitions/run.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pf-keys/definitions/solver.yaml"
  WORKING_DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pf-keys/generators/"
  COMMENT
    "Generate ParFlow keys documentation"
  VERBATIM
)

# -----------------------------------------------------------------------------

add_custom_target(
  GenerateKeysRST
  DEPENDS
    ${PARFLOW_PYTHON_DEPENDS}
    "${CMAKE_CURRENT_SOURCE_DIR}/python/keys.rst"
)

# -----------------------------------------------------------------------------
# Generate documentation locally
# -----------------------------------------------------------------------------

include(FindPythonModules)
find_python_module("sphinx" sphinx_found)
if (NOT sphinx_found)
  message(STATUS
    "Please install Sphinx to generate documentation for ParFlow keys.")
  return()
endif()

find_python_module("sphinx-rtd-theme" sphinx_theme_found)
if (NOT sphinx_theme_found)
  message(STATUS
    "Please install sphinx-rtd-theme to generate documentation for ParFlow keys.")
  return()
endif()

find_python_module("sphinxcontrib-bibtex" sphinx_bib_found)
if (NOT sphinx_bib_found)
  message(STATUS
    "Please install sphinxcontrib-bibtex to generate documentation for ParFlow keys.")
  return()
endif()

# -----------------------------------------------------------------------------

add_custom_target(
  ParFlowKeyDoc
  DEPENDS
    GenerateKeysRST
  COMMAND
    ${PARFLOW_PYTHON}
    -m sphinx
    -b html "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/build-site"
  SOURCES
    conf.py
    index.rst
    python/keys.rst
  # ensures that the output is continuously generated
  # when using ninja; avoids builder from timing out
  USES_TERMINAL
)
