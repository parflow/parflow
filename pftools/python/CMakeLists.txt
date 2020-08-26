# -----------------------------------------------------------------------------
# Make sure we update the generated.py file
# -----------------------------------------------------------------------------

# add_custom_command(
#   OUTPUT  "${_vtk_add_python_package_file_output}"
#   DEPENDS "${_vtk_add_python_package_file}"
#   COMMAND "${CMAKE_COMMAND}" -E copy_if_different
#           "${_vtk_add_python_package_file}"
#           "${_vtk_add_python_package_file_output}"
#   COMMENT "Copying ${_vtk_add_python_package_name} to the binary directory"
# )

# install(
#   FILES "${_vtk_add_python_package_name}"
#   DESTINATION "${_vtk_add_python_package_MODULE_DESTINATION}/${_vtk_add_python_package_path}"
#   COMPONENT   "${_vtk_add_python_package_COMPONENT}"
# )
# install(
#   TARGETS             "${target}"
#   ${_vtk_install_export}
#   ${ARGN}
#   ARCHIVE
#     DESTINATION "${_vtk_build_ARCHIVE_DESTINATION}"
#     COMPONENT   "${_vtk_install_headers_component}"
#   LIBRARY
#     DESTINATION "${_vtk_build_LIBRARY_DESTINATION}"
#     COMPONENT   "${_vtk_install_targets_component}"
#     NAMELINK_COMPONENT "${_vtk_build_HEADERS_COMPONENT}"
#   RUNTIME
#     DESTINATION "${_vtk_build_RUNTIME_DESTINATION}"
#     COMPONENT   "${_vtk_install_targets_component}"
# )

add_custom_target(
  pf-python ALL
  DEPENDS
    GeneratePythonKeys
)

# -----------------------------------------------------------------------------
# Generate Pypy package
# -----------------------------------------------------------------------------

# TODO
