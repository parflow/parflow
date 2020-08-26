# -----------------------------------------------------------------------------
# Make sure we update the generated.py file
# -----------------------------------------------------------------------------

# copy the source tree into the build tree (pftools/python/parflow)
# into CMAKE_BUILD_DIR (or something like that)

add_custom_command(
    OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/parflow"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/parflow"
            "${CMAKE_CURRENT_BINARY_DIR}/parflow"
   COMMENT "Copying Python pftools to the build directory"
)

# moving files to source tree when make install

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
    "${CMAKE_CURRENT_BINARY_DIR}/parflow"
    GeneratePythonKeys
)

# -----------------------------------------------------------------------------
# Generate Pypy package
# -----------------------------------------------------------------------------

# TODO
