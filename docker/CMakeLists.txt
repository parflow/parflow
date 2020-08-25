
option(PYTHON_DOCKER_BUILD "Building ParFlow Docker with Python" "FALSE")
if (PYTHON_DOCKER_BUILD)
  add_custom_command(
    OUTPUT
      "Docker build"
    COMMAND
      ./build.sh
    VERBATIM
  )
endif()