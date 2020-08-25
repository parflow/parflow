set(PARFLOW_DOCKER_BASE_IMAGE "ubuntu:20.04" CACHE STRING "Docker base image to use")
set(PARFLOW_DOCKER_VERSION "v3.6.0" CACHE STRING "Version of ParFlow to install in docker (v3.6.0, master)")
set(PARFLOW_DOCKER_HYPRE "v2.19.0" CACHE STRING "Version of Hypre to install in docker (v2.19.0, master)")

add_custom_target(buildDevelopmentDocker
  docker build --rm -t parflow-development
    --build-arg "BASE_IMAGE=${PARFLOW_DOCKER_BASE_IMAGE}"
    --build-arg "PARFLOW_VERSION=${PARFLOW_DOCKER_VERSION}"
    --build-arg "HYPRE_VERSION=${PARFLOW_DOCKER_HYPRE}"
    -f "${CMAKE_CURRENT_SOURCE_DIR}/dev/Dockerfile"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  VERBATIM
)

add_custom_target(buildRuntimeDocker
  docker build --rm -t parflow-runtime
    --build-arg "BASE_IMAGE=${PARFLOW_DOCKER_BASE_IMAGE}"
    -f "${CMAKE_CURRENT_SOURCE_DIR}/runtime/Dockerfile"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS
    buildDevelopmentDocker
  VERBATIM
)

add_custom_target(testDevelopmentDocker
  docker exec --rm -it parflow-development $CTEST --build-and-test /home/ubuntu/parflow/parflow /home/ubuntu/parflow/build
  DEPENDS
    buildDevelopmentDocker
  VERBATIM
)
