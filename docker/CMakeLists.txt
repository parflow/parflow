set(PARFLOW_DOCKER_BASE_IMAGE "ubuntu:20.04" CACHE STRING "Docker base image to use")
set(PARFLOW_DOCKER_VERSION "py-pftools-integration" CACHE STRING "Version of ParFlow to install in docker (v3.6.0, master)")
set(PARFLOW_DOCKER_HYPRE "v2.19.0" CACHE STRING "Version of Hypre to install in docker (v2.19.0, master)")

add_custom_target(DockerBuildDevelopment
  docker build --rm -t parflow-development
    --build-arg "BASE_IMAGE=${PARFLOW_DOCKER_BASE_IMAGE}"
    --build-arg "PARFLOW_VERSION=${PARFLOW_DOCKER_VERSION}"
    --build-arg "PARFLOW_GIT_URL=https://github.com/grapp1/parflow.git"        # Need to remove once part of ParFlow repo
    --build-arg "HYPRE_VERSION=${PARFLOW_DOCKER_HYPRE}"
    -f "${CMAKE_CURRENT_SOURCE_DIR}/dev/Dockerfile"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT
    "Build docker image 'parflow-development'"
  VERBATIM
)

add_custom_target(DockerBuildRuntime
  docker build --rm -t parflow-runtime
    --build-arg "BASE_IMAGE=${PARFLOW_DOCKER_BASE_IMAGE}"
    -f "${CMAKE_CURRENT_SOURCE_DIR}/runtime/Dockerfile"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS
    DockerBuildDevelopment
  VERBATIM
  COMMENT
    "Build docker image 'parflow-runtime'"
)

add_custom_target(DockerTestDevelopment
  docker run --rm parflow-development /home/ubuntu/parflow/dependencies/cmake/bin/ctest
  DEPENDS
    DockerBuildDevelopment
  USES_TERMINAL
  COMMENT
    "Run parflow tests inside 'parflow-development' docker image"
)
