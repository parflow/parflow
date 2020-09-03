set(PARFLOW_DOCKER_BASE_IMAGE "ubuntu:20.04" CACHE STRING "Docker base image to use")
set(PARFLOW_DOCKER_VERSION "py-pftools" CACHE STRING "Version of ParFlow to install in docker (v3.6.0, master)")
set(PARFLOW_DOCKER_HYPRE "v2.19.0" CACHE STRING "Version of Hypre to install in docker (v2.19.0, master)")
set(PARFLOW_DOCKER_CACHE_TS "Default" CACHE STRING "Change value to trigger a new ParFlow pull in docker (Useful for incremental build)")

add_custom_target(DockerBuildDevelopment
  docker build --rm -t parflow-development
    --build-arg "BASE_IMAGE=${PARFLOW_DOCKER_BASE_IMAGE}"
    --build-arg "PARFLOW_VERSION=${PARFLOW_DOCKER_VERSION}"
    --build-arg "HYPRE_VERSION=${PARFLOW_DOCKER_HYPRE}"
    --build-arg "CACHE_DATE=${PARFLOW_DOCKER_CACHE_TS}"
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
  docker run --rm --workdir /home/ubuntu/parflow/build parflow-development /home/ubuntu/parflow/dependencies/cmake/bin/ctest
  DEPENDS
    DockerBuildDevelopment
  USES_TERMINAL
  COMMENT
    "Run parflow tests inside 'parflow-development' docker image"
)

add_custom_target(DockerTestPyDevelopment
  docker run --rm --workdir /home/ubuntu/parflow/build parflow-development /home/ubuntu/parflow/dependencies/cmake/bin/ctest -R py
  DEPENDS
    DockerBuildDevelopment
  USES_TERMINAL
  COMMENT
    "Run parflow tests inside 'parflow-development' docker image"
)

add_custom_target(DockerTestRuntime
  docker run --rm -v "${CMAKE_CURRENT_SOURCE_DIR}/../test/python:/tests" -it parflow-runtime  /tests/new_features/full_clone/full_clone.py
  DEPENDS
    DockerBuildRuntime
  USES_TERMINAL
  COMMENT
    "Run a parflow test inside 'parflow-runtime' docker image"
)

# -----------------------------------------------------------------------------
# Examples on how to run a test by hand using the runtime image
# -----------------------------------------------------------------------------
#
# docker run --rm \
#   -v "/home/seb/Documents/code/Intern/parflow/:/PF_SRC" \
#   -e PF_SRC=/PF_SRC \
#   -it parflow-runtime \
#   /PF_SRC/test/python/silo/indicator_field/indicator_field.py
#
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
#
# docker run --rm \
#   -v "/home/seb/Documents/code/Intern/parflow/:/PF_SRC" \
#   -e PF_SRC=/PF_SRC \
#   -it parflow-runtime \
#   /PF_SRC/test/python/base_3d/default_single/default_single.py \
#   -p 1 -q 1 -r 2
#
# -----------------------------------------------------------------------------
