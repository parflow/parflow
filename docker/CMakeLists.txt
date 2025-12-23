add_custom_target(DockerBuildDevelopment
  docker build --rm -t parflow-development
    # --build-arg "BASE_IMAGE=ubuntu:20.04"
    # --build-arg "CMAKE_URL=https://cmake.org/files/v3.18/cmake-3.18.2-Linux-x86_64.tar.gz"
    # --build-arg "HDF5_URL=https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.0/src/hdf5-1.12.0.tar.gz"
    # --build-arg "NETCDF_URL=https://github.com/Unidata/netcdf-c/archive/v4.7.4.tar.gz"
    # --build-arg "SILO_URL=https://wci.llnl.gov/sites/wci/files/2021-01/silo-4.10.2.tgz"
    # --build-arg "HYPRE_VERSION=v2.19.0"
    -f "${CMAKE_CURRENT_SOURCE_DIR}/dev/Dockerfile"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
  COMMENT
    "Build docker image 'parflow-development'"
  VERBATIM
)

add_custom_target(DockerBuildRuntime
  docker build --rm -t parflow-runtime
    # --build-arg "BASE_IMAGE=ubuntu:20.04"
    -f "${CMAKE_CURRENT_SOURCE_DIR}/runtime/Dockerfile"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS
    DockerBuildDevelopment
  VERBATIM
  COMMENT
    "Build docker image 'parflow-runtime'"
)

add_custom_target(DockerTestDevelopment
  docker run --rm --workdir /home/ubuntu/parflow/build parflow-development /home/ubuntu/parflow/dependencies/cmake/bin/ctest --output-on-failure
  DEPENDS
    DockerBuildDevelopment
  USES_TERMINAL
  COMMENT
    "Run parflow tests inside 'parflow-development' docker image"
)

add_custom_target(DockerTestPyDevelopment
  docker run --rm --workdir /home/ubuntu/parflow/build parflow-development /home/ubuntu/parflow/dependencies/cmake/bin/ctest -R py --output-on-failure
  DEPENDS
    DockerBuildDevelopment
  USES_TERMINAL
  COMMENT
    "Run parflow tests inside 'parflow-development' docker image"
)

add_custom_target(DockerTestRuntime
  docker run --rm -v "${CMAKE_CURRENT_SOURCE_DIR}/../test/python:/tests" -v "${CMAKE_CURRENT_SOURCE_DIR}/..:/PF_SRC" -e PF_SRC=/PF_SRC -it parflow-runtime  /tests/new_features/full_clone.py
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