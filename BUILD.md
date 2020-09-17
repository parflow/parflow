# ParFlow Build System

ParFlow use CMake to configure its build environment. The following guide
capture the various settings and target available for you to use.

## Building ParFlow

### Configuring CMake

ParFlow is composed of several components that can be enabled independently
from each other. The list below explain what those `PARFLOW_ENABLE_XXX`
are for and how you can configure them based on what you are looking for.

- __SIMULATOR__: The simulator is actually the core of ParFlow as it represent the simulation code.
- __DOCKER__: This provide helpers for building docker images with ParFlow enable in them.
- __DOXYGEN__: Doxygen and building of code documentation (C/Fortran).
- __ETRACE__: builds ParFlow with etrace
- __HDF5__: builds ParFlow with HDF5 which is required for the _NETCDF_ file format.
- __HYPRE__: builds ParFlow with Hypre
- __KEYS_DOC__: builds documentation (rst files) from key definitions.
- __LATEX__: enables LaTEX and building of documentation (Manual PDF)
- __NETCDF__: builds ParFlow with NetCDF. (If ON, HDF5 is required)
- __PROFILING__: This allow to enable extra code execution that would enable code profiling.
- __TIMING__: enables timing of key Parflow functions; may slow down performance
- __TOOLS__: enables building of the Parflow tools (TCL version)
- __VALGRIND__: builds ParFlow with Valgrind support
- __PYTHON__: This is to enable you to build the Python version of __pftools__.
- __SILO__: builds ParFlow with Silo.
- __SLURM__: builds ParFlow with SLURM support (SLURM is queuing system on HPC).
- __SUNDIALS__: builds ParFlow with SUNDIALS
- __SZLIB__: builds ParFlow with SZlib compression library
- __ZLIB__: builds ParFlow with Zlib compression library

### Standard ParFlow build

When building ParFlow we tend to configure it with the following set of options:

| CMake property (basic)     | Value       |
| -------------------------- | ----------- |
| CMAKE_BUILD_TYPE           | Release     |
| HYPRE_ROOT                 | $HYPRE_DIR  |
| PARFLOW_ENABLE_HYPRE       | TRUE        |
| PARFLOW_HAVE_CLM           | TRUE        |
| PARFLOW_ENABLE_TIMING      | TRUE        |
| PARFLOW_AMPS_LAYER         | mpi1        |
| PARFLOW_AMPS_SEQUENTIAL_IO | TRUE        |

| CMake property (+py-pftools) | Value |
| ---------------------------- | ----- |
| PARFLOW_ENABLE_PYTHON        | TRUE  |
| PARFLOW_PYTHON_VIRTUAL_ENV   | TRUE  |

| CMake property (+netcdf)   | Value       |
| -------------------------- | ----------- |
| PARFLOW_ENABLE_HDF5        | TRUE        |
| PARFLOW_ENABLE_NETCDF      | TRUE        |
| HDF5_ROOT                  | $HDF5_DIR   |
| NETCDF_DIR                 | $NETCDF_DIR |

| CMake property (+silo)     | Value       |
| -------------------------- | ----------- |
| PARFLOW_ENABLE_SILO        | TRUE        |
| SILO_ROOT                  | $SILO_DIR   |

## Docker

Rather than building ParFlow on your computer, you can use the build system to
create a virtual machine for you and build ParFlow for you in it.

    cmake                        \
      -S ./parflow                \
      -B ./build-docker            \
      -D BUILD_TESTING=OFF          \
      -D PARFLOW_ENABLE_TOOLS=OFF    \
      -D PARFLOW_ENABLE_SIMULATOR=OFF \
      -D PARFLOW_ENABLE_DOCKER=ON

    cd ./build-docker && make DockerBuildRuntime

For more information look into our [Docker Readme](./docker/README.md)

## Python

If you just want to just build the new pftools without ParFlow, you can
configure your project as follow.

    cmake                        \
      -S ./parflow                \
      -B ./build-docker            \
      -D BUILD_TESTING=OFF          \
      -D PARFLOW_ENABLE_TOOLS=OFF    \
      -D PARFLOW_ENABLE_SIMULATOR=OFF \
      -D PARFLOW_ENABLE_PYTHON=ON      \
      -D PARFLOW_PYTHON_VIRTUAL_ENV=ON

    # Build the python package
    cmake --build ./build-docker

    # Install the python package inside ./install/python
    cmake --install ./build-docker --prefix ./install

    # This will create wheels locally
    make PythonCreatePackage

    # This will publish the wheels online
    make PythonPublishPackage

## Documentation

### ParFlow keys documentation

    cmake                        \
      -S ./parflow                \
      -B ./build-docker            \
      -D BUILD_TESTING=OFF          \
      -D PARFLOW_ENABLE_TOOLS=OFF    \
      -D PARFLOW_ENABLE_SIMULATOR=OFF \
      -D PARFLOW_ENABLE_KEYS_DOC=ON    \
      -D PARFLOW_ENABLE_PYTHON=ON       \
      -D PARFLOW_PYTHON_VIRTUAL_ENV=ON

    cd ./build-docker && make ParFlowKeyDoc

    open ./build-docker/docs/pf-keys/build-site/index.html

### ParFlow Manual

Latex [...]

### ParFlow API

Doxygen [...]
