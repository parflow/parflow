# ParFlow

![ParFlow Linux CI Test](https://github.com/parflow/parflow/actions/workflows/linux.yml/badge.svg)
![ParFlow MacOS CI Test](https://github.com/parflow/parflow/actions/workflows/macos.yml/badge.svg)

ParFlow is an open-source, modular, parallel watershed flow model. It
includes fully-integrated overland flow, the ability to simulate
complex topography, geology and heterogeneity and coupled land-surface
processes including the land-energy budget, biogeochemistry and snow
(via CLM). It is multi-platform and runs with a common I/O structure
from laptop to supercomputer. ParFlow is the result of a long,
multi-institutional development history and is now a collaborative
effort between CSM, LLNL, UniBonn and UCB. ParFlow has been coupled to
the mesoscale, meteorological code ARPS and the NCAR code WRF.

For an overview of the major features and capabilities see the
following paper: [Simulating coupled surface–subsurface flows with
ParFlow v3.5.0: capabilities, applications, and ongoing development of
an open-source, massively parallel, integrated hydrologic
model](https://www.geosci-model-dev.net/13/1373/2020/gmd-13-1373-2020.pdf).

An online version of the users manual is available on [Read the
Docks:Parflow Users
Manual](https://parflow.readthedocs.io/en/latest/index.html).  The
manual contains additional documentation on how to use ParFlow and
setup input files.  A quick start is included below.  A PDF version is
available at [Parflow Users
Manual PDF](https://parflow.readthedocs.io/_/downloads/en/latest/pdf/).

### Citing Parflow

If you want the DOI for a specific release see:
[Zendo](https://zenodo.org/search?page=1&size=20&q=parflow&version)

A generic DOI that always links to the most current release :
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4816884.svg)](https://doi.org/10.5281/zenodo.4816884)

If you use ParFlow in a publication and wish to cite a paper reference
please use the following that describe model physics:

* Ashby S.F. and R.D. Falgout, Nuclear Science and Engineering 124:145-159, 1996
* Jones, J.E. and C.S. Woodward, Advances in Water Resources 24:763-774, 2001
* Kollet, S.J. and R.M. Maxwell, Advances in Water Resources 29:945-958, 2006
* Maxwell, R.M. Advances in Water Resources 53:109-117, 2013

If you use ParFlow coupled to CLM in a publication, please also cite
two additional papers that describe the coupled model physics:

* Maxwell, R.M. and N.L. Miller, Journal of Hydrometeorology 6(3):233-247, 2005
* Kollet, S.J. and R.M. Maxwell, Water Resources Research 44:W02402, 2008

### Additional Parflow resources

The ParFlow website has additional information on the project:
- [Parflow Web Site](https://parflow.org/)

You can join the Parflow Google Group/mailing list to communicate with
the Parflow developers and users.  In order to post you will have to
join the group, old posts are visible without joining:
- [Parflow-Users](https://groups.google.com/g/parflow)

The most recent build/installation guides are now located on the Parflow Wiki:
- [Parflow Installation guides](https://github.com/parflow/parflow/wiki/ParFlow-Installation-Guides)

A Parflow blog is available with notes from users on how to use Parflow:
- [Parflow Blog](http://parflow.blogspot.com/)

To report Parflow bugs, please use the GitHub issue tracker for Parflow:
- [Parflow Issue Tracker](https://github.com/parflow/parflow/issues)

## Quick Start on Unix/Linux

Important note for users that have built with Autoconf, the CMake
configure process is one step by default.  Most builds of of ParFlow
are on MPP architectures or workstations where the login node and
compute nodes are same architecture the default build process builds
both the ParFlow executable and tools with the same compilers and
libraries in one step.  This will hopefully make building easier for
the majority of users.  It is still possible to build the two
components separately; see instruction below for building pftools and
pfsimulator separately.

CMake supports builds for several operating systems and IDE tools
(like Visual Studio on Windows and XCode on MacOS).  The ParFlow team
has not tested building on platforms other than Linux; there will
likely be some issues on other platforms.  The ParFlow team welcomes
bug reports and patches if you attempt other builds.

### Step 1: Setup

Decide where to install ParFlow and associated libraries.

Set the environment variable `PARFLOW_DIR` to the chosen location:

For bash:

```shell
   export PARFLOW_DIR=/home/snoopy/parflow
```

For csh and tcsh:

```shell
   setenv PARFLOW_DIR /home/snoopy/parflow
```

### Step 2: Extract the Source

Extract the source files from the compressed tar file.

Obtain the release from the ParFlow GitHub web site:

https://github.com/parflow/parflow/releases

and extract the release.  Here we assume you are building in new
subdirectory in your home directory:

```shell
   mkdir ~/parflow
   cd ~/parflow
   tar -xzvf ../parflow-<version>.tar.gz
```

Note the ParFlow tar file will be have a different name based on the
version number.

### Step 3: Running CMake to configure ParFlow

CMake is a utility that sets up makefiles for building ParFlow.  CMake
allows setting of compiler to use and other options.  First create a
directory for the build.  It is generally recommend to build outside
of the source directory to make it keep things clean.  For example,
restarting a failed build with a separate build directory simply
involves removing the build directory.

#### Building with the ccmake GUI

You can control build options for ParFlow using the ccmake GUI.

```shell
   mkdir build
   cd build
   ccmake ../parflow-<version>
```

First press `c` to generate an initial configuration.  Hereafter, at a minimum,
you will want to set the `CMAKE_INSTALL_PREFIX` value to the same thing
as `PARFLOW_DIR` was set to above.  Other variables should be set as desired.

After setting a variable `c` will configure ParFlow.  When you are
completely done setting configuration options, use `g` to generate the
configuration and exit ccmake.

If you are new to CMake, the creators of CMake provide some additional ccmake usage notes here:

https://cmake.org/resources/

#### Building with the cmake command line

CMake may also be configured from the command line using the cmake
command. Instructions to build with different accelerator backends are found from the following documents: [CUDA, KOKKOS](README-GPU.md), [OpenMP](README-OPENMP.md). The default will configure a sequential version of ParFlow
using MPI libraries.  CLM is being enabled.

```shell
   mkdir build
   cd build
   cmake ../parflow-<version> \
     -DCMAKE_INSTALL_PREFIX=${PARFLOW_DIR} \
     -DPARFLOW_HAVE_CLM=ON
```

If TCL is not installed in the standard locations (/usr or /usr/local)
you need to specify the path to the tclsh location:

```shell
    -DTCL_TCLSH=${PARFLOW_TCL_DIR}/bin/tclsh8.6
```

Building a parallel version of ParFlow requires the communications
layer to use must be set.  The most common option will be MPI.  Here
is a minimal example of an MPI build with CLM:

```shell
   mkdir build
   cd build
   cmake ../parflow \
     -DCMAKE_INSTALL_PREFIX=${PARFLOW_DIR} \
     -DPARFLOW_HAVE_CLM=ON \
     -DPARFLOW_AMPS_LAYER=mpi1
```

Here is a more complex example where location of various external
packages are being specified and some features are being enabled:

```shell
   mkdir build
   cd build
   cmake ../parflow-<version> \
     -DPARFLOW_AMPS_LAYER=mpi1 \
     -DHYPRE_ROOT=${PARFLOW_HYPRE_DIR} \
     -DHDF5_ROOT=${PARFLOW_HDF5_DIR} \
     -DSILO_ROOT=${PARFLOW_SILO_DIR} \
     -DCMAKE_BUILD_TYPE=Debug \
     -DPARFLOW_ENABLE_TIMING=TRUE \
     -DPARFLOW_HAVE_CLM=ON \
     -DCMAKE_INSTALL_PREFIX=${PARFLOW_DIR}
```

### Step 4: Building and installing

Once CMake has configured and created a set of Makefiles; building is
easy:

```shell
   cd build
   make
   make install
```

### Step 5: Running a sample problem

If all went well a sample ParFlow problem can be run using:

```shell
   cd parflow-<version>/test/tcl
   tclsh default_single.tcl 1 1 1
```

Note that the environment variable `PARFLOW_DIR` must be set for this
to work and it assumes tclsh is in your path.  Make sure to use the
same TCL shell as was used in the cmake configure.

Some parallel machines do not allow launching a parallel executable
from the login node; you may need to run this command in a batch file
or by starting a parallel interactive session.

## Building documentation

### User Manual

An online version of the user manual is also available on [Read the
Docks:Parflow Users
Manual](https://parflow.readthedocs.io/en/latest/index.html), a PDF
version is available at [Parflow Users
Manual PDF](https://parflow.readthedocs.io/_/downloads/en/latest/pdf/).

#### Generating the user manual in HTML

An HTML version of the user manual for Parflow may be built using:

```shell
cd docs/user_manual
pfpython -m pip install -r requirements.txt

make html
```

The main HTML page created at _build/html/index.html.   Open this using
a browser.  On MacOS:

```shell
open _build/html/index.html
```

or a browser if on Linux:

```shell
firefox _build/html/index.html
```

#### Generating the user manual in PDF

An HTML version of the user manual for Parflow may be built using:

```shell
cd docs/user_manual
pfpython -m pip install -r requirements.txt

make latexpdf
```

This command is currently failing for a number of users, possibly due
to old LaTex installs.  We are currently investigating.

### Code documentation

Parflow is moving to using Doxygen for code documentation.  The documentation is currently very sparse.

Adding the -DPARFLOW_ENABLE_DOXYGEN=TRUE option to the CMake configure
will enable building of the code documentation.  After CMake has been
run the Doxygen code documentation is built with:

```shell
   cd build
   make doxygen
```

HTML pages are generated in the build/docs/doxygen/html directory.

### ParFlow keys documentation

```shell
   cmake \
      -S ./parflow \
      -B ./build-docker \
      -D BUILD_TESTING=OFF \
      -D PARFLOW_ENABLE_TOOLS=OFF \
      -D PARFLOW_ENABLE_SIMULATOR=OFF \
      -D PARFLOW_ENABLE_KEYS_DOC=ON \
      -D PARFLOW_ENABLE_PYTHON=ON \

    cd ./build-docker && make ParFlowKeyDoc
```

On MacOS the key documentation may be viewed with `open` or use a browser to open the index.html file:

```
    open ./build-docker/docs/user_manual/build-site/index.html
```

## Configure options

A number of packages are optional for building ParFlow.  The optional
packages are enabled by PARFLOW_ENABLE_<package> value to be `TRUE` or
setting the <package>_ROOT=<directory> value.  If a package is enabled
with the using an ENABLE flag CMake will attempt to find the package
in standard locations.  Explicitly setting the location using the ROOT
variable for a package automatically enables it, you don't need to
specify both values.

Here are some common packages:

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

### KINSOL Solver

There are two compile options for the KINSOL solver used by ParFlow.
The current default is to use an old version of KINSOL that is
embedded in ParFlow.  The newer option is to use KINSOL from the
SUNDIALS package.   To use the SUNDIALS package you will need to have a 
SUNDIALS install that includes KINSOL and specify during the CMake configuration:

```shell
   -DSUNDIALS_ROOT=<directory of the location of sundials install>
```

The GitHub repository for [SUNDIALS](https://github.com/LLNL/sundials)
has downloads and build instructions.

### How to specify the launcher command used to run MPI applications

There are multiple ways to run MPI applications such as mpiexec,
mpirun, srun, and aprun.  The command used is dependent on the job
submission system used.  By default CMake will attempt to determine an
appropriate tool; a process that does not always yield the correct result.

There are several ways to modify the CMake guess on how applications
should be run.  At configure time you may override the MPI launcher
using:

```shell
   -DMPIEXEC="<launcher-name>"
   -DMPIEXEC_NUMPROC_FLAG="<flag used to set number of tasks>"
```

An example for mpiexec is -DMPIEXEC="mpiexec" -DMPIEXEC_NUMPROC_FLAG="-n".

The ParFlow script to run MPI applications will also include options
specified in the environment variable PARFLOW_MPIEXEC_EXTRA_FLAGS on
the MPI execution command line.  For example when running with OpenMPI
on a single workstation the following will enable running more MPI
tasks than cores and disable the busy loop waiting to improve
performance:

```shell
   export PARFLOW_MPIEXEC_EXTRA_FLAGS="--mca mpi_yield_when_idle 1 --oversubscribe"
```

Last the TCL script can explicitly set the command to invoke for
running ParFlow.  This is done by setting the Process.Command key in
the input database.  For example to use the mpiexec command and
control the cpu set used the following command string can be used:

```shell
   pfset Process.Command "mpiexec -cpu-set 1 -n %d parflow %s"
```

The '%d' will be replaced with the number of processes (computed using
the Process.Topology values : P * Q * R) and the '%s' will be replaced
by the name supplied to the pfrun command for the input database name.
The following shows how the default_single.tcl script could be
modified to use the custom command string:

```shell
   pfset Process.Command "mpiexec -cpu-set 1 -n %d parflow %s"
   pfrun default_single
   pfundist default_single
```
## Building simulator and tools support separately

This section is for advanced users running on heterogeneous HPC architectures.

ParFlow is composed of two main components that maybe configured and
built separately.  Some HPC platforms are heterogeneous with the login
node being different than the compute nodes.  The ParFlow system has
an executable for the simulator which needs to run on the compute
nodes and a set of TCL libraries used for problem setup that can be
run on the login node for problem setup.

The CMake variables PARFLOW_ENABLE_SIMULATOR and PARFLOW_ENABLE_TOOLS
control which component is configured.  By default both are `TRUE`.  To
build separately use two build directories and run cmake in each to
build the simulator and tools components separately. By specifying
different compilers and options for each, one can target different
architectures for each component.

# Using Docker

ParFlow includes a Docker file for configuring a Docker image for
running ParFlow.

## Pre-built Docker Image

A Docker image for ParFlow is available on Docker hub.  See the
following section for how to run the Docker image.  The Docker
latest image is automatically downloaded by Docker when run.

## Running ParFlow with Docker

The https://github.com/parflow/docker repository contains an example
setup for running ParFlow in a Docker instance.  See the README.md
file in this repository for more information.

## Building the Docker image

If you want to build a Docker image, the build script in the bin
directory will build an image using the latest ParFlow source in the
master branch.  If you want to build a different version of ParFlow
you will need to modify the 'Dockerfile' file.

### Unix/Linux/MacOS

```shell
./bin/docker-build.sh
```

### Windows

```PowerShell
.\bin\docker-build.bat
```

## Building the Docker image with CMake (expirmental not supported)

Rather than building ParFlow on your computer, you can use the build
system to create a container and build ParFlow in it.

```shell
cmake \
   -S ./parflow \
   -B ./build-docker \
   -D BUILD_TESTING=OFF \
   -D PARFLOW_ENABLE_TOOLS=OFF \
   -D PARFLOW_ENABLE_SIMULATOR=OFF \
   -D PARFLOW_ENABLE_DOCKER=ON

cd ./build-docker && make DockerBuildRuntime
```

For more information look into our [Docker Readme](./docker/README.md)


## Release

Copyright (c) 1995-2021, Lawrence Livermore National Security LLC.

Produced at the Lawrence Livermore National Laboratory.

Written by the Parflow Team (see the CONTRIBUTORS file)

CODE-OCEC-08-103. All rights reserved.

Parflow is released under the GNU General Public License version 2.1

For details and restrictions, please read the LICENSE.txt file.
- [LICENSE](./LICENSE.txt)
