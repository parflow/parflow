
## Quick Start with CMake on Unix/Linux

CMake support is beta; feedback is appreciated.  The ParFlow project
will be moving to CMake and removing support for the GNU Autoconf in a
future release.

Important change for users that have built with Autoconf, the CMake
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
   tar -xvf ../parflow.tar.gz
```

Note the ParFlow tar file will be have a different name based on the
version number.

If you are not using GNU tar or have a very old version GNU tar you
will need to uncompress the file first:

```shell
   mkdir ~/parflow
   cd ~/parflow
   gunzip ../parflow.tar.gz
   tar -xvf ../parflow.tar
```

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
   ccmake ../parflow
```
At a minimum, you will want to set the CMAKE_INSTALL_PREFIX value to the same thing
as PARFLOW_DIR was set to above.  Other variables should be set as desired.

After setting a variable 'c' will configure `ParFlow.  When you are
completely done setting configuration options, use 'g' to generate the
configuration and exit ccmake.

If you are new to CMake, the creators of CMake provide some additional ccmake usage notes here:

https://cmake.org/runningcmake/

#### Building with the cmake command line

CMake may also be configured from the command line using the cmake
command.  The default will configure a sequential version of ParFlow
using MPI libraries.  CLM is being enabled.

```shell
   mkdir build
   cd build
   cmake ../parflow \
   	 -DCMAKE_INSTALL_PREFIX=$(PARFLOW_DIR) \
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
      	 -DCMAKE_INSTALL_PREFIX=$(PARFLOW_DIR) \
   	 -DPARFLOW_HAVE_CLM=ON \
	 -DPARFLOW_AMPS_LAYER=mpi1
```
**Often you also want to set some mpi options defining communication used
and so on. Thus refer to
[How to specify command to run MPI applications](#how-to-specify-command-to-run-mpi-applications)
at the end of this document!**

Here is a more complex example where location of various external
packages are being specified and some features are being enabled:

```shell
   mkdir build
   cd build
   cmake ../parflow \
        -DPARFLOW_AMPS_LAYER=mpi1 \
	-DHYPRE_ROOT=$(PARFLOW_HYPRE_DIR) \
	-DHDF5_ROOT=$(PARFLOW_HDF5_DIR) \
	-DSILO_ROOT=$(PARFLOW_SILO_DIR) \
	-DSUNDIALS_ROOT=$(PARFLOW_SUNDIALS_DIR) \
	-DCMAKE_BUILD_TYPE=Debug \
	-DPARFLOW_ENABLE_TIMING=TRUE \
	-DPARFLOW_HAVE_CLM=ON \
	-DCMAKE_INSTALL_PREFIX=$(INSTALL_DIR)
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
cd parflow/test
tclsh default_single.tcl 1 1 1
```

Note that the environment variable `PAFLOW_DIR` must be set for this
to work and it assumes tclsh is in your path.  Make sure to use the
same TCL shell as was used in the cmake configure.

Some parallel machines do not allow launching a parallel executable
from the login node; you may need to run this command in a batch file
or by starting a parallel interactive session.

## Configure options

A number of packages are optional for building ParFlow.  The optional
packages are enabled by PARFLOW_ENABLE_<package> value to be `TRUE` or
setting the <package>_ROOT=<directory> value.  If a package is enabled
with the using an ENABLE flag CMake will attempt to find the package
in standard locations.  Explicitly setting the location using the ROOT
variable for a package automatically enables it, you don't need to
specify both values.

### How to specify the command to run MPI applications

There are multiple ways to run MPI applications such as mpiexec,
mpirun, srun, and aprun.  The command used is dependent on the job
submission system used.  By default CMake will attempt to determine an
appropriate tool; a process that does not always yield the correct result.

You may overwride the MPI launcher using
```shell
-DMPIEXEC="<launcher-name>"
-DMPIEXEC_NUMPROC_FLAG="<flag used to set number of tasks>"
-DMPIEXEC_POSTFLAGS="<post flags>"
-DMPIEXEC_PREFLAGS="<pre flags>"
```


An example for mpiexec is -DMPIEXEC="mpiexec" -DMPIEXEC_NUMPROC_FLAG="-n".

The ParFlow script to run MPI applications will also include options
specified in the environment variable PARFLOW_MPIEXEC_EXTRA_FLAGS on
the MPI execution command line.  For example when running with OpenMPI
on a single workstation the following will enable running more MPI
tasks than cores and disable the busy loop waiting to
improve performance:

```shell
   export PARFLOW_MPIEXEC_EXTRA_FLAGS="--mca mpi_yield_when_idle 1 --oversubscribe"
```

## Building simulator and tools support separately

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


## Building with parFlowVR

The following options can be activated to build with parFlowVR support:
Build with FlowVR support this option must be activated for all the following.
```shell
PARFLOW_ENABLE_FLOWVR=ON
```

Enable building of the FlowVR tools. It is absolutely recommended to built them as
otherwise starting of parFlowVR distributed applications will become a nightmare.
```shell
-DPARFLOW_ENABLE_FLOWVR_TOOLS=True
```

Build with NetCDF. This is necessary to use the netcdf writer module in parFlowVR
```shell
-DPARFLOW_ENABLE_NETCDF=ON
```

Enable building of the Parflow simulator. Only the Simulator can be uses in parFlowVR
```shell
-DPARFLOW_ENABLE_SIMULATOR=True
```

Enable building Python Analyzer API. Highly recommended.
```shell
-DBUILD_PYTHON_ANALYZER_API=ON
-DNUMPY_I_PATH=<Directory where to find numpy.i>
```

Enable building of the Visit Connector (visit libsim is needed!). This is optional.
```shell
-DPARFLOW_ENABLE_VISIT_CONNECTOR=ON
```
