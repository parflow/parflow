# Building ParFlow with Spack

This document covers building ParFlow with Spack.  Building with Spack may be useful
in some situations since Spack builds a complete dependency tree and
thus may have fewer compiling and linking issues.  Spack is a
multi-platform package manager that builds and installs multiple
versions and configurations of software. Spack has support for Linux,
macOS, and many supercomputers.

See [Spack documentation](https://spack.io) for additional information
on Spack and how to use it.

## Obtaining and installing Spack

The following will download Spack and setup the environment.  The
source command needs to be done in every new shell to set setup the
environment.

For bash:

```shell
git clone https://github.com/spack/spack.git
export SPACK_ROOT=$(pwd)/spack
source spack/share/spack/setup-env.sh
```

For tcsh/csh:

```shell
git clone https://github.com/spack/spack.git
setenv SPACK_ROOT `pwd`/spack
source spack/share/spack/setup-env.csh
```

## Building GNU compiler suite

This step is optional for Linux systems but is currently critical on
Mac platforms.  We have found issues with some releases of Clang
producing executable that do not yield correct results.  Clang is the
compiler used by Apple XCode so running ParFlow on MacOS exhibits the
issues.  We are currently investigating the issue, a work-around is to
compile the GNU compiler suite under Spack and avoid using Clang.
Building GCC will take considerable amount of time.

```shell
spack install gcc@12.2.0
```

Add the compiler to the set of compilers Spack can use:

```shell
spack load gcc@12.2.0
spack compiler find
spack compilers
```
## Building ParFlow

First download, configure, and build ParFlow and all of the libraries
ParFlow depends on.  Spack does all of this for you!  This step will
take a considerable amount of time; be prepared to wait.  You must be
connected to the internet so Spack can download everything.  You must
have a compiler suite (C/C++/Fortran) installed.

```shell
spack install parflow
```

## Running the Spack build of ParFlow 

First setup some environment variables for running out of the Spack
directories.  The Spack command has a location option to determine the
location of installs of TCL, OpenMPI and ParFlow.  We use the "spack
location" command to set some environment variables and then add some
bin directories to the path.  This will ensure that the Spack
installations will be used to run ParFlow.

For bash:

```bash
# The SPACK environment must be setup first, if you have not already done this
export SPACK_ROOT=$(pwd)/spack
source spack/share/spack/setup-env.sh

spack load parflow

# Are we using spack versions?
echo "Using PARFLOW_DIR : ${PARFLOW_DIR}"
echo "Using parflow     : $(which parflow)"
echo "Using tclsh       : $(which tclsh)"
echo "Using mpirun      : $(which mpirun)"
```

For tcsh/csh:

```shell
# The SPACK environment must be setup first, if you have not already done this
setenv SPACK_ROOT `pwd`/spack
source spack/share/spack/setup-env.csh

spack load parflow

# Are we using spack versions?
echo "Using PARFLOW_DIR : ${PARFLOW_DIR}"
echo "Using parflow     : `which parflow`"
echo "Using tclsh       : `which tclsh`"
echo "Using mpirun      : `which mpirun`"
```

## Testing 

### Quick test

A quick test to see if the build is working can be run with the Spack test command:

```shell
spack test run parflow
```

### Additional tests/examples

Spack is managing the directories for the ParFlow build so we checkout
the ParFlow repository to get a copy of the ParFlow tests in order
to access the test cases.

Clone the ParFlow repository:

```shell
git clone https://github.com/parflow/parflow.git
```

Run a few tests, the numbers indicate the parallel processor topology
so for 2x2x2 case you need 8 cores.  These tests may fail to run under
MPI if you do not have enough cores.  Examples are:

```shell
cd parflow/test/tcl

tclsh default_single.tcl 1 1 1

tclsh default_single.tcl 2 2 2

tclsh default_richards_with_netcdf.tcl 1 1 1

tclsh default_richards_with_netcdf.tcl 2 2 2
```
