# Building ParFlow with Spack

This document covers building ParFlow with Spack.  This feature is
experimental and under development.  Building with Spack may be useful
in some situations since Spack builds a complete dependency tree and
thus may have fewer compiling and linking issues.  Spack is a
multi-platform package manager that builds and installs multiple
versions and configurations of software. Spack has support for Linux,
macOS, and many supercomputers.

See [Spack documentation](https://spack.io) for additional information
on Spack and how to use it.

## Obtaining and installing Spack

A special version of Spack is currently needed for a ParFlow build.
There are only two small changes to Spack to address a Silo build
issue and adding the rules for the ParFlow configuration.  Once the
ParFlow build is stable we will ask for ParFlow to be added to the
standard Spack distribution.

The following will download Spack and setup the environment.  The
source command needs to be done in every new shell to set setup the
environment.

For bash:

```shell
git clone https://github.com/parflow/spack.git
export SPACK_ROOT=$(pwd)/spack
source spack/share/spack/setup-env.sh
```

For tcsh/csh:

```shell
git clone https://github.com/parflow/spack.git
setenv SPACK_ROOT `pwd`/spack
source spack/share/spack/setup-env.csh
```

## Building ParFlow

First download, configure, and build ParFlow and all of the libraries
ParFlow depends on.  Spack does all of this for you!  This step will
take a considerable amount of time to be prepared to wait.  You must
be connected to the internet so Spack can download everything.  You
must have a compiler suite (C/C++/Fortran) installed.

```shell
spack install parflow@develop
```

Hopefully everything built correctly.

## Running the Spack build of ParFlow 

First setup some environment variables for running out of the Spack
directories.  The Spack command has a location option to determine the
location of installs of TCL, OpenMPI and ParFlow.  We use the "spack
location" command to set some environment variables and then add some
bin directories to the path.  This will ensure that the Spack
installations will be used to run ParFlow.

For bash:

```shell
export PARFLOW_DIR=$(spack location --install-dir parflow)
export PARFLOW_TCL_DIR=$(spack location --install-dir tcl)
export PARFLOW_MPI_DIR=$(spack location --install-dir openmpi)
export PATH=${PARFLOW_MPI_DIR}/bin:${PARFLOW_TCL_DIR}/bin:${PARFLOW_DIR}/bin:$PATH

# Are we using spack versions?
echo "Using PARFLOW_DIR=${PARFLOW_DIR}"
echo "Using tclsh : $(which tclsh)"
echo "Using mpirun : $(which mpirun)"
```

For tcsh/csh:

```shell
setenv PARFLOW_DIR `spack location --install-dir parflow`
setenv PARFLOW_TCL_DIR `spack location --install-dir tcl`
setenv PARFLOW_MPI_DIR `spack location --install-dir openmpi`
set path = ( ${PARFLOW_MPI_DIR}/bin ${PARFLOW_TCL_DIR}/bin ${PARFLOW_DIR}/bin $path )
setenv export PATH 

# Are we using spack versions?
echo "Using PARFLOW_DIR=${PARFLOW_DIR}"
echo "Using tclsh : `which tclsh`"
echo "Using mpirun : `which mpirun`"
```

We are looking at how to make this installation simpler; possibly using a Spack view.

## Testing 

Spack is managing the directories for the ParFlow build so we checkout
the ParFlow repository to get a copy of the ParFlow regression tests
so we can run them.

Clone the ParFlow repository:

```shell
git clone https://github.com/parflow/parflow.git
```

Run a few tests:

```shell
cd parflow/test

tclsh default_single.tcl 1 1 1

tclsh default_single.tcl 2 2 2

tclsh default_richards_with_netcdf.tcl 1 1 1

tclsh default_richards_with_netcdf.tcl 2 2 2
```

Hopefully these all pass!

To run the full test suite:

```shell
cd parflow/test
make test
```

Some of the tests that require a large number of MPI ranks may fail
(The error message is something like "There are not enough slots
available in the system to satisfy the 27 slots") if you don't have
enough cores.  The test suite goes up to 27 cores.  We are working to
fix this issue (for MPI folks, over-committing is not being enabled or
working in the Spack OpenMPI setup).
