# ParFlow

ParFlow is an open-source, modular, parallel watershed flow model. It
includes fully-integrated overland flow, the ability to simulate
complex topography, geology and heterogeneity and coupled land-surface
processes including the land-energy budget, biogeochemistry and snow
(via CLM). It is multi-platform and runs with a common I/O structure
from laptop to supercomputer. ParFlow is the result of a long,
multi-institutional development history and is now a collaborative
effort between CSM, LLNL, UniBonn and UCB. ParFlow has been coupled to
the mesoscale, meteorological code ARPS and the NCAR code WRF.

See the "User's Manual" for info on "Getting Started" in ParFlow.

# ParFlowVR
ParFlowVR is parflow extended by the FlowVR package to enable in-situ analysis and
computational steering.

### Citing Parflow

To cite Parflow, please use the following reference.

If you use ParFlow in a publication, please cite the these papers that describe model physics:

* Ashby S.F. and R.D. Falgout, Nuclear Science and Engineering 124:145-159, 1996
* Jones, J.E. and C.S. Woodward, Advances in Water Resources 24:763-774, 2001
* Kollet, S.J. and R.M. Maxwell, Advances in Water Resources 29:945-958, 2006
* Maxwell, R.M. Advances in Water Resources 53:109-117, 2013

If you use ParFlow coupled to CLM in a publication, please also cite
two additional papers that describe the coupled model physics:

* Maxwell, R.M. and N.L. Miller, Journal of Hydrometeorology 6(3):233-247, 2005
* Kollet, S.J. and R.M. Maxwell, Water Resources Research 44:W02402, 2008

### Additional Parflow resources

You can join the Parflow mailing list to communicate with the Parflow
developers and users.  Join our mailing list over via:
- [Parflow-Users](https://mailman.mines.edu/mailman/listinfo/parflow-users)

A Parflow blog is available with notes from users on how to compile and use Parflow:
- [Parflow Blog](http://parflow.blogspot.com/)

To report Parflow bugs, please use the GitHub issue tracker for Parflow:
- [Parflow Issue Tracker](https://github.com/parflow/parflow/issues)

## Quick Start

Parflow currently uses a configure/build system based on the standard
GNU autoconf configure system.  The steps to configure/build with GNU
autoconf are below.  The project is moving to CMake; the CMake
configure/build process is documented in the README-CMAKE.md file.

Parflow is composed of two main components that are configured and
built seperately.  The main Parflow executable is built first then a
set of TCL libraries are built.  TCL is used for setting up a Parflow
run.  Since some MPP architectures use different
processors/OS/compilers for the compute nodes and login nodes Parflow
supports building the main simulation executable with a different set
compilers than the TCL libraries used for problem setup.

### Step 1: Setup

**The instructions here are DEPRECATED! Refer to README.cmake to build this branch please!**

Decide where you wish to install Parflow and associated libraries.

Set the environment variable `PARFLOW_DIR` to your chosen location:

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

```shell
   mkdir ~/parflow
   cd ~/parflow
   gunzip ../parflow.tar.Z
   tar -xvf ../parflow.tar
```


### Step 3: Build and install Parflow

This step builds the Parflow library and executable.  The library is
used when Parflow is used as a component of another simulation
(e.g. WRF).

```shell
   cd $PARFLOW_DIR
   cd pfsimulator
   ./configure --prefix=$PARFLOW_DIR --with-amps=mpi1
   make
   make install
```

This will build a parallel version of Parflow using MPI libraries.

You can control build options for Parflow, use

```shell
   ./configure --help
```

to see other configure options.

To compile with CLM add `--with-clm` to the configure line

To change compilers used, set the `CC`, `FC` and `F77` variables. For example to use Intel compilers:

```shell
   setenv CC icc
   setenv FC ifort
   setenv F77 ifort
```

Many MPI distributions supply compiler wrappers (e.g. `mpicc`), simply
specify the wrapper name for the appropriate compiler variable.

Note that Parflow defaults to building a sequential version so
`--with-amps` is needed when building for a parallel computer.  You
can explicitly specify the MPI to use by using the provided compiler
wrapper (e.g. `mpicc`) or by specifying a path to the MPI install using
the `--with-mpi` option to configure.


### Step 4: Build and install pftools

pftools is a package of utilities and a TCL library that is used to
setup and postprocess Parflow files.  The input files to Parflow are
TCL scripts so TCL must be installed on the system.

A typical configure and build looks like:

```shell
  cd pftools
  ./configure --prefix=$PARFLOW_DIR --with-amps=mpi1
  make
  make install
  make doc_install
```

Note that pftools is NOT parallel but some options for how files are
written are based on the communication layer so pftools needs to know
what used to build the Parflow library.  Use the same
`--with-amps=<amps-option>` as was used to build the main executable.

If TCL is not installed in the system locations (/usr or /usr/local)
you need to specify the path with the `--with-tcl=<path-to-tcl>` configure
option.

Use `./configure --help` to list additional configure options for pftools.


### Step 5: Running a sample problem

If all went well a sample Parflow problem can be run using:

```shell
cd test
tclsh default_single.tcl 1 1 1
```

Note that `PAFLOW_DIR` must be set for this to work and it assumes
tclsh is in your path.  Make sure to use the same TCL as was used in
the pftools configure.


## Release

Copyright (c) 1995-2009, Lawrence Livermore National Security LLC.

Produced at the Lawrence Livermore National Laboratory.

Written by the Parflow Team (see the CONTRIBUTORS file)

CODE-OCEC-08-103. All rights reserved.

Parflow is released under the GNU General Public License version 2.1

For details and restrictions, please read the LICENSE.txt file.
- [LICENSE](./LICENSE.txt)
