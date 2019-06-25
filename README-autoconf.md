# Configuring with Autoconf

__IMPORTANT WARNING : Building Parflow with Autoconf (configure) is being deprecated;
support will be dropped in next release.__

CMake is the prefered method for building Parflow.  See the
README.md file for CMake instructions.  The configure script build is
still being tested for basic configurations but is not being actively
maintained.

Parflow is composed of two main components that are configured and
built seperately.  The main Parflow executable is built first then a
set of TCL libraries are built.  TCL is used for setting up a Parflow
run.  Since some MPP architectures use different
processors/OS/compilers for the compute nodes and login nodes Parflow
supports building the main simulation executable with a different set
compilers than the TCL libraries used for problem setup.

### Step 1: Setup

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
