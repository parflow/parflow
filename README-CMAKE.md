
## Quick Start with CMake on Unix

CMake support is beta; feedback is appriciated.  The Parflow project
will be moving to CMake and removing support for the GNU Autoconf.

Important change for users used to previous building, the configure
process is one step by default.  Most builds of of Parflow are now on
MPP architectures or workstations where the login node and compute
nodes are same archticture the default build process builds both the
Parflow executable and tools as one process.  This will hopefully make
building easier for the majority of users.  It is still possible to
build the two components seperatly; see instuction below.

CMake supports builds for other OS's and IDE tools (like Visual Studio
on Windows and XCode on MacOS).  The Parflow team has not tested
building on other platforms; there will likely be some issues.  We
welcome bug reports and patches if you attempt other builds.

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

### Step 3: Running CMake to configure Parflow

CMake is a utility that sets up makefiles for building parflow.  First
create a directory for the build.  It is generally recommend to build
outside of the source directory to make it keep things clean.  For
example, restarting a failed build with a seperate build directoy
simply involves removing the build directory.

#### Building with the ccmake GUI

You can control build options for Parflow using the ccmake GUI.

```shell
   mkdir build
   cd build
   ccmake ../parflow 
```
You will want to set the CMAKE_INSTALL_PREFIX value to the same thing
as PARFLOW_DIR was set to above.

#### Building with the cmake command line

CMake may also be configured from the command line.  The default will
configure a sequential version of Parflow using MPI libraries.  CLM is
being enabled.

```shell
   mkdir build
   cd build
   cmake ../parflow \
   	 -DCMAKE_INSTALL_PREFIX=$(PARFLOW_DIR) \
   	 -DPARFLOW_HAVE_CLM=ON
```

To build a parallel version of Parflow the communications layer to use
must be set.  The most common option will be MPI.  Here is a minimal
example of an MPI build with CLM:

```shell
   mkdir build
   cd build
   cmake ../parflow \
      	 -DCMAKE_INSTALL_PREFIX=$(PARFLOW_DIR) \
   	 -DPARFLOW_HAVE_CLM=ON \
	 -DPARFLOW_AMPS_LAYER=mpi1
```

Here is a more complex example where location of various external
packages are being specified and some features are being enabled.

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

If TCL is not installed in the system locations (/usr or /usr/local)
you need to specify the path with the <TODO> option.

### Step 4: Building and installing

Once CMake has configured and created a set of Makefiles; building is
easy:

```shell
   cd build
   make 
   make install
```

Use `./configure --help` to list additional configure options for pftools.


### Step 5: Running a sample problem

If all went well a sample Parflow problem can be run using:

```shell
cd parflow/test
tclsh default_single.tcl 1 1 1
```

Note that the envirobnment variable `PAFLOW_DIR` must be set for this
to work and it assumes tclsh is in your path.  Make sure to use the
same TCL as was used in the cmake configure.

## Building simulator and tools support seperately

Parflow is composed of two main components that maybe configured and
built seperately.  Some HPC platfroms are hetergenous with the login
node being different than the compute nodes.  The Parflow system has
an executable for the simulator which needs to run on the compute
nodes and a set of TCL libraries used for problem setup that may need
to run on the login node for problem setup.

The CMake variables PARFLOW_ENABLE_SIMULATOR and PARFLOW_ENABLE_TOOLS
control which component is configured.  By default both are true.  To
build seperately use to build directories and run cmake in each to
build the simulator and tools components seperately. By specifing
different compilers for each, one can build for different
archtictures for each component.

