Linux Install Guide
===================

Ubuntu 20.04.1 LTS — factory condition. February 28, 2021. By Nick Engdahl (nick.engdahl@wsu.edu).

Overview
--------

After much neglect of linux, this installation guide covers the installation of Parflow and it's required supporting software on Ubuntu 20.04.1 LTS. It's been tested in Parallels Desktop and on native hardware with identical results and syntax on both. As always, we have to assume a basic level of computational knowledge (i.e. command line). The steps we'll be following are:

1. Setup the environment and compilers
2. Install the CMake package
3. Install OpenMPI
4. Install Hypre library
5. Install Silo library
6. Install ParFlow

Commands entered at the command line are prefaced with a '> ' and everything after can be copied and pasted. Sometimes double hyphens and quotes don't copy/paste well so you may need to go back a correct those. *Note that you must have admin privileges on your system to do this because we need to use "sudo" to install some packages.* 

1.0 - Setting up the environment
--------------------------------

By default Ubuntu doesn't have all the required programs/libraries we need so we need to make sure those get setup first. Also, we use some environment variables in the ParFlow install so we'll go ahead and set those now too. 

Installing many necessary things
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first thing we need to do is install a fortran compiler; a C/C++ compiler is already bundled with Ubuntu so we don't need to mess with it. The simplest way to do this is

.. code-block:: none

   >  sudo apt install gfortran

verify that the compiler is working with

.. code-block:: none

   > gfortran -v
   Using built-in specs.
   COLLECT_GCC=gfortran
   ...

with several more lines following that with the specifics. If you don't see that something went wrong.

The remainder of what we need are a text editor (we're going to use vim), the git command interface, and the TCL scripting language with developer extensions. The commands for those are:

.. code-block:: none

   > sudo apt install vim
   > sudo apt install gfortran
   > sudo apt install git 
   > sudo apt install tcl
   > sudo apt install tcl8.6-dev

If anything goes wrong you'll need to troubleshoot and get that resolved before moving on but Ubuntu is usually pretty good about installing those for us.

Defining the environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We'll use a handful of shortcuts in this installation so we need to add a few lines to the shell so these are defined automatically. Depending on how you login these might go in your ``.bash_profile`` but I'm adding them to the *bottom* of my ``.bashrc`` using vim. 
Add the following lines:

.. code-block:: none

   export HYPRE_DIR=~/PFTree/hypre
   export SILO_DIR=~/PFTree/silo
   export PARFLOW_DIR=~/PFTree/parflow
   export MPI_DIR=~/PFTree/OMPI-4.1.0
   export MPI_RUN_DIR=~/PFTree/OMPI-4.1.0/bin
   
   export PATH=$MPI_RUN_DIR:$PATH

The be sure to source the file before continuing with (for example) ``source .bashrc``
We'll use these later on but note that if you decide to call your specific directories by another name these definitions will need to be updated.

2.0 - CMake setup
-----------------

CMake is used to build the ParFlow package and can be used to build some of the supporting packages. We're going to use the "bootstrap" installation script provided with CMake and you can find the latest build on the CMake downloads page here: `CMake downloads <https://cmake.org/download/>`_

Build a "home" for ParFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~

Since we're installing a bunch of stuff that is associated with ParFlow it's usually nice to keep it all together, so we'll create a directiry to hold all our ParFlow things. In a terminal we'll do the following

.. code-block:: none

   > cd ~
   > mkdir PFTree
   > cd PFTree

From there we can get CMake setup with the following (update version numbers as needed)

.. code-block:: none

   > cd ~/PFTree
   > wget https://github.com/Kitware/CMake/releases/download/v4.4.5/cmake-4.4.5.tar.gz
   > gunzip cmake-4.4.5.tar.gz 
   > tar -xvf cmake-4.4.5.tar
   > cd cmake-4.4.5/
   > ./bootstrap -- -DCMAKE_USE_OPENSSL=OFF
   > make
   > sudo make install

Be particularly careful with the ``./bootstrap -- -DCMAKE_USE_OPENSSL=OFF`` command if you copy/paste, but that should take care of the setup

3.0 - OpenMPI installation
--------------------------

The way we're going to install OpenMPI will put it in a non-standard location. The reason for this is that we'll be able to more easily update/switch versions if necessary and this is the reason we defined ``$MPI_DIR`` and ``$MPI_DIR_RUN`` which are the locations we'll install the package and binaries to, respectively. 

Download and install OpenMPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You'll need to obtain the latest ``Release`` and you can find the list of those at `OpenMPI downloads <https://www.open-mpi.org/software/ompi/>`_. Right now that is ``openmpi-4.1.0.tar.gz`` but you can update the next few lines accordingly if not. Next we'll navigate to PFTree and do the following

.. code-block:: none

   > cd ~/PFTree
   > wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.0.tar.gz
   > gunzip openmpi-4.1.0.tar.gz
   > tar -xvf openmpi-4.1.0.tar
   > cd openmpi-4.1.0
   > ./configure --prefix=$MPI_DIR
   > make
   > make install

That installation can take a long time but afterwards it can be verified with 

.. code-block:: none

   > ompi_info
                    Package: Open MPI stan.marsh@ubuntu Distribution
                   Open MPI: 4.1.0
     Open MPI repo revision: v4.1.0
      Open MPI release date: Dec 18, 2020
   ...

There's likely a TON of output from that command but I'm just showing the first few (top) lines. Your output should be similar but not necessarily exactly the same.

One quick note: We are NOT installing OpenMPI in ``/usr/local`` on purpose so that multiple versions can co-exist. We've had some issues with incompatible versions. If you omit the ``--prefix=$MPI_DIR`` from the ``./configure`` command then the default installation path will be used instead and you also don't need the ``MPI_DIR`` and ``MPI_RUN_DIR`` environment variables.

4.0 - Installing the Hypre library
----------------------------------

Hypre is a library of solvers that ParFlow uses and it lives on GitHub. The simplest way to get it is to clone the repository and install from there. The following commands show you how to do that

.. code-block:: none

   > cd ~/PFTree
   > git clone https://github.com/hypre-space/hypre.git --branch master --single-branch
   > cd hypre/src
   > ./configure --prefix=$HYPRE_DIR 
   > make
   > make install

If for some reason the MPI version of the compiler isn't found you can change the configure line to ``./configure --prefix=$HYPRE_DIR CC=$MPI_RUN_DIR/mpicc`` but I didn't need to on wither system I tested this installation on.

5.0 - Installing the Silo library
---------------------------------

SILO is sort of like a library that defines a kind of database that ParFlow has used for a long time. To check the latest version go here `SILO download <https://wci.llnl.gov/simulation/computer-codes/silo/>`_ and click on the Downloads tab. I'm working with version 4.10.2 but if the version changed update below accordingly.

.. code-block:: none

   > cd ~/PFTree
   > wget https://wci.llnl.gov/sites/wci/files/2021-01/silo-4.10.2.tgz
   > gunzip silo-4.10.2.tgz
   > tar -xvf silo-4.10.2.tar
   > mv silo-4.10.2 silo
   > cd silo
   > ./configure --disable-silex
   > make
   > make install

The ``mv silo-4.10.2 silo`` line is strictly optional but the key is that whatever this directory ends up being called needs to match your $SILO_DIR environment variable. That means you need to either 1) change the extracted directory name as shown here, or 2) update $SILO_DIR to match your particular version.

6.0 - Installing and testing ParFlow
------------------------------------

Installing with CMake
~~~~~~~~~~~~~~~~~~~~~

CMake uses a separate build folder to "configure" everything, which is then copied to the "install" path after it all checks out. The complete set of instructions to build, install, and test are in the block below

*Note: The cmake command below is really long so be sure to scroll over and copy the whole thing if you're clever and using a copy/paste driven strategy. Also, it IS NOT the same as the command for MacOS! Ubuntu one needs an extra flag at the end)*

.. code-block:: none

   > cd ~/PFTree
   > git clone https://github.com/parflow/parflow.git --branch master --single-branch
   > mkdir build
   > cd build
   > cmake ../parflow -DCMAKE_INSTALL_PREFIX=$PARFLOW_DIR -DHYPRE_ROOT=$HYPRE_DIR -DPARFLOW_AMPS_LAYER=mpi1 -DPARFLOW_AMPS_SEQUENTIAL_IO=true -DPARFLOW_ENABLE_TIMING=true -DSILO_ROOT=$SILO_DIR -DPARFLOW_HAVE_CLM=ON -DTCL_INCLUDE_PATH=/usr/include/tcl8.6
   > make 
   > make install

Lastly be sure to test ParFlow using the built in test script

.. code-block:: none

   > make test

If MOST of those pass, you should be all set. Most computers will see at least some of them (sometimes lots of them) fail because certain tests may use processor topologies you don't have. For example, if one of the tests requires 8 cores and you only have 6, that one is going to fail.  

Optional cleanup
~~~~~~~~~~~~~~~~

Since ParFlow is now configured in the location identified by your $PARFLOW_DIR the temporary build folder we made is no longer necessary and can be removed if you'd like. To do so, assuming you are in ``~/PFTree/build`` 

.. code-block:: none

   > cd ..
   > rm -rf build

It's not really necessary to do this but you can if you want to (definitely leave it if you plan on updating your ParFlow build). There are also some tarball archives in ``~/PFTree`` but since most of those are small I generally like to keep them so I know I have a complete set of working components.
