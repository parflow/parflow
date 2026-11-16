MacOS Install Guide
===================

MacOS 26.6 (Tahoe) — Apple or Intel chip. November 5, 2026. By Nick Engdahl (nick.engdahl@wsu.edu).

*Note: These instructions have been tested on Apple and Intel chips. The differences in the installation for each chip are noted inline below. This guide uses MacOS' native ``clang`` compiler as much as possible to avoid symbolic link issues that may arise when using gcc.* 

Overview
--------

Here we're going to briefly walk through the steps needed to install ParFlow on MacOS. We'll be assuming some basic level of computational knowledge and familiarity with the command line in the interest of brevity. The steps we'll be following are:

1. Setup the required compilers
2. Install the CMake package
3. Install OpenMPI
4. Install Hypre library
5. Install Silo library
6. Install ParFlow

Commands entered at the command line are prefaced with a '> ' and everything after can be copied and pasted. Sometimes double hyphens and quotes don't copy/paste well so you may need to go back a correct those.

1.0 - Setting up the compilers
------------------------------

Enable command line tools
~~~~~~~~~~~~~~~~~~~~~~~~~

The first thing we need to do is install the necessary components so we can build programs, which aren't setup in MacOS by default. Open a terminal window and type the following

.. code-block:: none

   > xcode-select --install

(or 'sudo xcode-select --install' if necessary). Verify that the default compilers are working with

.. code-block:: none

   > clang -v
   Apple clang version 12.0.0 (clang-1200.0.32.27)
   Target: x86_64-apple-darwin20.1.0
   Thread model: posix
   InstalledDir: /Library/Developer/CommandLineTools/usr/bin

If you see that, you're on your way (though your version may be different).

Install gfortran
~~~~~~~~~~~~~~~~

There are a few ways to do this but the easiest is to use the pre-built binaries available at `HPC Sourceforge <http://hpc.sourceforge.net>`_ where you can select the version for your system. For reasons that will become clear later, we're ONLY going to get the gfortran package. This is the second link along each line, the filename starts with "gfortran" and is a compressed tarball. 

**_If you are running a Mac with an Apple chip be sure to use the download link that specifically indicates that it is for an apple chip._ The other links are for an Intel chip. If you don't know which one, see the "About this Mac" information on your computer; choosing the right one is important to maximize performance.**

For Macs with Intel chips
^^^^^^^^^^^^^^^^^^^^^^^^^

In terminal, navigate to your "Downloads" folder then type

.. code-block:: none

   >  sudo tar -xvf gfortran-10.2-bin.tar -C /.

which copies the pre-built binaries into /usr/local. If you downloaded a different version update the filename accordingly but that name is current as of December 2020. If the archive did not automatically expand you can run ``gunzip`` to expand it manually. Next, verify that the compiler is working with

.. code-block:: none

   > gfortran -v
   Using built-in specs.
   COLLECT_GCC=gfortran
   COLLECT_LTO_WRAPPER=/usr/local/libexec/gcc/x86_64-apple-darwin19/10.2.0/lto-wrapper
   Target: x86_64-apple-darwin19
   Configured with: ../gcc-10.2.0/configure --build=x86_64-apple-darwin19 --disable-nls --enable-checking=release --with-system-zlib --disable-multilib --with-sysroot=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk --enable-languages=c,c++,fortran --with-gmp=/usr/local --with-mpc=/usr/local --with-mpfr=/usr/local
   Thread model: posix
   Supported LTO compression algorithms: zlib
   gcc version 10.2.0 (GCC) 

Your version info may differ. 

For Macs with Apple chips
^^^^^^^^^^^^^^^^^^^^^^^^^

As of December 2020, the gfortran compiler tuned for Apple's ARM architecture doesn't have a version number right now and is labeled by the chip name "M1" but that might change so keep a sharp eye and adjust filenames as needed. Download it from the website, then, in terminal, navigate to your "Downloads" folder then type

.. code-block:: none

   >  sudo tar -xvf gfortran-m1-bin.tar -C /.

which copies the pre-built binaries into /usr/local. If you downloaded a different version update the filename accordingly but that name is current as of December 2020. If the archive did not automatically expand you can run ``gunzip`` to expand it manually. Next, verify that the compiler is working with

.. code-block:: none

   > gfortran -v
   Using built-in specs.
   COLLECT_GCC=gfortran
   COLLECT_LTO_WRAPPER=/usr/local/libexec/gcc/aarch64-apple-darwin20/11.0.0/lto-wrapper
   Target: aarch64-apple-darwin20
   Configured with: ../gcc-darwin-arm64-master-wip-apple-si/configure --enable-languages=c++,fortran --with-sysroot=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk --build=aarch64-apple-darwin20 --with-gmp=/usr/local --with-mpfr=/usr/local --with-mpc=/usr/local
   Thread model: posix
   Supported LTO compression algorithms: zlib
   gcc version 11.0.0 20201128 (experimental) (GCC) 

Again, your version info may differ. The important thing here is to verify the ``Target: aarch64-apple-darwin20`` which shows that this is for Apple's ARM chip.

*Note: Your macOS might quarantine the files. In case you get a popup window saying "the file cannot be opened", you can fix this by running:*

.. code-block:: none

   sudo xattr -dr com.apple.quarantine /usr/local/bin/gfortran /usr/local/libexec/gcc /usr/local/lib

2.0 - CMake setup
-----------------

CMake is used to build the ParFlow package and can be used to build some of the supporting packages. It's not included with MacOS by default but is fairly simple to build. You can find the latest build on the CMake downloads page here: `CMake downloads <https://cmake.org/download/>`_

Click on the "cmake-4.5.0-rc3.tar.gz" version to download it to "Downloads" and it should be unzipped automatically. Note that this is the source code NOT the pre-built binary. (This is the current latest build, you can also check the CMake version minimum required in the ParFlow repo).

Build a "home" for ParFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~

Since we're installing a bunch of stuff, it's often helpful not to clutter Downloads, so we'll make a folder to hold all our ParFlow things. In a terminal we'll do the following

.. code-block:: none

   > cd ~
   > mkdir PFTree
   > cd PFTree

Do all the following from within "PFTree" to get CMake setup (update version numbers as needed)

.. code-block:: none

   > cp ~/Downloads/cmake-4.5.0-rc3.tar .
   > tar -xvf cmake-4.5.0-rc3.tar
   > cd cmake-4.5.0-rc3
   > ./bootstrap 
   > make
   > sudo make install

and that does it for CMake. 

3.0 - OpenMPI installation
--------------------------

Defining necessary/helpful environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before setting up OpenMPI were going to start adding a few things to the environment variables to make upgrading easier in the future. These first steps may seem abstract but you'll see later on where some of them are used. Environment variables may be in a different place depending on the shell you're using but the default is z-shell, which is what I'll assume you're using. In your text editor of choice (vim, for example), add the following variables to your ``.zshrc`` configuration file, which lives in your home folder (i.e. ``cd ~``):

.. code-block:: none

   export MPI_DIR=~/PFTree/OMPI-4.0.5
   export MPI_RUN_DIR=~/PFTree/OMPI-4.0.5/bin
   export HYPRE_DIR=~/PFTree/hypre
   export SILO_DIR=~/PFTree/silo
   export PARFLOW_DIR=~/PFTree/parflow
   
   export PATH=$MPI_RUN_DIR:$PATH

That last line with PATH needs to be the last line in your .zshrc file. Next source it so the changes take effect

.. code-block:: none

   > source ~/.zshrc

You can verify that those are defined using echo, for example ``echo $MPI_DIR``

The variables we set here that involve MPI will be used presently and rest of those variables will be used a bit later. As we move through the installation you'll see some of those environment variables names pop up in the various configure commands; in each case you can change the path to one more to your liking, just be sure the path you create matches the defined variables or, at best, things won't be where you expect.

Download and install OpenMPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can find OpenMPI on GitHub but as of today those tend to be developer versions and I'd suggest you only use that if you plan to work directly on OpenMPI. Fortunately for the rest of us, the ``Releases`` are very stable so we'll use the most recent at `OpenMPI downloads <https://www.open-mpi.org/software/ompi/>`_ which today is ``openmpi-4.0.5.tar.gz`` so go ahead and download that into "Downloads" paying attention to the extension of the file since there are multiple versions on the OpenMPI page. Next we'll navigate to PFTree and do the following

.. code-block:: none

   > cd ~/PFTree
   > mv ~/Downloads/openmpi-4.0.5.tar . 
   > tar -xvf openmpi-4.0.5.tar
   > cd openmpi-4.0.5
   > ./configure --prefix=$MPI_DIR
   > make
   > make install

That installation can take a long time but afterwards it can be verified with 

.. code-block:: none

   > ompi_info
                    Package: Open MPI stan.marsh@Darsh-MacBook-Pro.local
                             Distribution
                   Open MPI: 4.0.5
     Open MPI repo revision: v4.0.5
      Open MPI release date: Aug 26, 2020
   ...

There's likely a TON of output from that command but I'm just showing the first few (top) lines. If you see something like that then you're ready to move on.

One quick note: We're intentionally NOT installing OpenMPI in ``/usr/local`` so that multiple versions can co-exist. We've had some issues with incompatible versions. If you omit the ``--prefix=$MPI_DIR`` from the ``./configure`` command then the default installation path will be used instead and you also don't need the ``MPI_DIR`` and ``MPI_RUN_DIR`` environment variables.

4.0 - Installing the Hypre library
----------------------------------

Hypre has been moved over to GitHub so we can just clone the repository and install that way. The following commands will get it done for you

.. code-block:: none

   > cd ~/PFTree
   > git clone https://github.com/hypre-space/hypre.git --branch master --single-branch
   > cd hypre/src
   > ./configure --prefix=$HYPRE_DIR CC=$MPI_RUN_DIR/mpicc
   > make
   > make install

The options passed to configure are important here as the latter points to a specific compiler version that is linked to OpenMPI. 

5.0 - Installing the Silo library
---------------------------------

We have to download the Silo library the old fashioned way but the latest version can be found here `SILO download <https://wci.llnl.gov/simulation/computer-codes/silo/>`_ and I'm working with version 4.10.2. Assuming that's in "Downloads" the rest of what we need is

.. code-block:: none

   > cd ~/PFTree
   > cp ~/Downloads/silo-4.10.2.tar .
   > tar -xvf silo-4.10.2.tar
   > mv silo-4.10.2 silo
   > cd silo
   > ./configure --disable-silex
   > make
   > make install

The ``mv silo-4.10.2 silo`` line is technically optional, the key is that whatever this directory ends up being called needs to match your $SILO_DIR environment variable, so either 1) change the extracted directory name as shown here, or 2) update $SILO_DIR to match your particular version.

A potential alternative
~~~~~~~~~~~~~~~~~~~~~~~

It may be possible that Silo doesn't feel like building with your default compiler, it didn't for me, but there's a fairly simple work around. If that happens to you, replace the last three sets of commands with the following

.. code-block:: none

   > ./configure --disable-silex CC=gfortran CXX=gfortran
   > make
   > make install

Now, the astute observer will have noticed something a bit odd about the inline "CC" and "CXX" definitions, which set the C and C++ compilers to be used. Yes, that IS ``gfortran`` you see there, *not gcc*. I'm cheating a bit here but it's OK. As I said, the Silo build wasn't playing nice with the default ``clang`` compiler and if we were to use CC=gcc that can cause some issues because ``gcc`` actually points to ``clang`` by default in MacOS; this is the reason we *installed only gfortran* earlier. However, the ``gfortran`` we copied is actually built with support for C and C++ programs so we can just call it and it'll switch modes when it detects that this is not fortran source code. This way we don't have to mess with updating or changing symbolic links, which is overwriting Apple's links and making it easier for strange things to happen.

6.0 - Installing and testing ParFlow
------------------------------------

Installing with CMake
~~~~~~~~~~~~~~~~~~~~~

CMake uses a separate build folder to "configure" everything, which is then copied to the "install" path after it all checks out. The complete set of instructions to build, install, and test are in the block below

*Note: The cmake command below is really long so be sure to scroll over and copy the whole thing if you're wisely adopting the copy/paste route*

.. code-block:: none

   > cd ~/PFTree
   > git clone https://github.com/parflow/parflow.git --branch master --single-branch
   > mkdir build
   > cd build
   > cmake ../parflow -DCMAKE_INSTALL_PREFIX=$PARFLOW_DIR -DHYPRE_ROOT=$HYPRE_DIR -DPARFLOW_AMPS_LAYER=mpi1 -DPARFLOW_AMPS_SEQUENTIAL_IO=true -DPARFLOW_ENABLE_TIMING=true -DSILO_ROOT=$SILO_DIR -DPARFLOW_HAVE_CLM=ON
   > make 
   > make install

Now, we're going to test ParFlow as the final step in the installation process. To do so, we'll use the built in test script

.. code-block:: none

   > make test

If MOST of those pass, you should be all set. They probably won't all pass because some might use processor topologies you don't have. For example, if one of the tests requires 8 cores and you only have 6, that one is going to fail.  

To give you a better idea, at the end of make test you'll see something like this (at the bottom):

.. code-block:: none

   97% tests passed, 7 tests failed out of 212
   
   Total Test time (real) = 326.45 sec
   
   The following tests FAILED:
   	100 - default_single.tcl_3_3_3 (Failed)
   	108 - default_richards.tcl_3_3_3 (Failed)
   	151 - clm.tcl_3_3_1 (Failed)
   	157 - clm.jac.tcl_3_3_1 (Failed)
   	163 - clm_forc_veg.tcl_3_3_1 (Failed)
   	169 - clm_varDZ.tcl_3_3_1 (Failed)
   	175 - clm_slope.tcl_3_3_1 (Failed)
   Errors while running CTest

Since this was run on an M1 chip (8 cores total; 4-high, 4-low), all the tests that require more 8 cores failed; this is completely normal behavior. If you're on a Mac with an Apple chip you can also verify that it is running natively by opening ``Activity Monitor`` while the ``make test`` is running since that takes, in this case, 326 seconds. In ``Activity Monitor`` find ``parflow`` and look in the "Architecture" column. If it says "Apple" you're running natively on the ARM chip; if it says "Intel" in that column then you're running through the "Rosetta 2" binary translator/emulator. You might see the latter if you inadvertently installed an Intel version of GCC. As of December 2020, if you open ``Activity Monitor`` and there is no "Architecture" column, news flash: you're on an Intel Mac. 

Optional cleanup
~~~~~~~~~~~~~~~~

Since ParFlow is now configured in the location identified by your $PARFLOW_DIR the temporary build folder we made is no longer necessary and can be removed if you'd like. To do so, assuming you are in ``~/PFTree/build`` 

.. code-block:: none

   > cd ..
   > rm -rf build

and it is no more, though it's not really necessary to do this. There are also some tarball archives in ``~/PFTree`` and also in ``Downloads`` but since most of those are small I generally like to keep them so I know I have a complete set of working components.
