#!/bin/bash
# This script can be seen as a reference on how to install parflow and its dependencies.
# It is recommended to not start it but rather take it as guideline which option to
# use for which dependency to get a running setup.

# In the following we define some install options:
export PREFIX=/home/hectorb/PARFLOW/SOURCES/PFVR
export PYTHON="python2.7"

export MPICC=`which mpicc`
export MPICXX=`which mpicxx`
export MPIF90=`which mpif90`
export WGET="wget -c --no-check-certificate"
export SRC=$PREFIX/src



# The following environment variables are needed during runtime
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PREFIX/lib64:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=$PREFIX/lib64/pkgconfig:$PKG_CONFIG_PATH
export PARFLOW_DIR=$PREFIX

export PYTHONPATH=$PREFIX/python:$PYTHONPATH

# When running with FlowVR this is needed too during runtime:
# Attention: The following line will fail if FlowVR is not yet installed.
source flowvr-suite-config.sh



### Installing all the dependencies and parflow

# HINT: check if all the svn co commands run without user interaction before run this
# script ;)

mkdir -p $SRC

#openmpi is already installed?
cd $SRC
if [ ! -d "openmpi" ]; then
  $WGET https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
  tar xvf openmpi-3.0.0.tar.gz
  mv  openmpi-3.0.0 openmpi
  cd openmpi
  ./configure --prefix=$PREFIX
  make -j$N all install
fi

export MPICC=`which mpicc`
export MPICXX=`which mpicxx`
export MPIF90=`which mpif90`

mkdir -p $PREFIX/bin
cd $PREFIX
$WGET https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i
cd $PREFIX/bin
#ln -s `which $PYTHON` python

#$WGET http://www.python.org/ftp/python/2.7.12/Python-2.7.12.tgz
#tar xzvf Python-2.7.12.tgz
#cd Python-2.7.12
#./configure --prefix=$PREFIX --enable-shared
#make
#make install -j$N

# numpy and netCDF4 are already installed?
cd $PREFIX
if [ ! -d "python" ]; then
  mkdir python
  cd python
  $WGET https://bootstrap.pypa.io/get-pip.py
  $PYTHON get-pip.py -t .
  # to install netCDF4 only:
  $PYTHON pip install netCDF4 -t .
  # to install netCDF4 and numpy uncomment the following line:
  #$PYTHON pip install numpy netCDF4 -t .
fi


# install a recent cmake version
cd $SRC
if [ ! -d "cmake" ]; then
$WGET https://cmake.org/files/v3.10/cmake-3.10.0-rc3.tar.gz
tar xvf cmake-3.10.0-rc3.tar.gz
mv cmake-3.10.0-rc3 cmake
cd cmake
./bootstrap --prefix=$PREFIX --parallel=$N
make -j$N && make install -j$N
fi

cd $SRC
if [ ! -d "hypre" ]; then
git clone https://github.com/LLNL/hypre --depth 1
cd hypre/src
./configure --prefix=$PREFIX
make -j$N && make install -j$N
fi


cd $SRC
if [ ! -d "silo" ]; then
$WGET https://wci.llnl.gov/content/assets/docs/simulation/computer-codes/silo/silo-4.10.2/silo-4.10.2-bsd-smalltest.tar.gz
tar -xvf silo-4.10.2-bsd-smalltest.tar.gz
mv silo-4.10.2-bsd silo
cd silo
./configure --enable-shared --prefix=$PREFIX
make -j$N && make install -j$N
fi


cd $SRC
if [ ! -d "hdf5" ]; then
$WGET https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.19/src/hdf5-1.8.19.tar.gz
tar xvf hdf5-1.8.19.tar.gz
mv hdf5-1.8.19 hdf5
cd hdf5
CC=$MPICC ./configure --enable-parallel --prefix=$PREFIX
make -j$N && make install -j$N
fi


cd $SRC
if [ ! -d "pnetcdf" ]; then
# we are not using the git version here as the bootstrap does not work on froggy.
$WGET http://cucis.ece.northwestern.edu/projects/PnetCDF/Release/parallel-netcdf-1.8.1.tar.gz
tar xvf parallel-netcdf-1.8.1.tar.gz
mv parallel-netcdf-1.8.1 pnetcdf
cd pnetcdf/
./configure --prefix=$PREFIX --enable-parallel
make -j$N && make install -j$N
fi

cd $SRC
if [ ! -d "netcdf-c" ]; then
#$WGET ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.5.0.tar.gz
#tar -xvf v4.5.0
#mv netcdf-c-4.5.0 netcdf-c
git clone https://github.com/Unidata/netcdf-c --depth 1
cd netcdf-c/
#CC=$MPICC CPPFLAGS=-I${PREFIX}/include LDFLAGS=-L${PREFIX}/lib --disable-shared --enable-parallel-tests --prefix=${PREFIX}

mkdir build
cd build/
cmake .. -DBUILD_SHARED_LIBS:BOOL=OFF \
  -DBUILD_TESTING:BOOL=ON \
  -DBUILD_TESTSETS:BOOL=ON \
  -DBUILD_UTILITIES:BOOL=ON \
  -DCMAKE_BUILD_TYPE:STRING=RELEASE \
  -DCMAKE_C_COMPILER:FILEPATH=$MPICC \
  -DCMAKE_INSTALL_PREFIX:PATH=$PREFIX \
  -DENABLE_CDF5:BOOL=ON \
  -DENABLE_CONVERSION_WARNINGS:BOOL=ON \
  -DENABLE_COVERAGE_TESTS:BOOL=OFF \
  -DENABLE_DAP:BOOL=ON \
  -DENABLE_DAP2:BOOL=ON \
  -DENABLE_DAP4:BOOL=ON \
  -DENABLE_DAP_GROUPS:BOOL=ON \
  -DENABLE_DAP_LONG_TESTS:BOOL=OFF \
  -DENABLE_DAP_REMOTE_TESTS:BOOL=ON \
  -DENABLE_DISKLESS:BOOL=ON \
  -DENABLE_DOXYGEN:BOOL=OFF \
  -DENABLE_DYNAMIC_LOADING:BOOL=ON \
  -DENABLE_ERANGE_FILL:BOOL=OFF \
  -DENABLE_EXAMPLES:BOOL=ON \
  -DENABLE_EXAMPLE_TESTS:BOOL=OFF \
  -DENABLE_EXTRA_TESTS:BOOL=OFF \
  -DENABLE_EXTREME_NUMBERS:BOOL=ON \
  -DENABLE_FAILING_TESTS:BOOL=OFF \
  -DENABLE_FFIO:BOOL=OFF \
  -DENABLE_FSYNC:BOOL=OFF \
  -DENABLE_HDF4:BOOL=OFF \
  -DENABLE_JNA:BOOL=OFF \
  -DENABLE_LARGE_FILE_SUPPORT:BOOL=ON \
  -DENABLE_LARGE_FILE_TESTS:BOOL=OFF \
  -DENABLE_LOGGING:BOOL=OFF \
  -DENABLE_MMAP:BOOL=ON \
  -DENABLE_NETCDF4:BOOL=ON \
  -DENABLE_NETCDF_4:BOOL=ON \
  -DENABLE_PARALLEL4:BOOL=ON \
  -DENABLE_PARALLEL_TESTS:BOOL=ON \
  -DENABLE_PNETCDF:BOOL=ON \
  -DENABLE_REMOTE_FORTRAN_BOOTSTRAP:BOOL=OFF \
  -DENABLE_RPC:BOOL=OFF \
  -DENABLE_SHARED_LIBRARY_VERSION:BOOL=ON \
  -DENABLE_STDIO:BOOL=OFF \
  -DENABLE_TESTS:BOOL=ON \
  -DENABLE_V2_API:BOOL=ON \
  -DENABLE_ZERO_LENGTH_COORD_BOUND:BOOL=OFF \
  -DPNETCDF:FILEPATH=$PREFIX/lib/libpnetcdf.a \
  -DPNETCDF_INCLUDE_DIR:PATH=$PREFIX/include \
  -DTEST_PARALLEL:BOOL=ON \
  -DTEST_PARALLEL4:BOOL=ON \
  -DUSE_CDF5:BOOL=ON \
  -DUSE_DAP:BOOL=ON \
  -DUSE_HDF5:BOOL=ON \
  -DUSE_LIBDL:BOOL=ON \
  -DUSE_NETCDF4:BOOL=ON \
  -DUSE_PARALLEL:BOOL=ON \
  -DUSE_PNETCDF:BOOL=ON


  make -j$N && make install -j$N
fi

cd $SRC
if [ ! -d "pcre" ]; then
  svn co svn://vcs.exim.org/pcre/code/trunk pcre
  cd pcre
  ./autogen.sh
  ./configure --prefix=$PREFIX
  make -j$N && make install -j$N
fi

cd $SRC
if [ ! -d "swig" ]; then
  $WGET https://downloads.sourceforge.net/swig/swig-3.0.12.tar.gz
  tar xvf swig-3.0.12.tar.gz
  mv swig-3.0.12 swig
  cd swig
  ./configure --prefix=$PREFIX
  ./autogen.sh
  make -j$N && make install -j$N
fi

cd $SRC
if [ ! -d "flowvr" ]; then
  # To get flowvr you need an account at gitlab.inria.fr with access to this project:
  # https://gitlab.inria.fr/flowvr/flowvr-ex . In the next days this will open sourced
  # making account creation obsolete.
  # Furthermore the dev branch containing needed changes to run with parFlowVR will be
  # pulled into master. At the moment we recommend the version
  # downloaded by the commands below:
  git clone https://gitlab.inria.fr/flowvr/flowvr-ex.git
  mv flowvr-ex flowvr
  cd flowvr
  git checkout dev
  git checkout 628fd3b7348c3fb4e282360f90da1e2636b9e42a
  mkdir build
  cd build
  cmake .. -DBUILD_CONTRIB:BOOL=ON \
    -DCMAKE_C_COMPILER:FILEPATH=$MPICC \
    -DCMAKE_CXX_COMPILER:FILEPATH=$MPICXX \
    -DCMAKE_INSTALL_PREFIX:PATH=$PREFIX \
    -DBUILD_CONTRIB_CEGUI_MENU_MOD:BOOL=OFF \
    -DBUILD_CONTRIB_FLOWVR_IMMERSIVE:BOOL=OFF \
    -DBUILD_CONTRIB_FLOWVR_TESTS:BOOL=ON \
    -DBUILD_CONTRIB_FLOWVR_VRPN:BOOL=OFF \
    -DBUILD_CONTRIB_JOYPAD_MOD:BOOL=ON \
    -DBUILD_CONTRIB_SETTINGFREQ_MOD:BOOL=OFF \
    -DBUILD_FLOWVRD_MPI_PLUGIN:BOOL=ON \
    -DBUILD_FLOWVR_CONTRIB_AFFINITY_TEST:BOOL=OFF \
    -DBUILD_FLOWVR_CONTRIB_BUFFER_TEST:BOOL=OFF \
    -DBUILD_FLOWVR_CONTRIB_FINITEQUEUE_TEST:BOOL=OFF \
    -DBUILD_FLOWVR_CONTRIB_SEGMENTS_TEST:BOOL=OFF \
    -DBUILD_FLOWVR_CONTRIB_SHAREDMEMORY_TEST:BOOL=OFF \
    -DBUILD_FLOWVR_CONTRIB_TCPTEST_TEST:BOOL=OFF \
    -DBUILD_FLOWVR_DEBUG_MESSAGES:BOOL=ON \
    -DBUILD_FLOWVR_GLGRAPH:BOOL=FALSE \
    -DBUILD_FLOWVR_GLTRACE:BOOL=ON \
    -DBUILD_FLOWVR_PYTHONMODULEAPI:BOOL=ON \
    -DBUILD_TESTING:BOOL=ON

  make -j$N && make install -j$N
fi
#
#
cd $SRC
if [ ! -d "parflow" ]; then
git clone https://github.com/xy124/parflow
cd $SRC/parflow/
git checkout parFlowVR
mkdir build
cd build
cmake .. -DBUILD_TESTING:BOOL=ON \
  -DCMAKE_C_COMPILER:FILEPATH=$MPICC \
  -DCMAKE_CXX_COMPILER:FILEPATH=$MPICXX \
  -DCMAKE_Fortran_COMPILER:FILEPATH=$MPIF90 \
  -DCMAKE_INSTALL_PREFIX:PATH=$PREFIX \
  -DHYPRE_INCLUDE_DIR:PATH=$PREFIX/include \
  -DHYPRE_LIBRARY_LS:FILEPATH=$PREFIX/lib/libHYPRE.a \
  -DHYPRE_LIBRARY_MV:FILEPATH=$PREFIX/lib/libHYPRE.a \
  -DPARFLOW_AMPS_LAYER:STRING=mpi1 \
  -DPARFLOW_AMPS_SEQUENTIAL_IO:BOOL=OFF \
  -DPARFLOW_ENABLE_HDF5:BOOL=ON \
  -DPARFLOW_ENABLE_HYPRE:BOOL=ON \
  -DPARFLOW_ENABLE_NETCDF:BOOL=ON \
  -DPARFLOW_ENABLE_PROFILING:BOOL=False \
  -DPARFLOW_ENABLE_SILO:BOOL=ON \
  -DPARFLOW_ENABLE_SIMULATOR:BOOL=True \
  -DPARFLOW_ENABLE_SLURM:BOOL=False \
  -DPARFLOW_ENABLE_SUNDIALS:BOOL=False \
  -DPARFLOW_ENABLE_SZLIB:BOOL=False \
  -DPARFLOW_ENABLE_TIMING:BOOL=True \
  -DPARFLOW_ENABLE_TOOLS:BOOL=True \
  -DPARFLOW_ENABLE_ZLIB:BOOL=False \
  -DPARFLOW_HAVE_CLM:BOOL=ON \
  -DPARFLOW_HAVE_OAS3:BOOL=OFF \
  -DnetCDF_DIR:PATH=$PREFIX/lib/cmake/netCDF \
  -DCMAKE_C_FLAGS:STRING=-std=gnu99 \
  -DFLOWVR_PREFIX:PATH=$PREFIX \
  -DPARFLOW_ENABLE_FLOWVR:BOOL=ON \
  -DPARFLOW_ENABLE_FLOWVR_TOOLS:BOOL=True \
  -DNUMPY_I_PATH:PATH=$PREFIX

make -j$N && make install -j$N
fi

echo do not forget to add the upper section of this script to your setenv.sh script / your .bashrc
echo ------END!-------
