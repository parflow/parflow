#RMM ParFlow test

#-----------------------------------------------------------------------------
# start by building the basic container
#-----------------------------------------------------------------------------
FROM centos:centos7
MAINTAINER Steven Smith <smith84@llnl.gov>

#-----------------------------------------------------------------------------
#  Package dependencies
#-----------------------------------------------------------------------------
RUN yum  install -y  \
    curl \
    libcurl-devel \
    gcc  \
    gcc-c++  \
    gcc-gfortran \
    git \
    m4 \
    make \
    openmpi \
    openmpi-devel \
    tcl-devel \
    tk-devel \
    wget \
    which \
    zlib \
    zlib-devel && mkdir -p /home/parflow

#-----------------------------------------------------------------------------
# Set environment vars
#-----------------------------------------------------------------------------
ENV CMAKE_DIR /home/parflow/cmake-3.14.5-Linux-x86_64
ENV PARFLOW_DIR /usr/local
ENV PATH $PATH:/usr/lib64/openmpi/bin:$CMAKE_DIR/bin:$PARFLOW_DIR/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/lib64/openmpi/lib

#-----------------------------------------------------------------------------
# Build libraries
#-----------------------------------------------------------------------------

#
# HDF5
#
WORKDIR /home/parflow
run wget -q https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.21/src/hdf5-1.8.21.tar.gz && \ 
    tar -xf hdf5-1.8.21.tar.gz && \
    source /etc/profile.d/modules.sh && \
    module load mpi/openmpi-x86_64 && \
    cd hdf5-1.8.21 && \
    CC=mpicc ./configure \
      --prefix=$PARFLOW_DIR \
      --enable-parallel && \
    make && make install && \
    cd .. && \
    rm -fr hdf5-1.8.21 hdf5-1.8.21.tar.gz

#
# NetCDF
#
WORKDIR /home/parflow
run wget -q https://github.com/Unidata/netcdf-c/archive/v4.5.0.tar.gz && \ 
    tar -xf v4.5.0.tar.gz && \
    source /etc/profile.d/modules.sh && \
    module load mpi/openmpi-x86_64 && \
    cd netcdf-c-4.5.0 && \
    CC=mpicc CPPFLAGS=-I${PARFLOW_DIR}/include LDFLAGS=-L${PARFLOW_DIR}/lib \
    ./configure --disable-shared --prefix=${NCDIR} && \
   make && \
   make install && \
   cd .. && \
   rm -fr netcdf-c-4.5.0 v4.5.0.tar.gz

#
# SILO && CMake
#
WORKDIR /home/parflow
RUN wget -nv --no-check-certificate http://cmake.org/files/v3.14/cmake-3.14.5-Linux-x86_64.tar.gz && \
    tar -xvf cmake-3.14.5-Linux-x86_64.tar.gz && \
    curl "https://wci.llnl.gov/sites/wci/files/2021-01/silo-4.10.2.tgz" -o "silo-4.10.2.tar.gz" && \
    tar -xf silo-4.10.2.tar.gz && \
    cd silo-4.10.2 && \
    ./configure  --prefix=$PARFLOW_DIR --disable-silex --disable-hzip --disable-fpzip && \
    make install && \
    cd .. && \
    rm -fr silo-4.10.2 silo-4.10.2.tar.gz

#
# Hypre
#
WORKDIR /home/parflow
RUN source /etc/profile.d/modules.sh && \
   module load mpi/openmpi-x86_64 && \
   wget -q https://github.com/hypre-space/hypre/archive/v2.18.2.tar.gz && \
   tar xf v2.18.2.tar.gz && \
   cd hypre-2.18.2/src && \
   ./configure --prefix=$PARFLOW_DIR && \
   make install && \
   cd ../.. && \
   rm -fr hypre-2.18.2 v2.18.2.tar.gz

#-----------------------------------------------------------------------------
# Parflow configure and build
#-----------------------------------------------------------------------------

ENV PARFLOW_MPIEXEC_EXTRA_FLAGS "--mca mpi_yield_when_idle 1 --oversubscribe --allow-run-as-root"

WORKDIR /home/parflow

RUN git clone -b master --single-branch https://github.com/parflow/parflow.git parflow

RUN mkdir -p build && \
    cd build && \
    LDFLAGS="-lcurl" cmake ../parflow \
       -DPARFLOW_AMPS_LAYER=mpi1 \
       -DPARFLOW_AMPS_SEQUENTIAL_IO=TRUE \
       -DHYPRE_ROOT=$PARFLOW_DIR \
       -DSILO_ROOT=$PARFLOW_DIR \
       -DPARFLOW_ENABLE_HDF5=TRUE \
       -DPARFLOW_ENABLE_NETCDF=TRUE \
       -DPARFLOW_ENABLE_TIMING=TRUE \
       -DPARFLOW_HAVE_CLM=TRUE \
       -DCMAKE_INSTALL_PREFIX=$PARFLOW_DIR && \
     make install && \
     cd .. && \
     rm -fr parflow build

WORKDIR /data

ENTRYPOINT ["tclsh"]

