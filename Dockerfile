#-----------------------------------------------------------------------------
# start by building the basic container
#-----------------------------------------------------------------------------
FROM ubuntu:22.04
MAINTAINER Steven Smith <smith84@llnl.gov>

# Non interactive mode
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install -y \
        build-essential \
	wget \
        curl \
        libcurl4 \
        git \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        openmpi-bin \
        libopenmpi-dev \
	libhdf5-openmpi-dev \
	libhdf5-openmpi-103 \
        python3 \
        python3-pip \
        python3-venv \
        tcl-dev \
        tk-dev \
	cmake \
	libxml2 \
	libxml2-dev

RUN mkdir -p /home/parflow

#-----------------------------------------------------------------------------
# Set environment vars
#-----------------------------------------------------------------------------
ENV PARFLOW_DIR /usr/opt/parflow
ENV PARFLOW_DEP_DIR /usr/opt/parflow
ENV PATH $PATH:$PARFLOW_DIR/bin

#-----------------------------------------------------------------------------
# Build libraries
#-----------------------------------------------------------------------------

#
# HDF5
#  New version does not work with NetCDF; use Ubunuto supplied version
#  https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.2/src/hdf5-1.12.2.tar.bz2
#
# WORKDIR /home/parflow
# RUN wget -q https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.21/src/hdf5-1.8.21.tar.gz && \ 
#     tar -xf hdf5-1.8.21.tar.gz && \
#     cd hdf5-1.8.21 && \
#     CC=mpicc ./configure \
#       --prefix=$PARFLOW_DIR \
#       --enable-parallel && \
#     make && make install && \
#     cd .. && \
#     rm -fr hdf5-1.8.21 hdf5-1.8.21.tar.gz

#
# NetCDF
# 
WORKDIR /home/parflow
run wget -q https://github.com/Unidata/netcdf-c/archive/v4.9.0.tar.gz && \ 
    tar -xf v4.9.0.tar.gz && \
    cd netcdf-c-4.9.0 && \
    CC=mpicc CPPFLAGS=-I/usr/include/hdf5/openmpi LDFLAGS=-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi ./configure --prefix=${PARFLOW_DIR} && \
    make && \
    make install && \
    cd .. && \
    rm -fr netcdf-c-4.9.0 v4.9.0.tar.gz

#
# NetCDF Fortran
#
WORKDIR /home/parflow
run wget -q https://github.com/Unidata/netcdf-fortran/archive/v4.5.4.tar.gz && \ 
    tar -xf v4.5.4.tar.gz && \
    cd netcdf-fortran-4.5.4 && \
    CC=mpicc FC=mpifort CPPFLAGS=-I${PARFLOW_DIR}/include LDFLAGS=-L${PARFLOW_DIR}/lib ./configure --prefix=${PARFLOW_DIR} && \
    make && \
    make install && \
    cd .. && \
    rm -fr netcdf-fortran-4.5.4 v4.5.4.tar.gz

#
# SILO
#
WORKDIR /home/parflow
RUN wget -q https://github.com/LLNL/Silo/archive/refs/tags/4.10.2.tar.gz && \
    tar -xf 4.10.2.tar.gz && \
    cd Silo-4.10.2 && \
    ./configure  --prefix=$PARFLOW_DIR --disable-silex --disable-hzip --disable-fpzip && \
    make install && \
    cd .. && \
    rm -fr Silo-4.10.2 4.10.2.tar.gz

#
# Hypre
#
WORKDIR /home/parflow
RUN  wget -q https://github.com/hypre-space/hypre/archive/v2.26.0.tar.gz && \
   tar xf v2.26.0.tar.gz && \
   cd hypre-2.26.0/src && \
   ./configure --prefix=$PARFLOW_DIR && \
   make install && \
   cd ../.. && \
   rm -fr hypre-2.18.2 v2.18.2.tar.gz

#-----------------------------------------------------------------------------
# Parflow configure and build
#-----------------------------------------------------------------------------

ENV PARFLOW_MPIEXEC_EXTRA_FLAGS "--mca mpi_yield_when_idle 1 --oversubscribe --allow-run-as-root"

# Disable HWLOC warnings from showing up, confusing messages, this has been fixed in later OpenMPI versions
ENV HWLOC_HIDE_ERRORS "2"

WORKDIR /home/parflow

RUN git clone -b master --single-branch https://github.com/parflow/parflow.git parflow

RUN mkdir -p build && \
    cd build && \
    cmake ../parflow \
       -DPARFLOW_AMPS_LAYER=mpi1 \
       -DPARFLOW_AMPS_SEQUENTIAL_IO=TRUE \
       -DHYPRE_ROOT=$PARFLOW_DIR \
       -DSILO_ROOT=$PARFLOW_DIR \
       -DPARFLOW_ENABLE_HDF5=TRUE \
       -DPARFLOW_ENABLE_NETCDF=TRUE \
       -DPARFLOW_ENABLE_TIMING=TRUE \
       -DPARFLOW_HAVE_CLM=TRUE \
       -DPARFLOW_ENABLE_PYTHON=TRUE \
       -DPARFLOW_PYTHON_VIRTUAL_ENV=ON \
       -DCURL_LIBRARY=/usr/lib/x86_64-linux-gnu/libcurl.so.4 \
       -DCMAKE_INSTALL_PREFIX=$PARFLOW_DIR && \
     make install && \
     cd .. && \
     rm -fr parflow build

WORKDIR /data

ENTRYPOINT ["pfrun"]

