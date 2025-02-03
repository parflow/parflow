# -----------------------------------------------------------------------------
# Dockerfile for building ParFlow image on Ubuntu
# -----------------------------------------------------------------------------
# Build options (* = default)
#
#  BASE_IMAGE
#       ubuntu:18.04
#    *  ubuntu:20.04
#
#  CMAKE_URL
#    *  https://cmake.org/files/v3.18/cmake-3.18.2-Linux-x86_64.tar.gz
#
#  HDF5_URL
#    *  https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.0/src/hdf5-1.12.0.tar.gz
#
#  NETCDF_URL
#    *  https://github.com/Unidata/netcdf-c/archive/v4.7.4.tar.gz
#
#  SILO_URL
#    *  https://wci.llnl.gov/sites/wci/files/2021-01/silo-4.10.2.tgz
#
#  HYPRE_VERSION
#    *  v2.19.0
#       master
#
# -----------------------------------------------------------------------------

ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE}

# Non interactive mode
ENV DEBIAN_FRONTEND noninteractive

# Fetch the latest definitions of packages
RUN apt-get update && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install -y \
        build-essential \
        curl \
        libcurl4 \
        git \
        vim \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        openssh-client \
        openssh-server \
        openmpi-bin \
        libopenmpi-dev \
        python3 \
        python3-pip \
        python3-venv \
        tcl-dev \
        tk-dev

RUN groupadd -g 1000 ubuntu && \
    useradd -u 1000 -g ubuntu -m -N -s /bin/bash ubuntu

USER ubuntu
WORKDIR /home/ubuntu/
ENV BASE_PATH /home/ubuntu/

RUN mkdir -p                              \
    $BASE_PATH/parflow/build               \
    $BASE_PATH/parflow/dependencies/cmake   \
    $BASE_PATH/parflow/dependencies/silo-src \
    $BASE_PATH/parflow/dependencies/hdf5-src  \
    $BASE_PATH/parflow/dependencies/hypre-src  \
    $BASE_PATH/parflow/dependencies/netcdf-src

# -----------------------------------------------------------------------------
# Setup CMake 3.18.2
# -----------------------------------------------------------------------------

ARG CMAKE_URL=https://cmake.org/files/v3.18/cmake-3.18.2-Linux-x86_64.tar.gz

RUN cd $BASE_PATH/parflow/dependencies/cmake && \
    curl -L $CMAKE_URL | tar --strip-components=1 -xzv

ENV CMAKE $BASE_PATH/parflow/dependencies/cmake/bin/cmake
ENV CTEST $BASE_PATH/parflow/dependencies/cmake/bin/ctest

# -----------------------------------------------------------------------------
# Install HDF5
# -----------------------------------------------------------------------------

ARG HDF5_URL=https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.0/src/hdf5-1.12.0.tar.gz

ENV HDF5_DIR /opt/hdf5

WORKDIR $BASE_PATH/parflow/dependencies/hdf5-src
RUN curl -L $HDF5_URL | tar --strip-components=1 -xzv && \
    CC=mpicc ./configure \
      --prefix=$HDF5_DIR \
      --enable-parallel && \
    make

USER root
RUN make install
USER ubuntu

# -----------------------------------------------------------------------------
# Install NetCDF
# -----------------------------------------------------------------------------

ARG NETCDF_URL=https://github.com/Unidata/netcdf-c/archive/v4.7.4.tar.gz

ENV NETCDF_DIR /opt/netcdf

WORKDIR $BASE_PATH/parflow/dependencies/netcdf-src
RUN curl -L $NETCDF_URL | tar --strip-components=1 -xzv && \
    CC=mpicc CPPFLAGS=-I$HDF5_DIR/include LDFLAGS=-L$HDF5_DIR/lib \
    ./configure --disable-shared --disable-dap --prefix=${NETCDF_DIR} && \
    make

USER root
RUN make install
USER ubuntu

# -----------------------------------------------------------------------------
# Install SILO
# -----------------------------------------------------------------------------

ARG SILO_URL=https://wci.llnl.gov/sites/wci/files/2021-01/silo-4.10.2.tgz

ENV SILO_DIR /opt/silo

WORKDIR $BASE_PATH/parflow/dependencies/silo-src
RUN curl -L $SILO_URL | tar --strip-components=1 -xzv && \
    ./configure  --prefix=$SILO_DIR --disable-silex --disable-hzip --disable-fpzip && \
    make

USER root
RUN make install
USER ubuntu

# -----------------------------------------------------------------------------
# Install Hypre
# -----------------------------------------------------------------------------

ARG HYPRE_VERSION=v2.19.0

ENV HYPRE_DIR /opt/hypre

WORKDIR $BASE_PATH/parflow/dependencies/hypre-src
RUN git clone https://github.com/hypre-space/hypre.git --single-branch --branch $HYPRE_VERSION

WORKDIR $BASE_PATH/parflow/dependencies/hypre-src/hypre/src
RUN ./configure --prefix=$HYPRE_DIR --with-MPI

USER root
RUN make install
USER ubuntu

# -----------------------------------------------------------------------------
# Install Parflow
# -----------------------------------------------------------------------------

COPY --chown=ubuntu:ubuntu . $BASE_PATH/parflow/src
RUN pip3 install -r $BASE_PATH/parflow/src/pftools/python/requirements_all.txt

# ARG PARFLOW_VERSION=master
# ARG PARFLOW_GIT_URL=
# RUN git clone                           \
#     --recursive --single-branch          \
#     --branch $PARFLOW_VERSION             \
#     $PARFLOW_GIT_URL                       \
#     $BASE_PATH/parflow/src

RUN $CMAKE                     \
    -S $BASE_PATH/parflow/src   \
    -B $BASE_PATH/parflow/build  \
    -D CMAKE_BUILD_TYPE=Release   \
    -D HDF5_ROOT=$HDF5_DIR         \
    -D PARFLOW_ENABLE_HDF5=TRUE     \
    -D HYPRE_ROOT=$HYPRE_DIR         \
    -D PARFLOW_ENABLE_HYPRE=TRUE      \
    -D PARFLOW_ENABLE_SILO=TRUE        \
    -D SILO_ROOT=$SILO_DIR              \
    -D PARFLOW_HAVE_CLM=TRUE             \
    -D PARFLOW_ENABLE_PYTHON=TRUE         \
    -D PARFLOW_ENABLE_TIMING=TRUE          \
    -D PARFLOW_AMPS_LAYER=mpi1              \
    -D PARFLOW_AMPS_SEQUENTIAL_IO=TRUE       \
    -D PARFLOW_ENABLE_NETCDF=TRUE             \
    -D NETCDF_DIR=$NETCDF_DIR                  \
    -D CURL_LIBRARY=/usr/lib/x86_64-linux-gnu/libcurl.so.4

RUN $CMAKE --build $BASE_PATH/parflow/build

USER root
RUN $CMAKE --install $BASE_PATH/parflow/build --prefix /opt/parflow
USER ubuntu

ENV PARFLOW_DIR /opt/parflow/
WORKDIR /opt/parflow/