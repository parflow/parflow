# -----------------------------------------------------------------------------
# Dockerfile for building ParFlow image on Ubuntu
# -----------------------------------------------------------------------------
# Build options (* = default)
#
#  BASE_IMAGE
#    *  ubuntu:18.04
#       ubuntu:20.04
#
#  PARFLOW_VERSION
#    *  v3.6.0
#       master
#
#  HYPRE_VERSION
#    *  v2.19.0
#       master
#
# -----------------------------------------------------------------------------
ARG BASE_IMAGE=ubuntu:18.04
ARG PARFLOW_VERSION=v3.6.0
ARG PARFLOW_GIT_URL=https://github.com/parflow/parflow.git
ARG HYPRE_VERSION=v2.19.0
# -----------------------------------------------------------------------------
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
        tcl-dev \
        tk-dev && \
    pip3 install pyyaml

RUN groupadd -g 1000 ubuntu && \
    useradd -u 1000 -g ubuntu -m -N -s /bin/bash ubuntu

USER ubuntu
WORKDIR /home/ubuntu/
ENV BASE_PATH /home/ubuntu/

RUN mkdir -p                            \
    $BASE_PATH/parflow/build             \
    $BASE_PATH/parflow/dependencies/cmake \
    $BASE_PATH/parflow/dependencies/hypre-src

# -----------------------------------------------------------------------------
# Setup CMake 3.17
# -----------------------------------------------------------------------------

RUN cd $BASE_PATH/parflow/dependencies/cmake && \
    curl -L "https://cmake.org/files/v3.18/cmake-3.18.2-Linux-x86_64.tar.gz" | tar --strip-components=1 -xzv

ENV CMAKE $BASE_PATH/parflow/dependencies/cmake/bin/cmake
ENV CTEST $BASE_PATH/parflow/dependencies/cmake/bin/ctest

# -----------------------------------------------------------------------------
# Install Hypre
# -----------------------------------------------------------------------------

ENV HYPRE_DIR $BASE_PATH/parflow/dependencies/hypre

RUN cd $BASE_PATH/parflow/dependencies/hypre-src && \
    git clone https://github.com/hypre-space/hypre.git \
    --single-branch --branch $HYPRE_VERSION

RUN cd $BASE_PATH/parflow/dependencies/hypre-src/hypre/src && \
    ./configure --prefix=$HYPRE_DIR --with-MPI && \
    make install

# -----------------------------------------------------------------------------
# Install Parflow
# -----------------------------------------------------------------------------

RUN git clone                           \
    --recursive --single-branch          \
    --branch $PARFLOW_VERSION             \
    $PARFLOW_GIT_URL                       \
    $BASE_PATH/parflow/src

RUN $CMAKE                        \
    -S $BASE_PATH/parflow/src      \
    -B $BASE_PATH/parflow/build     \
    -D HYPRE_ROOT=$HYPRE_DIR         \
    -D PARFLOW_AMPS_LAYER=mpi1        \
    -D PARFLOW_AMPS_SEQUENTIAL_IO=TRUE \
    -D PARFLOW_ENABLE_TIMING=TRUE       \
    -D PARFLOW_HAVE_CLM=TRUE             \
    -D PARFLOW_ENABLE_PYTHON=TRUE

RUN $CMAKE --build $BASE_PATH/parflow/build && \
    $CMAKE --install $BASE_PATH/parflow/build --prefix $BASE_PATH/parflow/install

ENV PARFLOW_DIR $BASE_PATH/parflow/install
