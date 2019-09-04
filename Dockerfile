#RMM ParFlow test

#-----------------------------------------------------------------------------
# start by building the basic container
#-----------------------------------------------------------------------------
FROM centos:latest
MAINTAINER Steven Smith <smith84@llnl.gov>

#-----------------------------------------------------------------------------
#  Package dependencies
#-----------------------------------------------------------------------------
RUN yum  install -y  \
    build-essential \    
    g++  \
    gcc  \
    gcc-c++  \
    gcc-gfortran \
    gdc \
    git \
    hdf5-devel \
    hdf5-openmpi \
    hdf5-openmpi-devel \
    hfd5 \
    make \
    openmpi \
    openmpi-devel \
    tcl-devel \
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
# SILO && CMake
#
WORKDIR /home/parflow
RUN wget -nv --no-check-certificate http://cmake.org/files/v3.14/cmake-3.14.5-Linux-x86_64.tar.gz && \
    tar -xvf cmake-3.14.5-Linux-x86_64.tar.gz && \
    curl "https://wci.llnl.gov/content/assets/docs/simulation/computer-codes/silo/silo-4.10.2/silo-4.10.2.tar.gz" -o "silo-4.10.2.tar.gz" && \
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
   git clone -b master --single-branch https://github.com/LLNL/hypre.git hypre && \
   cd hypre/src && \
   ./configure --prefix=$PARFLOW_DIR && \
   make install && \
   cd ../.. && \
   rm -fr hypre

#-----------------------------------------------------------------------------
# Parflow configure and build
#-----------------------------------------------------------------------------

ENV PARFLOW_MPIEXEC_EXTRA_FLAGS "--mca mpi_yield_when_idle 1 --oversubscribe --allow-run-as-root"

WORKDIR /home/parflow

RUN git clone -b master --single-branch https://github.com/parflow/parflow.git parflow && \
    mkdir -p build && \
    cd build && \
    cmake ../parflow \
       -DPARFLOW_AMPS_LAYER=mpi1 \
       -DPARFLOW_AMPS_SEQUENTIAL_IO=TRUE \
       -DHYPRE_ROOT=$HYPRE_DIR \
       -DSILO_ROOT=$SILO_DIR \
       -DPARFLOW_ENABLE_TIMING=TRUE \
       -DPARFLOW_HAVE_CLM=ON \
       -DCMAKE_INSTALL_PREFIX=$PARFLOW_DIR && \
     make install && \
     cd .. && \
     rm -fr parflow build

WORKDIR /data

ENTRYPOINT ["tclsh"]
