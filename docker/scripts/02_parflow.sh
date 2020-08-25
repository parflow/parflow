#!/bin/bash

# -----------------------------------------------------------------------------
# ParFlow build script for Ubuntu and macOS
# -----------------------------------------------------------------------------
# Environment variables as build options (* = default)
#
#  PARFLOW_BRANCH
#     * v3.6.0
#       master
#
#  HYPRE_BRANCH
#     * v2.19.0
#       master
#
#  CMAKE_TYPE
#     * cmake-3.17.2-Linux-x86_64.tar.gz
#       cmake-3.17.2-Darwin-x86_64.tar.gz
#
#  ROOT
#     * $PWD
#       ~
#
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Process scripts options
# -----------------------------------------------------------------------------

[ ! -z "$ROOT" ] || export ROOT=$PWD
[ ! -z "$CMAKE_TYPE" ] || export CMAKE_TYPE="cmake-3.17.2-Linux-x86_64.tar.gz"
[ ! -z "$PARFLOW_BRANCH" ] || export PARFLOW_BRANCH="v3.6.0"
[ ! -z "$HYPRE_BRANCH" ] || export HYPRE_BRANCH="v2.19.0"

# -----------------------------------------------------------------------------
# Fail on error
# -----------------------------------------------------------------------------

set -e

# -----------------------------------------------------------------------------
# Create directories
# -----------------------------------------------------------------------------

mkdir -p                         \
  $ROOT/parflow/build             \
  $ROOT/parflow/dependencies/cmake \
  $ROOT/parflow/dependencies/hypre-src

# -----------------------------------------------------------------------------
# Setup CMake 3.17
# -----------------------------------------------------------------------------

cd $ROOT/parflow/dependencies/cmake
curl -L "https://cmake.org/files/v3.17/${CMAKE_TYPE}" | tar --strip-components=1 -xzv

export CMAKE=$ROOT/parflow/dependencies/cmake/bin/cmake
export CTEST=$ROOT/parflow/dependencies/cmake/bin/ctest

# -----------------------------------------------------------------------------
# Install Hypre
# -----------------------------------------------------------------------------

export HYPRE_DIR=$ROOT/parflow/dependencies/hypre

cd $ROOT/parflow/dependencies/hypre-src
git clone https://github.com/hypre-space/hypre.git \
    --single-branch --branch $HYPRE_BRANCH

cd hypre/src
./configure --prefix=$HYPRE_DIR --with-MPI
make install

# -----------------------------------------------------------------------------
# Install Parflow
# -----------------------------------------------------------------------------

git clone                             \
  --recursive --single-branch          \
  --branch $PARFLOW_BRANCH              \
  https://github.com/parflow/parflow.git \
  $ROOT/parflow/src

$CMAKE                           \
   -S $ROOT/parflow/src           \
   -B $ROOT/parflow/build          \
   -D HYPRE_ROOT=$HYPRE_DIR         \
   -D PARFLOW_AMPS_LAYER=mpi1        \
   -D PARFLOW_AMPS_SEQUENTIAL_IO=TRUE \
   -D PARFLOW_ENABLE_TIMING=TRUE       \
   -D PARFLOW_HAVE_CLM=TRUE             \
   -D PARFLOW_ENABLE_PYTHON=TRUE         \

$CMAKE --build $ROOT/parflow/build
$CMAKE --install $ROOT/parflow/build \
       --prefix $ROOT/parflow/install

# -----------------------------------------------------------------------------
# Provide environment variables
# -----------------------------------------------------------------------------

echo export PARFLOW_DIR=$ROOT/parflow/install >> $ROOT/parflow/env.sh
