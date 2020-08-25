#!/bin/bash

# -----------------------------------------------------------------------------
# Installing required system libraries for Ubuntu using apt-get
#
# Caution:
#   Need to be 'root'
# -----------------------------------------------------------------------------

# Non interactive mode
export DEBIAN_FRONTEND=noninteractive

# Fetch the latest definitions of packages
apt-get update

# Solve timezone issue for 20.04
ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
apt-get install -y tzdata
dpkg-reconfigure --frontend noninteractive tzdata

# Install required pieces to build code for Parflow
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
  tk-dev
