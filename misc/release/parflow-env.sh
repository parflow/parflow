#!/usr/bin/env bash
# Source this file after unpacking a ParFlow Linux binary bundle:
#   source /path/to/bundle/parflow-env.sh
#
# The bundle layout is:
#   <bundle>/parflow-env.sh
#   <bundle>/parflow/   (CMAKE_INSTALL_PREFIX)
#   <bundle>/deps/      (third-party install prefix, e.g. Hypre, Silo, NetCDF)
#
# You still need a compatible MPI/HDF5 stack on the host (typically the same
# OpenMPI + HDF5 packages used on Ubuntu 22.04). See README-LINUX-BINARY.md.

_bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PARFLOW_DIR="${_bundle_root}/parflow"
export PARFLOW_DEP_DIR="${_bundle_root}/deps"

export PATH="${PARFLOW_DIR}/bin:${PARFLOW_DEP_DIR}/bin:${PATH}"

# Install RPATH targets parflow/lib; deps live alongside; many transitive libs
# come from system OpenMPI/HDF5 (Debian/Ubuntu MPI paths).
export LD_LIBRARY_PATH="${PARFLOW_DIR}/lib:${PARFLOW_DEP_DIR}/lib:${PARFLOW_DEP_DIR}/lib64:/usr/lib/x86_64-linux-gnu/hdf5/openmpi:${LD_LIBRARY_PATH:-}"
