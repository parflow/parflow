#!/usr/bin/env bash
# Source this file after unpacking a ParFlow Linux release bundle:
#   tar xf parflow-<tag>-linux-x86_64.tar.gz
#   cd release-install && source parflow-env.sh
#
# Layout:
#   release-install/
#     bin/ lib/ config/ share/ etc/ parflow-env.sh
#     (self-contained: ParFlow, OpenMPI, HDF5, NetCDF, HYPRE, …)

_bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PARFLOW_DIR="${_bundle_root}"
export PARFLOW_DEP_DIR="${_bundle_root}"

# Relocatable OpenMPI (see also config/pf-cmake-env.sh when using bin/run)
export OPAL_PREFIX="${PARFLOW_DIR}"
export OMPI_HOME="${PARFLOW_DIR}"
export PMIX_INSTALL_PREFIX="${PARFLOW_DIR}"
