#!/usr/bin/env bash
# Source this file after unpacking a ParFlow Linux binary bundle:
#   tar xf parflow-*-linux-x86_64.tar.gz
#   source /path/to/install/parflow-env.sh
#
# Layout:
#   install/
#     bin/ lib/ config/ share/ etc/ parflow-env.sh
#     (self-contained: ParFlow, OpenMPI, HDF5, NetCDF, HYPRE, …)

_bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PARFLOW_DIR="${_bundle_root}"
export PARFLOW_DEP_DIR="${_bundle_root}"

export PATH="${PARFLOW_DIR}/bin:${PATH}"

# RPATH on binaries should resolve most libs; this is a fallback.
export LD_LIBRARY_PATH="${PARFLOW_DIR}/lib:${LD_LIBRARY_PATH:-}"

# Relocatable OpenMPI (see also config/pf-cmake-env.sh when using bin/run)
export OPAL_PREFIX="${PARFLOW_DIR}"
export OMPI_HOME="${PARFLOW_DIR}"
export PMIX_INSTALL_PREFIX="${PARFLOW_DIR}"
export PMIX_MCA_pcompress_base_silence_warning=1
