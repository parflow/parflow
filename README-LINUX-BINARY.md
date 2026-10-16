# ParFlow Pre-Built Binary (Linux x86_64)

This archive contains a self-contained ParFlow installation for **Linux x86_64**
built on the project's **self-hosted HPC runner** using cluster modules for
OpenMPI, HDF5, and NetCDF. HYPRE is built in CI; all runtime libraries are
bundled under a single install prefix — no separate `deps/` tree and no system
MPI/HDF5 packages required to run ParFlow.

Binaries are produced when a [GitHub Release](https://github.com/parflow/parflow/releases)
is **published** (see workflow `release-linux-binaries.yml`).

## Downloading the bundle

1. Open the release page for your version (e.g. tag `v3.12.0`).
2. Under **Assets**, download the Linux x86_64 tarball:

   ```text
   parflow-<tag>-linux-x86_64.tar.gz
   ```

   Example: `parflow-v3.12.0-linux-x86_64.tar.gz`

3. Extract:

   ```bash
   tar -xvf parflow-<tag>-linux-x86_64.tar.gz
   ```

   This creates a `release-install/` directory (the bundled prefix).

## Quick start

```bash
cd /path/to/release-install
source parflow-env.sh

# Verify (use full path; parflow-env.sh does not prepend PATH)
"$PARFLOW_DIR/bin/parflow" -v
```

## Python tools (pftools)

The Python package (`pftools`) is **not** included in this archive.
Install it from PyPI into your own Python environment:

```bash
pip install pftools
```

## Important: `PARFLOW_DIR`

`parflow-env.sh` sets `PARFLOW_DIR` to the extracted install directory.
Both ParFlow and the bundled OpenMPI expect this when launching parallel runs.

## What is included

```
release-install/
  bin/           parflow, mpiexec, mpirun, helper scripts
  lib/           ParFlow and bundled shared libraries
  libexec/       OpenMPI ORTE helpers (e.g. orted for singleton MPI_Init)
  config/        pf-cmake-env.sh (relocatable MPI paths)
  share/         OpenMPI / PRTE / PMIx runtime data
  etc/           OpenMPI configuration
  parflow-env.sh Environment setup script
```

## Supported platform

- **Architecture:** x86_64
- **GPU:** CPU / MPI1 only (no CUDA/Kokkos in this workflow).

## Building from source

For other Linux distributions, GPU builds, or different glibc, build from source.
See the main [README](https://github.com/parflow/parflow#readme).
