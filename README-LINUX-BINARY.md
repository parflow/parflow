# ParFlow Pre-Built Binary (Linux x86_64)

This archive contains a self-contained ParFlow installation for **Linux x86_64**
built on the project's **self-hosted HPC runner** using cluster modules for
OpenMPI, HDF5, and NetCDF. HYPRE is built in CI; all runtime libraries are
bundled under a single `install/` prefix — no separate `deps/` tree and no
system MPI/HDF5 packages required to run ParFlow.

## Downloading the bundle

### From a GitHub Actions workflow (PR / CI testing)

Workflow artifacts are named like:

```text
parflow-pr-<number>-<sha>-linux-x86_64.tar.gz
```

GitHub **wraps artifacts in an extra ZIP** when you download them from the
Actions tab. The file on your machine will look like:

```text
parflow-pr-742-<sha>-linux-x86_64.tar.gz.zip
```

Extract in two steps:

```bash
unzip parflow-pr-<number>-<sha>-linux-x86_64.tar.gz.zip
tar -xvf parflow-pr-<number>-<sha>-linux-x86_64.tar.gz
```

This creates a `release-install/` directory (the bundled prefix).

### From a GitHub Release (when published)

Release assets are uploaded as plain `.tar.gz` files — **no extra ZIP**:

```bash
tar -xvf parflow-<version>-linux-x86_64.tar.gz
```

## Quick start

```bash
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
install/
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
- **Build host:** Self-hosted HPC runner (OpenMPI/gcc modules used for GPU CI)
- **GPU:** CPU / MPI1 only (no CUDA/Kokkos in this workflow).

## Building from source

For other Linux distributions, GPU builds, or different glibc, build from source.
See the main [README](https://github.com/parflow/parflow#readme).
