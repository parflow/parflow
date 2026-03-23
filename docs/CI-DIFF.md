# Bundle Workflow vs Existing CI — Exact Differences

This document shows precisely what the bundle workflow shares with
and changes from the existing `macos.yml` CI workflow, to make
code review easier.

## Identical to existing CI

| Step | Source |
|------|--------|
| `actions/checkout@v4` | same |
| `actions/setup-python@v5` (3.13) | same |
| `fortran-lang/setup-fortran@v1` (gcc) | same |
| `mpi4py/setup-mpi@v1` (openmpi) | same |
| `brew link tcl-tk@8` (macOS) | same |
| Directory layout (`~/install`, `~/depend`) | same |
| NetCDF-C 4.9.0 from source | same |
| NetCDF-Fortran 4.5.4 from source | same |
| Silo 4.11.1 from source | same |
| hypre 2.26.0 from source | same |
| ParFlow cmake flags (mpi1, CLM, timing, etc.) | same |
| `ctest --output-on-failure` validation | same |

## Changed from existing CI

| What | Existing CI (`macos.yml`) | Bundle workflow | Why |
|------|--------------------------|-----------------|-----|
| HDF5 | `brew install hdf5-mpi` | Built from source (1.14.3) | Homebrew dylibs have absolute paths; can't redistribute |
| NetCDF CPPFLAGS | `-I/opt/homebrew/include` | `-I${PARFLOW_DEP_DIR}/include` | Points at our source-built HDF5 instead of Homebrew |
| NetCDF LDFLAGS | `-L/opt/homebrew/lib/` | `-L${PARFLOW_DEP_DIR}/lib` | Same reason |
| Dependency caching | `actions/cache@v4` | None | Bundles are release artifacts; always build fresh |
| TCL paths | Hardcoded `/opt/homebrew/...` | Same Homebrew paths for build | TCL is build-time only; not shipped in bundle |
| cmake RPATH | Not set | `-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE` | Needed for relocatability |

## New steps (not in existing CI)

| Step | Purpose |
|------|---------|
| Merge dependencies into bundle | Combines `~/depend` and `~/install` into one prefix |
| Bundle MPI | Copies OpenMPI binaries + libs into prefix |
| Fix pf-cmake-env.sh | Rewrites hardcoded MPIEXEC path to `$PARFLOW_DIR/bin/mpiexec` so the `run` script finds the bundled MPI |
| Bundle Fortran Runtime | Copies libgfortran/libquadmath/libgcc_s into prefix |
| Fix rpaths (macOS) | Rewrites dylib references to @rpath for relocatability |
| Fix rpaths (Linux) | Sets RUNPATH to $ORIGIN via patchelf |
| Bundle smoke test | Validates bundle integrity before tarballing |
| Add setup.sh | User-facing environment setup script |
| Create tarball | Packages the bundle for distribution |
| Create GitHub Release | Publishes tarballs with checksums on tag push |

## Platform matrix comparison

| | Existing CI | Bundle workflow |
|---|------------|-----------------|
| macOS ARM64 (macos-14) | ✅ | ✅ |
| macOS x86_64 (macos-13) | ❌ | ✅ (new) |
| Linux (ubuntu-22.04) | ✅ (in linux.yml) | ✅ |
| Linux (ubuntu-24.04) | ✅ (in linux.yml) | ❌ (22.04 has wider glibc compat) |

Note: ubuntu-22.04 is intentional for the Linux bundle — it produces
binaries with glibc 2.35 which run on 22.04+ including WSL2. Building
on 24.04 would require glibc 2.39+ at runtime.
