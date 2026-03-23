# ParFlow Pre-Built Binary Bundles

Adds distributable binary tarballs to the existing ParFlow release process.

## What this adds

One new workflow file: `.github/workflows/bundle-release.yml`

That's it. No changes to the existing `macos.yml` or `linux.yml` CI workflows.

## How it relates to existing CI

The bundle workflow follows the **exact same build pattern** as the
existing `macos.yml` and `linux.yml`:

- Same compiler setup (`fortran-lang/setup-fortran`, `mpi4py/setup-mpi`)
- Same dependency versions (hypre 2.26.0, NetCDF-C 4.9.0, Silo 4.11.1)
- Same cmake flags for ParFlow
- Same `ctest` validation

**One intentional difference:** HDF5 is built from source instead of
`brew install hdf5-mpi`. The Homebrew HDF5 works fine for CI testing,
but its dylibs have absolute paths baked in that make them
non-relocatable for distribution.

After the standard build + test, the workflow adds four new steps:

1. **Merge** — copies dependency libs/bins from `~/depend` into `~/install`
   to create a single prefix
2. **Bundle MPI + Fortran runtime** — copies OpenMPI and libgfortran into the
   prefix so users don't need either installed
3. **Fix rpaths** — rewrites library references to use `@rpath` (macOS)
   or `$ORIGIN` (Linux) so the bundle works from any filesystem location
4. **Tarball + Release** — packages everything and publishes to GitHub Releases

## Usage

### For release managers

Tag a release and the workflow runs automatically:

```bash
git tag v3.15.0
git push origin v3.15.0
```

GitHub Release appears with tarballs for all three platforms.

### For testing

Use "Run workflow" on the Actions tab to build bundles from any branch
without creating a release.

### For end users

```bash
tar xzf parflow-3.15.0-macos-arm64.tar.gz
cd parflow-3.15.0-macos-arm64/
source setup.sh
python my_simulation.py
```

## Platform matrix

| Runner | Platform | Notes |
|--------|----------|-------|
| `macos-14` | macOS ARM64 | M1/M2/M3/M4 Apple Silicon |
| `macos-13` | macOS x86_64 | Intel Macs |
| `ubuntu-22.04` | Linux x86_64 | Also works in WSL2 |

## Dependency version sync

Dependency versions in `bundle-release.yml` should stay in sync with
`macos.yml` and `linux.yml`. If someone bumps hypre to 2.33.0 in
`linux.yml`, the bundle workflow should follow. Consider extracting
shared versions into a matrix or env block if this becomes a
maintenance burden.

## Known issues / future work

- **macOS code signing**: Gatekeeper will quarantine unsigned downloads.
  Users see "cannot be opened because the developer cannot be verified."
  Fix: add an Apple Developer ID signing step, or tell users to
  `xattr -d com.apple.quarantine parflow-*` after untarring.
- **Silo**: The existing CI builds Silo and includes it. We carry it
  forward in the bundle but it could be made optional.
- **PFTools Python**: The bundle includes the built ParFlow Python
  extensions but users still need to `pip install` pftools. A future
  enhancement could bundle a self-contained Python venv.
- **Conda-forge (Stage 2)**: A `meta.yaml` recipe using conda-forge's
  existing hdf5/netcdf parallel variants. The build logic here feeds
  directly into that recipe.
- **WSL2 installer (Stage 3)**: A PowerShell script to bootstrap WSL2
  + download the Linux bundle.
