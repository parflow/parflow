# ParFlow Pre-Built Binary (macOS arm64)

This archive contains a self-contained ParFlow installation for
Apple-Silicon Macs (arm64). All required shared libraries (MPI, HDF5,
NetCDF, HYPRE, TCL, etc.) are bundled — no Homebrew or other package
manager is needed to run ParFlow itself.

## Quick start

```bash
# 1. Download and extract (substitute the actual version)
curl -LO https://github.com/parflow/parflow/releases/download/vX.Y.Z/parflow-vX.Y.Z-macos-arm64.tar.gz
tar xzf parflow-vX.Y.Z-macos-arm64.tar.gz

# 2. Remove macOS quarantine flag
xattr -dr com.apple.quarantine install/

# 3. Set environment (add to your shell profile for persistence)
export PARFLOW_DIR=$PWD/install
export PATH=$PARFLOW_DIR/bin:$PATH

OR in your python script add
```python
import os
os.environ["PARFLOW_DIR"] = '/path/to/your/download/'

# 4. Verify
parflow -v
```

## Gatekeeper / quarantine

The `xattr` step above is required because the binaries are not
Apple-notarized. If you skip it, macOS may block execution or silently
kill processes.

## Python tools (pftools)

The Python package (`pftools`) is **not** included in this archive.
Install it from PyPI into your own Python environment:

```bash
pip install pftools
```

## Important: `PARFLOW_DIR`

The `PARFLOW_DIR` environment variable **must** be set and point to the
extracted `install/` directory. Both the ParFlow `run` script and the
bundled OpenMPI runtime depend on it to locate configuration files and
support data at runtime.

## What is included

```
install/
  bin/           ParFlow executables, MPI launchers, helper scripts
  lib/           ParFlow libraries and all bundled shared dependencies
  lib/openmpi/   OpenMPI MCA plugin modules
  config/        CMake/build metadata (including pf-cmake-env.sh)
  share/openmpi/ OpenMPI help text and runtime data
  etc/           OpenMPI configuration files
```

## Supported platform

- **Architecture:** Apple Silicon (arm64)
- **Minimum macOS:** 14 (Sonoma) — the version used by the CI build
  runner. Newer versions should work; older versions may or may not.

## Building from source

If you need a different configuration (GPU acceleration, OASIS coupling,
x86_64, etc.) please build from source. See the main
[README](https://github.com/parflow/parflow#readme) for instructions.
