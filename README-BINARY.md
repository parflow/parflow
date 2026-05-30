# ParFlow Pre-Built Binary (macOS arm64)

Apple Silicon (arm64) builds are distributed as a signed, notarized
**ParFlow.app** (recommended) or as an unsigned `install/` tarball from CI
test runs.

## Recommended: ParFlow.app (signed release)

### Install

1. Download from [GitHub Releases](https://github.com/parflow/parflow/releases):
   - `parflow-<version>-macos-arm64.dmg`, or
   - `parflow-<version>-macos-arm64.app.zip`
2. **DMG:** open the disk image and drag **ParFlow** to **Applications**.
3. **ZIP:** unzip and move **ParFlow.app** to **Applications**.

No `xattr` step is required for signed, notarized releases.

### Run from Terminal

ParFlow is a **command-line** tool. The `.app` bundle is a signed container
for the same `bin/`, `lib/`, and MPI stack as a source install.

```bash
export PARFLOW_DIR="/Applications/ParFlow.app/Contents/Resources/parflow"
export PATH="$PARFLOW_DIR/bin:$PATH"

parflow -v
```

Or invoke the launcher (sets `PARFLOW_DIR` and runs `parflow`):

```bash
/Applications/ParFlow.app/Contents/MacOS/ParFlow -v
```

### Python (`pftools`)

The Python package is **not** inside the app. Install from PyPI:

```bash
pip install pftools
```

In Python, point at the bundled install:

```python
import os
os.environ["PARFLOW_DIR"] = "/Applications/ParFlow.app/Contents/Resources/parflow"
```

### MPI

```bash
export PARFLOW_DIR="/Applications/ParFlow.app/Contents/Resources/parflow"
export PATH="$PARFLOW_DIR/bin:$PATH"
mpirun -np 4 parflow my_simulation
```

`PARFLOW_DIR` must point at `Contents/Resources/parflow` so OpenMPI/PMIx/PRTE
find their config and plugin directories.

---

## Alternative: unsigned `install/` tarball (CI artifacts)

Development workflow artifacts may ship as `parflow-*-macos-arm64.tar.gz`
with a top-level `install/` directory (not notarized).

```bash
curl -LO https://github.com/parflow/parflow/releases/download/vX.Y.Z/parflow-vX.Y.Z-macos-arm64.tar.gz
tar xzf parflow-vX.Y.Z-macos-arm64.tar.gz

# Required for unsigned builds only:
xattr -dr com.apple.quarantine install/

export PARFLOW_DIR=$PWD/install
export PATH=$PARFLOW_DIR/bin:$PATH
parflow -v
```

---

## What is inside the app

```
ParFlow.app/Contents/
  MacOS/ParFlow              small launcher → parflow
  Resources/parflow/         same layout as install/
    bin/                     parflow, mpirun, mpiexec, …
    lib/                     libraries and OpenMPI plugins
    config/                  pf-cmake-env.sh, …
    share/, etc/             OpenMPI / PMIx / PRTE data
```

## Supported platform

- **Architecture:** Apple Silicon (arm64)
- **Minimum macOS:** 14 (Sonoma) — matches the CI runner

## Building from source

For GPU, OASIS, x86_64, or custom options, build from source. See the main
[README](https://github.com/parflow/parflow#readme).
