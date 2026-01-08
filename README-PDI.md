# Building Parflow with PDI Data Interface

This guide explains how to build ParFlow with support for the PDI (ParFlow Data Interface).

## Overview

PDI is a library designed to decouple high-performance simulation codes from
input/output (I/O) management. It provides a declarative API that lets codes
expose their data buffers and notify PDI of key simulation events. With a flexible
plugin system, it integrates seamlessly with existing I/O libraries such as HDF5,
NetCDF, and Python, or can also be combined within a single execution. By handling
I/O through a dedicated YAML configuration file rather than embedding it in the
simulation code, PDI enhances portability and maintainability. Its flexible plugin
API also allows users to tailor data handling to their specifc I/O needs whether
optimizing for performance, storage format, or hardware constraints.

For more details, see the [PDI GitHub repository](https://github.com/pdidev/pdi/tree/main?tab=readme-ov-file#the-pdi-distribution).

## Current PDI Features
- Per-Process Data Collection: Each process collects simulation data independently.
- HDF5 File Output: Data is saved in HDF5 files, with filenames that include the corresponding process rank.

### Future Enhancements:

In-situ Data Processing: Future iterations will focus on in-situ data processing and analysis, reducing the need for disk I/O operations.

## Obtaining and installing PDI

Follow these steps to download and install PDI:

### Bash Instructions

```shell
# Download the PDI release (example: version 1.8.1)
wget https://github.com/pdidev/pdi/archive/refs/tags/1.8.1.tar.gz
tar -xzf 1.8.1.tar.gz

# Prepare the build directory
rm -rf pdi-1.8.1/build
mkdir pdi-1.8.1/build
cd pdi-1.8.1/build

# Set the installation path and compile
export PDI_INSTALL=../install
rm -rf ${PDI_INSTALL}
mkdir ${PDI_INSTALL}
cmake .. -DCMAKE_INSTALL_PREFIX=${PDI_INSTALL}
make -j 8 install
```

## Building ParFlow with PDI

To build ParFlow with PDI support, HDF5 must be enabled. Specify the PDI installation path using the `PDI_ROOT` CMake option:

```shell
cmake ../parflow -DCMAKE_INSTALL_PREFIX=${PARFLOW_DIR} \
    -DPARFLOW_AMPS_LAYER=mpi1 \
    -DPARFLOW_ENABLE_HDF5=TRUE \
    -DPARFLOW_ENABLE_TIMING=TRUE \
    -DCMAKE_BUILD_TYPE=Release \
    -DPDI_ROOT=$PDI_INSTALL
```

## Running ParFlow with PDI

PDI is configured via a YAML file (`conf.yml`), which defines the data specifications and plugins that manage ParFlow outputs. This file is located in:

```swift
/parflow/pfsimulator/third_party/pdi/conf.yml
```

It describes the vector data structure that ParFlow uses to store physical parameters such as pressure, saturation, etc. When running ParFlow with PDI, ensure that `conf.yml` is copied to your working directory.

### Copying the conf.yml Configuration File

To use the required conf.yml file in your working directory, you can copy it using one of the following methods depending on your scripting environment:

In a TCL script:

```tcl
file copy -force /parflow/pfsimulator/third_party/pdi/conf.yml ./
```

In a Python script:

```python
import shutil

# Source and destination paths
src = '/parflow/pfsimulator/third_party/pdi/conf.yml'
dst = './conf.yml'

# Copy file with overwrite
shutil.copy2(src, dst)
```

In a Bash script:

```shell
scp /parflow/pfsimulator/third_party/pdi/conf.yml .
```
## Usage: Enabling PDI Output in Solver Configuration

To expose ParFlow outputs through PDI, enable the following options in your solver configuration.

### Configurations for Impec Solver
#### Python Example:
```python
{run_name}.Solver.WritePDISubsurfData = True
{run_name}.Solver.WritePDIPressure = True
{run_name}.Solver.WritePDIVelocities = True
{run_name}.Solver.WritePDISaturation = True
{run_name}.Solver.WritePDIWells = True
{run_name}.Solver.WritePDIConcentration = True
```
#### TCL Example:
```tcl
pfset Solver.WritePDISubsurfData True
pfset Solver.WritePDIPressure True
pfset Solver.WritePDIVelocities True
pfset Solver.WritePDISaturation True
pfset Solver.WritePDIWells True
pfset Solver.WritePDIConcentration True
```

### Configurations for Richards solver
#### Python Example:
```python
{run_name}.Solver.WritePDISubsurfData = True
{run_name}.Solver.WritePDIMannings = True
{run_name}.Solver.WritePDISlopes = True
{run_name}.Solver.WritePDIPressure = True
{run_name}.Solver.WritePDISpecificStorage = True
{run_name}.Solver.WritePDIVelocities = True
{run_name}.Solver.WritePDISaturation = True
{run_name}.Solver.WritePDIMask = True
{run_name}.Solver.WritePDIDZMultiplier = True
{run_name}.Solver.WritePDIEvapTransSum = True
{run_name}.Solver.WritePDIEvapTrans = True
{run_name}.Solver.WritePDIOverlandSum = True
{run_name}.Solver.WritePDIOverlandBCFlux = True
```
#### TCL Example:
```tcl
pfset Solver.WritePDISubsurfData True
pfset Solver.WritePDIMannings True
pfset Solver.WritePDISlopes True
pfset Solver.WritePDIPressure True
pfset Solver.WritePDISpecificStorage True
pfset Solver.WritePDIVelocities True
pfset Solver.WritePDISaturation True
pfset Solver.WritePDIMask True
pfset Solver.WritePDIDZMultiplier True
pfset Solver.WritePDIEvapTransSum True
pfset Solver.WritePDIEvapTrans True
pfset Solver.WritePDIOverlandSum True
pfset Solver.WritePDIOverlandBCFlux True
```

### Configurations for LB solver
#### Python Example:
```python
{run_name}.Solver.WritePDISubsurfData = True
{run_name}.Solver.WritePDIPressure = True
{run_name}.Solver.WritePDISaturation = True
{run_name}.Solver.WritePDIWells = True
{run_name}.Solver.WritePDIConcentration = True
```
### TCL Example:
```tcl
pfset Solver.WritePDISubsurfData True
pfset Solver.WritePDIPressure True
pfset Solver.WritePDISaturation True
pfset Solver.WritePDIWells True
pfset Solver.WritePDIConcentration True
```

## Testing PDI Outputs 

The `compare_pdi_pfb.py` script in `/parflow/pftools/` can be used to verify
outputs generated by PDI. It compares PDI (HDF5) files with corresponding PFB
binary files for a given base name.  

The script processes all matching HDF5 files, extracts relevant data, and validates
it against the corresponding PFB file. This includes analyzing subvectors and
applying optional tolerance thresholds to identify discrepancies.

For example, to verify pressure data outputs for `run_name`, use:

```shell
python /parflow/pftools/python/parflow/cli/compare_pdi_pfb.py run_name.press
```

## Further Documentation for Developers

For more information about the structure of `conf.yml `and the internal implementation
of PDI, see the [PDI README.md](/https://github.com/parflow/parflow/tree/master/pfsimulator/third_party/pdi/README.md in the `/parflow/pfsimulator/third_party/pdi/` directory.