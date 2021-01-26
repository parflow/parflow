# PDI tools for ParFlow

## Yaml configuration file

The yaml file `conf.yml` contains the file architecture used by PDI to manage and output ParFlow data.
It describes the `vector` data structure used by ParFlow to store all physical parameters.
The whole structure is exposed to PDI.

If one runs ParFlow with PDI, the `conf.yml` mush be copied in the working directory.
This can be done easily by addind this at the beginning of the tcl file :

```tcl
file copy -force $env(PARFLOW_DIR)/../pdi/conf.yml ./
```

## Installation

Using `cmake`, add the following flag `-DPDI_ROOT=$PDI_HOME`.

```bash
FC=mpif90 CC=mpicc CXX=mpic++ cmake ../ -DPARFLOW_AMPS_LAYER=mpi1 -DPARFLOW_HAVE_CLM=ON -DCMAKE_INSTALL_PREFIX=$PARFLOW_DIR/ -DHDF5_ROOT=$HDF5_HOME -DSILO_ROOT=$SILO_HOME -DPARFLOW_ENABLE_NETCDF=ON -DNETCDF_DIR=$NETCDF_HOME -DHYPRE_ROOT=$HYPRE_HOME -DPDI_ROOT=$PDI_HOME
```

## Solvers

### Impec solver

```tcl
pfset Solver.WritePDISubsurfData True
pfset Solver.WritePDIPressure True
pfset Solver.WritePDIVelocities True
pfset Solver.WritePDISaturation True
pfset Solver.WritePDIWells True
pfset Solver.WritePDIConcentration True
```

### Richards solver

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

### LB solver

```tcl
pfset Solver.WritePDISubsurfData True
pfset Solver.WritePDIPressure True
pfset Solver.WritePDISaturation True
pfset Solver.WritePDIWells True
pfset Solver.WritePDIConcentration True
```

## Python script to check files

```
python $PARFLOW_DIR../pdi/compare_pdi_pfb.py <file name without extension>
```
