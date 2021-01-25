# ParFlow Release Notes 3.7.0
---

ParFlow improvements and bug-fixes would not be possible without
contributions of the ParFlow community.  Thank you for all the great
work.

## Overview of Changes

* Autoconf support has been removed.
* Support for on-node parallelism using OpenMP and CUDA
* New overland flow formulations
* Utility for writing PFB file from R
* Additional solid file utilities in TCL
* NetCDF and HDF5 added to Docker instance

## User Visible Changes

### Autoconf support has been removed

The GNU Autoconf (e.g. configure) support has been dropped.  Use CMake
to build ParFlow.  See the README.md file for information on building
with CMake.

### Support for on-node parallelism using OpenMP and CUDA

ParFlow now has an option so support on-node parallelism in addition to
using MPI.   OpenMP and CUDA backends are currently supported.

See the README-CUDA.md and README-OPENMP.md files for information on
how to compile with support for CUDA and OpenMP.

Big thank you goes to Michael Burke and Jaro Hokkanen and the teams at
Boise State, U of Arizona, and FZ-Juelich for their hard work on
adding OpenMP and CUDA support.

### CMake dependency on version 3.14

ParFlow now requires CMake version 3.14 or better.

###  New overland flow formulations

Overland flow saw significant work:

* OverlandKinematic and OverlandDiffusive BCs per LEC
* Add OverlandKinematic as a Module
* Adding new diffusive module
* Added TFG slope upwind options to Richards Jacobian
* Added overland eval diffusive module to new OverlandDiffusive BC condition
* Adding Jacobian terms for diffusive module
* Updating OverlandDiffusive boundary condition Jacobian
* Updated documentation for new boundary conditions

### Utility for writing PFB file from R

A function to take array inputs and write them as pfbs.  See the file:
pftools/prepostproc/PFB-WriteFcn.R

### Additional solid file utilities in TCL

A new PF tools for creating solid files with irregular top and bottom
surfaces and conversion utilities to/from ascii or binary solid
files. See user-manual documentation on pfpatchysolid and
pfsolidfmtconver for information on the new TCL commands.

### NetCDF and HDF5 added to Docker instance

The ParFlow Docker instance now includes support for NetCDF and HDF5.

## Bug Fixes

### Fixed compilation issue with NetCDF

CMake support for NetCDF compilation has been improved.

### Memory leaks

Several memory leaks were addressed in ParFlow and PFTools.

### Parallel issue with overland flow boundary conditions

Fixed bug in nl_function_eval.c that caused MPI error for some
overland BCs with processors outside computational grid.

### pfdist/undist issues
 
Fixed pfdist/undist issues when using the sequential I/O model.

## Internal Changes

### Boundary condition refactoring

The loops for boundary conditions were refactored to provide a higher
level of abstraction and be more self-documenting (removed magic
numbers).  ForPatchCellsPerFace is a new macro for looping over patch
faces.  See nl_function_eval.c for example usage and problem_bc.h for
documentation on the new macros.

### PVCopy extended to include boundary cells.

PVCopy now includes boundary cells in the copy.

### DockerHub Test

A simple automated test of generated DockerHub instances was added.

### Etrace support was added

Support for generating function call traces with Etrace was added.  Add
-DPARFLOW_ENABLE_TRACE to CMake configure line.

See https://github.com/elcritch/etrace for additional information.

### Compiler warnings treated as errors

Our development process now requires code compile cleanly with the
-Wall option on GCC.  Code submissions will not be accepted that do
not cleanly compile. 

## Known Issues

See https://github.com/parflow/parflow/issues for current bug/issue reports.

