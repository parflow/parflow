# ParFlow Release Notes
---

## IMPORTANT NOTE

```
Support for GNU Autoconf will be removed in the next release of
ParFlow.  Future releases will only support configuration using CMake.
```

## Overview of Changes

* New overland flow boundary conditions
* Flow barrier added
* Support for metadata file
* Boundary condition refactoring
* Bug fixes
* Coding style update

## User Visible Changes

## New overland flow boundary conditions

Three new boundary conditions as modules - OverlandKinematic,
OverlandDiffusive and Seepage.

OverlandKinematic is similar to the original OverlandFlow boundary
condition but uses a slightly modified flux formulation that uses the
slope magnitude and it is developed to use face centered slopes (as
opposed to grid centered) and does the upwinding internally.x.  See user
manual for additional information on the new boundary conditions.

New test cases were added exercising the new boundary conditions:
overland_slopingslab_DWE.tcl, overland_slopingslab_KWE.tcl,
overland_tiltedv_DWE.tcl, overland_tiltedV_KWE.tcl,
Overland_FlatICP.tcl

Two new options were added to the terrain following grid formulation
to be consistent with the upwinding approach used in the new overland
flow formulation these are specified with the new
TFGUpwindFormullation keys documented in the manual.

For both OverlandDiffusive and OverlandKinematic analytical jacobians
were implemented in the new modules and these were tested and can be
verified in the new test cases noted above.

### Flow barrier added

Ability to create a flow barrier capability equivalent to the
hydraulic flow barrier (HFB) or flow and transport parameters at
interfaces. The flow barriers are placed at the fluxes as scalar
multipliers between cells (at cell interfaces).

Flow barriers are set using a PFB file, see user manual for additional
information.  The flow barrier is turned off by default.

### Support for metadata file

A metadata file is written in JSON format summarizing the inputs to a
run and its output files. This file provides ParaView and other
post-processing tools a simple way to aggregate data for
visualizations and analyses.

Metadata is collected during simulation startup and updated to include
timestep information with each step the simulation takes.  It is
rewritten with each timestep so that separate processes may observe
simulation progress by watching the file for changes.

### Bug Fixes

Fixed segmentation fault when unitialized variable was referenced in
cases with processors is outside of the active domain.

## Internal Changes

### Boundary condition refactoring

The framework for boundary conditions was significantly refactored to provide a
macro system to simplify adding new boundary conditions. See
bc_pressure.h for additional documentation.

### Coding style update

The Uncrustify coding style was updated and code was reformated.

## Known Issues

