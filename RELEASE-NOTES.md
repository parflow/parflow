# ParFlow Release Notes 3.15.0
------------------------------

This release contains several significant feature additions, bug fixes, and infrastructure improvements. Notable enhancements include expanded CLM snow physics options, improved overland flow output capabilities, KINSOL solver flexibility, and enhanced CI/CD infrastructure.

ParFlow development and bug-fixes would not be possible without contributions of the ParFlow community. Thank you for all the great contributions.

***

# VERY IMPORTANT: TCL support and ParFlow 4.x

# ParFlow 4.x

This will likely be the last release of ParFlow 3.x.

We will be rolling out ParFlow 4.0, and we will likely break some compatibility in order to address long-standing ParFlow issues. One change we plan to make is to deprecate TCL support. ParFlow releases in January 2027 and beyond may or may not include TCL 8.x support depending on support costs, and we have no plans to update ParFlow to support TCL 9.x.

## TCL 9.x does not work with ParFlow

TCL 9.x has been released and is not backward compatible with TCL 8.x. ParFlow does not fully work with TCL 9.x. Please use a TCL 8.x release, or use the Python pftools API instead.

If you have TCL 9.x installed, only the ParFlow development branch currently compiles with it. We have patched ParFlow to compile with TCL 9.x, but not all custom TCL commands work. If you only have TCL 9.x, you should use the Python interface only and can compile without TCL.

To check your TCL version:

```
tclsh
info patchlevel
Configuring CMake to use a TCL 8.x installation
If you have a TCL 8.x install on your system, you can point CMake to it, for example:
TCL_DIR=<path to where TCL is installed>
cmake \
  -DTCL_TCLSH=${TCL_DIR}/bin/tclsh8.6 \
  -DTCL_LIBRARY=${TCL_DIR}/lib/libtcl8.6.so \
  <your other cmake options>
```

## Building without TCL

TCL has been a required dependency of all previous versions of ParFlow but you can now disable TCL if you are only using Python.   Add the "-DPARFLOW_ENABLE_TCL=NO" flag to your CMake configuration to disable TCL.

## Why are we deprecating TCL?

TCL 9.x is incompatible with the TCL 8.x versions we have been using. Supporting TCL 9.x will take time. Given the small size of the ParFlow developer community, supporting both Python and TCL reduces the time available for other support and development activities.  Most scripting interface development is occurring on the Python side, and we do not have resources to port everything to both APIs. The developer community has therefore decided to deprecate TCL.

## What should you do if you are using TCL?
In the short term, the key point is: ParFlow does not work with TCL 9.x, so use a TCL 8.x release.
For new problem setups, consider using the Python interface, and plan to update existing TCL workflows over time.

## Is there help to convert TCL scripts to Python?

We have automated scripts that can convert substational parts of a ParFlow TCL input script to Python. Conversion of pfset commands is automated, but additional TCL logic (moving files, simple calculations, etc.) will require manual updates.  AI tools can also help with code conversion. Asking how to convert specific TCL statements to Python often produces reasonable results.

The ParFlow documentation includes a short tutorial on TCL to Python conversion, see [From TCL to Python](https://parflow.readthedocs.io/en/latest/python/tutorials/tcl2py.html).

If you have concerns about this change feel free to post to [ParFlow User Group](https://groups.google.com/g/parflow).   We will try to help with this conversion as much as we can.

***

## Overview of Changes

Version 3.15.0 introduces major enhancements to CLM snow parameterization with multiple rain-snow partitioning schemes, improved output options for overland flow dynamics, and more flexible KINSOL solver configuration.

## User Visible Changes

### CLM Snow Parameterization Enhancements

Comprehensive updates to CLM snow physics have been added to improve snow modeling in diverse climates:

#### Rain-Snow Partitioning Methods
Multiple rain-snow partitioning methods are now available as alternatives to the standard air temperature threshold:
- **WetbulbThreshold** and **WetbulbLinear**: Account for evaporative cooling of falling hydrometeors, particularly beneficial for dry mountain climates
- **Dai (2008)**: Sigmoidal temperature-dependent function
- **Jennings (2018)**: Bivariate logistic regression using temperature and relative humidity

Configure via `Solver.CLM.SnowPartition` key. Reference: Wang et al. (2019) GRL, Dai (2008) GRL, Jennings et al. (2018) Nat Commun.

#### Thin Snow Damping
New `Solver.CLM.ThinSnowDamping` key reduces spurious melt energy in shallow early-season snowpacks to prevent premature ablation.

#### Solar Zenith Angle (SZA) Snow Damping
The `Solver.CLM.SZASnowDamping` key corrects for CLM's narrowband snow optical parameters which assume a 60° solar zenith angle. At higher zenith angles, actual snow albedo is higher than CLM assumes, requiring damping of melt energy. Reference: Dang et al. (2019) The Cryosphere.

#### Advanced Albedo Schemes
Three snow albedo calculation methods are now available:
- **CLM**: Age-based exponential decay (default)
- **VIC**: Separate decay rates for cold and warm conditions
- **Tarboton**: Arrhenius temperature-dependent aging

Configure via `Solver.CLM.AlbedoScheme` key.

#### Fractional Snow Cover Options
Enhanced fractional snow covered area calculations:
- **CLM**: Standard formulation using roughness length
- **SZA**: Solar zenith angle-modulated formulation for energy-driven accumulation/melt asymmetry

Configure via `Solver.CLM.FracSnoScheme` key.

#### Snow Age Parameterization for VIS/NIR Bands
Separate aging parameters for visible and near-infrared bands following Abolafia-Rosenzweig et al. (2022), allowing independent calibration per spectral band.

#### ET Formulation Improvements
New options for improving evapotranspiration calculations:
- **StomataScheme**: Ball-Berry (default) or Medlyn stomatal conductance models
- **InterceptionScheme**: CLM3 exponential or CLM5 tanh-based canopy interception
- Enhanced interception and wet canopy fraction parameters

### Overland Flow Output Enhancements

New output capabilities for overland flow analysis:
- `Solver.PrintQxOverland` and `Solver.PrintQyOverland`: Print x and y direction surface flow velocities as 2D ParFlow binary files
- `Solver.WriteSiloQxOverland` and `Solver.WriteSiloQyOverland`: Silo format output for surface flow velocities
- `Solver.WritePDIQxOverland` and `Solver.WritePDIQyOverland`: PDI library exposure of overland flow data
- `NetCDF.WriteQxOverland` and `NetCDF.WriteQyOverland`: NetCDF4 output for surface flow velocities

### Bottom of Domain Output
Complementary to existing top-of-domain output:
- `Solver.PrintBottom`: Control writing of bottom Z-index files
- `Solver.WriteSiloBottom`: Silo format output for bottom domain data

### Surface Predictor Lateral Flow Enhancement
New `Solver.SurfacePredictor.LateralFlows` key enables use of overland flow lateral fluxes in the surface predictor water balance calculation for improved ponding predictions.

### Well Flux Corrections for Variable dz
The `Wells.CorrectForVarDz` key enables correction of well fluxes for variable dz cell thickness spacing, ensuring physically accurate volumetric flux calculations. Recommended to set to True rather than apply manual corrections.

### Seepage Patch Support for Overland Kinematic BC
New `Patch.{patch_name}.BCPressure.Seepage` key allows designation of specific patches as seepage patches in OverlandKinematic boundary condition formulations.

### IndicatorFieldNC Geometry Input Type
New `IndicatorFieldNC` geometry input type added alongside existing IndicatorField and SolidFile types for improved NetCDF geometry support.

### Flexible KINSOL Solver Configuration
ParFlow now supports choosing between:
- **Embedded KINSOL**: Original version embedded in ParFlow (default for backward compatibility)
- **External SUNDIALS**: Newer KINSOL from the SUNDIALS package via `SUNDIALS_ROOT` CMake variable

When using external SUNDIALS, additional solver options and improved numerical methods are available.

### Improved TCL Deprecation Handling
While TCL support remains available, it is now explicitly marked as deprecated with migration guidance to Python-based pftools. Warnings are issued at build time for Tcl 9.x compatibility issues. New `PARFLOW_ENABLE_TCL` CMake flag allows disabling TCL entirely if desired.

### Improved CONTRIBUTING.md Documentation
Enhanced documentation on code formatting requirements and Black version specifications for contributors.

## Bug Fixes

### CUDA 13+ Compatibility
Updated CUDA memory prefetch API calls to use `cudaMemPrefetchAsync` with new `cudaMemLocation` struct for compatibility with CUDA 13.0 and later.

### CMake Enhancements
- Fixed CMake minimum version to 3.22 (was 3.14)
- Updated FindHYPRE.cmake and FindSILO.cmake with proper case-sensitive names
- Improved libm math library detection for platforms requiring explicit linking
- Fixed Fortran source file extension handling for free-form Fortran compilation

### Dependency Updates
- Silo updated to version 4.12.0 (with --enable-shared support)
- Hypre updated to 2.33.0 in CI testing
- Python updated to 3.13 in macOS CI
- RMM pinned to v25.08.00 with depth-limited cloning
- Umpire updated to v2025.03.1

### Kokkos Backend Detection
Enhanced Kokkos configuration to properly detect and report active backends (CUDA, HIP, SYCL, OpenMP Target, Serial).

### AMPS/GPU Memory Management
Improved AMPS CMakeLists.txt organization with better dependency handling for RMM and Umpire. Added C++17 requirement detection when using newer RMM (≥25.06.00) or Umpire (≥2025.09.0) versions.

### CI/CD Improvements
- Added GitHub Actions concurrency controls to prevent duplicate workflow runs
- Updated Ubuntu testing from 22.04/24.04 to primarily 24.04
- Added Sundials external package testing
- Enhanced TCL deprecation testing with configuration option
- Improved self-hosted GPU runner support for CUDA testing
- Fixed Docker container architecture detection for multi-platform builds

### Valgrind Suppressions
Added suppressions for Intel MPI and libmlx4/libibverbs memory leak reports to reduce false positives in valgrind analysis.

## Internal/Developer Changes

### Python Code Style Enforcement
Black code formatter version pinned to **26.3.1** for consistent CI style checking. Updated `pfformat` script enforces this version requirement.

### CMake Configuration Enhancements
- Memory manager (UMPIRE/RMM) configuration messages improved
- Accelerator backend reporting enhanced with detailed status messages
- SILO find module now uses SILO in uppercase for consistency
- HYPRE find module now uses HYPRE in uppercase for consistency

### Documentation Updates
- Release process documentation clarified with additional directory navigation step
- User manual moved from docs/user_Manual/ to docs/user_manual/ (lowercase)
- Extended keys.rst documentation with additional CLM snow physics keys and output options

### Removed Legacy Build Files
Old Makefile.depend files removed from AMPS transport layer directories (common, mpi1, oas3, smpi, cuda) as CMake now handles dependencies.

### Code Formatting and Style
- ParFlow AMPS CMakeLists.txt refactored for improved readability and consistency
- RMM wrapper C++ code formatting improved with modern style
- Umpire wrapper significantly enhanced with better resource selection logic and detailed comments
- Removed unnecessary blank lines and improved code organization throughout

### Python pftools Updates
- Python package configuration updated for modern pyproject.toml workflow
- Example scripts cleaned up with removed trailing blank lines

## Known Issues

See https://github.com/parflow/parflow/issues for current bug/issue reports.

### Deprecated Features

- **TCL Support**: Marked as deprecated. Users are strongly encouraged to migrate to Python-based pftools. See https://github.com/parflow/parflow/issues/707 for migration guidance.
  - Tcl 9.x is not compatible with PFIDB file creation/reading due to API changes in Tcl.
  - Using Tcl 8.x is recommended if TCL support is still needed.


# ParFlow Release Notes 3.14.1
------------------------------

This release contains several bug fixes and minor feature updates.

ParFlow development and bug-fixes would not be possible without contributions of the ParFlow community.  Thank you for all the great contributions.

## Overview of Changes

**Release 3.14.1 was generated due to errors the integrated DOI number generation on Zenodo for the 3.14.0 release.**   3.14.0 does not have an explicit DOI number attatched to it.

## User Visible Changes

### Initial reservoir support added

An initial feature to support reservoirs has been added.   Reservoirs are only supported on problems where the top of domain lies at the top of the grid. This applies to all box domains and some  terrain following grid domains.   See the Reservoirs section of the ParFlow Users Manual for additional information.

### Add CLM transpiration / RZ formulation option

Transpiration / RZ formulation in Ferguson et al EES 2016 as an option.   For backward compatibility with prior formulation is kept as the default.  The key Solver.CLM.RZWaterStress control which formulation is used.  Option 0 (default) will limit transpiration when the top soil layer drops below wilting point, option 1 limits each layer independently.

### Feature predictor Darcy

Updated the predictor capability to include Darcy flux in addition to infiltration flux.  See Solver.SurfacePredictor key in the ParFlow User Manual.

### Mg semi of jacobian preconditioning

The solver has been been updated to run with the true jacobian and MGSemi. This allows efficient overland flow simulation with that preconditioner and now provides full PF capabilities on GPUs namely OverlandFlow, OverlandKinematic, and OverlandDiffusive. This is backwardly compatible and preserves all prior configurations (jacobian false, PFMG and other Hypre preconditions, etc).

### PDI integration 

Preliminary support for PDI has been added to ParFlow.  PDI is a
library that aims to decouple high-performance simulation codes from
-input/output concerns. It offers a declarative API that enables codes
to expose the buffers in which they store data and to notify PDI of
significant simulation steps. Additionally, it supports a plugin
system to make existing libraries such as HDF5, NetCDF, or Python
available to codes—potentially mixing several in a single execution.

### Added key to control output of initial conditions

A input key "PrintInitialConditions" was added to turn off output of initial condition output. Useful for restart to avoid duplication of information on restart.

### Timestep added to the Netcdf reader 

Add optional TimeStep for reading NetCDF files. Allows reading timesteps
from NetCDF files that contain more than a single timestep. Negative
TimeSteps are allowed and work like Python indexing (-1 is last
timestep).   See the TimeStep key in the ParFlow Users Manual.

```
      <runname>.ICPressure.Type = "NCFile"    ## Python syntax
      <runname>.Geom.domain.ICPressure.FileName = "initial_condition.nc" ## Python syntax
      <runname>.Geom.domain.ICPressure.TimeStep = -1 ## Python syntax
```

### Improved error reporting for output files

Some additional output is provided when opening of output files fails to make it easier to debug failed runs.


### Add pfsb reader and fix default_single tests

A `read_pfsb` function was added to the Python pftools for reading ParFlow scattered binary files.

### Hypre support updated

Support for Hypre 2.33.0 has been added.   The default Hypre used for ParFlow testing has been updated to Hypre 2.33.0.

### CMake updates

CMake dependency finding has been improved for CUDA, Hypre, NetCDF, OASIS.

### CUDA architecture can be specified when configuring

The standard -CMAKE_CUDA_ARCHITECTURES CMake variable can be used to set the CUDA architecture by users using NVidia accelerators.

### 

## Bug Fixes

### Update overland flow evaluation

Fix to overlandflow_eval_Kin to correct issue with overland surface boundary and Dirichlet side boundary.

OpenMP compiler flags should now be compiler-dependent to improve portability.

### Compressible storage positive enforcement

The subsurface storage computation previously allowed negative storage
values when the pressure head was negative, potentially leading to
physically unrealistic results. This issue occurred in both the Python
and Tcl pftools scripts. To address this, a minimum value of zero has
been imposed for compressible storage in both scripts, ensuring that
negative storage values are avoided. The implementation is backward
compatible and should not affect existing workflows.

### "undist" Sequential io fixes

Fixed the TCL undist script to undist new output files. Added undist to Python interface.

### Python run.dist() bug

A bug in run.dist() was not correctly files that had been distributed in a way that p, q, r did not exactly divide NX, NY, NZ.

### Fixed bug when printing mask files

The PrintInitialConfiditions flag was preventing mask files from being written when PrintMask was true.   The two flags should now be independent.

### Removed use of relative path for tclsh

Replace hard-coded tclsh command name with TCL_TCLSH found during CMake configuration.

### 

## Internal/Developer Changes

### Python package setup updated.

Update Python to use the more modern pyproject.toml package file.

### Python style check added

Python Black code style format check was added to pfformat.   Checking for proper formatting is part of the CI system and merge requests will not be accepted if style checks fail.

### GitHub CI

Previous method of using variable to force rebuilds was manual and caused issues. Forks can not see the cache variable which also can cause problems. Changed to use hash of the workflow files to trigger cache builds. This is conservative and will cause some unnecessary builds but workflow changes are relatively infrequent.

### Netcdf timing

Timing for Netcdf file I/O was added.

### Channel width enhancements

Input options have been added to read/print Channelwidths.   See "ChannelWidths" section of the ParFlow user manual for additional information.

### Support for using C++ as compiler was added.

ParFlow is moving to using C++ as the compiler for all of ParFlow.   Initial support has been added to CMake for doing this. The PARFLOW_BUILD_WITH_CPP CMake flag will enable use of C++ as the compiler.

### Add GPU execution CI action support

Add support for self-hosted runner at Princeton to test GPU execution of the CI tests.  Currently is testing CUDA.

## Remove local cub dependency

CUB is officially supported as part of the CUDA toolkit.

### CI Updates

Updated most tests to execute under Ubuntu 24.04.   Ubuntu 22.04 testing is retained for base executions only.

### Umpire support added and RMM update to current version 25.10.00a

The CUDA memory managers Umpire and RMM were updated.


## Known Issues

See https://github.com/parflow/parflow/issues for current bug/issue reports.


# ParFlow Release Notes 3.13.0
------------------------------

This release contains several bug fixes and minor feature updates.

ParFlow development and bug-fixes would not be possible without contributions of the ParFlow community.  Thank you for all the great contributions.

## Overview of Changes

## User Visible Changes

### Building ParFlow-CLM only is supported

Configuration flags have been added to support building only ParFlow-CLM for use cases where only CLM is desired.

### Kokkos version support updated to version 4.2.01 and performance improved

The Kokkos supported version has been updated to version 4.2.01.  This is the version used in our regression suite. Other versions may or may not work.

Refactors `KokkosMemSet` operations by replacing manual zero-initialization via `parallel_for `with a more efficient and portable. Up to ~38% reduction in solve time compared to the previous loop based implementation.

### OASIS version support updated to version 5.1.

The OASIS version used in the regression test suite was updated to OASIS 5.1.

### vegm file reading performance improvements in Python pftools

Improve speed of reading large vegm files in the Python read_vegm function.

### Documentation updates

Clarified top flux boundary conditions (e.g., OverlandFlow, SeepageFace) and EvapTrans files sign convention.  Typo in Haverkamp saturation formula: alpha replaced with A.  Key names "TensorByFileX" renamed to the correct "TensorFileX" key name.

ReadTheDocs had the irrigation type key with an 's' at the end but there should not be an 's' there.   This example shows the corrected documentation:
```
<runname>.Solver.CLM.IrrigationType "Drip"
```

## Bug Fixes

### Python pftools StartCount incorrect bounds check

The StartCount input value was incorrectly checked to be -1 or 0.  This prevented setting to larger value for doing a restart.  Larger values are now allowed.

### Python pftools reading porosity from a file not working

The Python input was throwing an error if the type for the porosity input was set to PFBFile.   This has been fixed and using a PFB file for input should work in Python.

### Memory corruption when using the PressureFile option

The PressureFile option (and others) were causing memory corruption due to multiple frees of the filenames. Removed the incorrect free calls in the shutdown logic.  This will fix some segmentation issues seen by users.

### 

## Internal/Developer Changes

### Direchelt boundary condition fix in nl_function_eval

Z_mult was incorrectly being divided by 2 in nl_function_eval.

### GitHub Actions updated

The CI testing suite was using out-dated GitHub Action modules; the modules have been updated.

### Added Python CI test result checks

The Python tests were incorrectly not checking results of runs and passing if the test ran.   Checks have been added as in the TCL test suite to check output results for regressions.

See the `pf_test_file` and `pf_test_file_with_abs` Python methods.

### Python CI tests for optional external package dependencies

Python CI tests are now guarded for optional package dependencies such as Hypre, Silo, etc.

See the `pf_test_file` and `pf_test_file_with_abs` Python methods.

### Compilation with Intel-OneAPI compiler fixes

The Intel-OneAPI compiler with fast-floating-point mode does not support isnan() (always evaluates to false).   NaN sentinel value was replaced with FLT_MIN.

### Improvements to C/C++ standards compliance

Minor code cleanup to remove old K&R style definitions and declarations.

### Updated etrace

Update the etrace script to work with Python3.

### 

## Known Issues

See https://github.com/parflow/parflow/issues for current bug/issue reports.


********************************************************************************
# ParFlow Release Notes 3.12.0
------------------------------

This release contains several bug fixes and minor feature updates.

ParFlow development and bug-fixes would not be possible without contributions of the ParFlow community.  Thank you for all the great contributions.

## Overview of Changes

* Documentation updates
* 


## User Visible Changes

### DockerHub Container has Python and TCL support.

The DockerHub Container has been updated to support both Python and TCL input scripts.   Previously only TCL was supported.   The type of script is determined by the file extension so make sure to use .tcl for TCL and and .py for Python as per standard file extension naming.

Simple examples using Docker:

```
docker run --rm -v $(pwd):/data parflow/parflow:version-3.12.0 default_single.py
```

```
docker run --rm -v $(pwd):/data parflow/parflow:version-3.12.0 default_single.tcl 1 1 1
```


### Dependency Updates

We have tested and updated some dependencies in ParFlow to use more current releases.  The following are used in our continuous integration builds and tests.

Ubuntu               22.04
Ubuntu               20.04

CMake                3.25.1
Hypre                2.26.0
Silo                 4.11
NetCDF-C             4.9.0
NetCDF-Fortan        4.5.5


CUDA                 11.8.0  (with OpenMPI 4.0.3)
UCX                  1.13.1
RMM                  0.10

Kokkos               3.3.01

Dependencies not listed are coming from the Ubuntu packages.   We try to have as few version specific dependencies as possible so other release may work.


### Surface Pressure Threshold

The surface pressure may now have a threshold applied.  This is controlled with several keys.

```
pfset Solver.ResetSurfacePressure        True      ## TCL syntax
<runname>.Solver.ResetSurfacePressure  = "True"    ## Python syntax
```
	  
This key changes any surface pressure greater than a threshold value to 
another value in between solver timesteps. It works differently than the Spinup keys and is intended to 
help with slope errors and issues and provides some diagnostic information.  The threshold keys are specified below.

The threshold value is specified with ```ResetSurfacePressure```

```
pfset Solver.ResetSurfacePressure.ThresholdPressure        10.0    ## TCL syntax
<runname>.Solver.ResetSurfacePressure.ThresholdPressure  = 10.0    ## Python syntax
```

The Solver.SpinUp key removes surface pressure in between solver timesteps.

```
pfset Solver.SpinUp   True        ## TCL syntax
<runname>.Solver.SpinUp = "True"  ## Python syntax
```
	  
### Top of domain indices output 

The capability to output the Top Z index and Top Patch Index have been added to allow easier processing of surface values.   The new input keys are PrintTop and WriteSiloTop.

```
pfset Solver.PrintTop False                    ## TCL syntax
<runname>.Solver.PrintTop = False              ## Python syntax

pfset Solver.WriteSiloTop True                  ## TCL syntax
<runname>.Solver.WriteSiloTop = True            ## Python syntax
```

The keys are used to turn on printing of the top of domain data.  'TopZIndex' is a NX * NY file with the Z index of the top of the domain. 'TopPatch' is the Patch index for the top of the domain.  A value of -1 indicates an (i,j) column does not intersect the domain. The data is written as a PFB or Silo formats.

### Documentation Updates

The read-the-docs manual has been cleaned up; many formatting and typos have been fixed from the Latex conversion.

## Bug Fixes

### CLM 

Fixed an issue that was identified by @danielletijerina where some bare soil on vegetated surfaces wasn't being beta-limited in CLM. Fixes to clm_thermal.F90 were implemented. At the same time, CLM snow additions and dew corrections by LBearup were added. A snow-age fix for deep snow was implemented along with canopy dew.

### Python PFtools
The _overland_flow_kinematic method was updated to match the outflow of ParFlow along the edges of irregular domains, which the prior Hydrology Python PFTools did not.

a) the slope in both x and y are corrected (by copying the corresponding value inside the mask) outside the mask edges in lower x and y as they both come into play through "slope".
b) because the correction is now done at lower x and y edges in both slopex and slopey, this could lead to overwriting the slopes outside for grid cells that are both outside x and y lower edges. For this, the calculation in x (q_x, qeast) is done first, after adjusting slopes outside lower x edges and then the calculation in y (q_y, qnorth) is done second, after adjusting slopes outside lower y edges.

## Internal/Developer Changes

### CI Testing Updates

The GitHub Actions tests have been updated to use later Ubuntu releases.   The 18.04 tests were removed and tests were moved to to 22.04.   Currently testing is done with both 20.04 and 22.04.
Dependencies have been updated for NetCDF, Hypre, GCC

### NetCDF Testing

The NetCDF testing has been updated to unify the GitHub Actions for OASIS3 tests and the other regression tests.

### Regression Test Comparison Directory

The TCL script pfTestFile used for regression testing has been updated to enable setting the directory for the regression test comparison files.  Example usage:

```
set correct_output_dir "../../correct_output/clm_output"
pftestFile clm.out.press.$i_string.pfb "Max difference in Pressure for timestep $i_string" $sig_digits $correct_output_dir
```

## Known Issues

See https://github.com/parflow/parflow/issues for current bug/issue reports.
