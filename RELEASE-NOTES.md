# ParFlow Release Notes 3.13.0
------------------------------

This release contains several bug fixes and minor feature updates.

ParFlow development and bug-fixes would not be possible without contributions of the ParFlow community.  Thank you for all the great contributions.

## Overview of Changes

## User Visible Changes

### Building ParFlow-CLM only is supported

Configuration flags have been added to support building only ParFlow-CLM for use cases where only CLM is desired.

### Kokkos version support updated to version 4.2.01

The Kokkos supported version has been updated to version 4.2.01.  This is the version used in our regression suite.  Other versions may or may not work.

### OASIS version support updated to version 5.1.

The OASIS version used in the regression test suite was updated to OASIS 5.1.

### vegm file reading performance improvements in Python pftools

Improve speed of reading large vegm files in the Python read_vegm function.

### Documentation updates

Clarified top flux boundary conditions (e.g., OverlandFlow, SeepageFace) and EvapTrans files sign convention.  Typo in Haverkamp saturation formula: alpha replaced with A.  Key names "TensorByFileX" renamed to the correct "TensorFileX" key name.

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
