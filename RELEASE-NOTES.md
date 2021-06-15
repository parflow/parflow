# ParFlow Release Notes 3.9.0
-----------------------------

ParFlow improvements and bug-fixes would not be possible without
contributions of the ParFlow community.  Thank you for all the great
contributions.

Note : Version 3.9.0 is a minor update to v3.8.0.  These release notes cover 
changes made in both 3.8.0 and 3.9.0.   In 3.9.0, we are improving our Spack
support and added a smoke test for running under Spack and creating a release tag 
for Spack to download.   If you have version 3.8.0 installed the 3.9.0 update
does NOT add any significant features or bug fixes.

## Overview of Changes

* ParFlow Google Group
* Default I/O mode changed to amps_sequential_io
* CLM Solar Zenith Angle Calculation
* Parallel NetCDF dataset compression
* Kokkos support
* Output of Van Genuchten variables
* Python interface updates
* Update Hypre testing to v2.18.2
* MPI runner change
* Python Interface
* Segmentation fault at end of simulation run with Van Genuchten
* Memory errors when rank contained no active cells
* PFMGOctree solver 
* GFortran compilation errors
* CMake CI fixes
* CLM initialization bug 
* CMake cleanup
* Fixed compilation issues in sequential amps layer
* CI has moved to Google Actions
* Sponsors acknowledgment 
* PFModule extended to support output methods
* Hypre SMG and PFMG
* Installation of example for smoke test

## User Visible Changes

### ParFlow Google Group

ParFlow has switched to using a Google group for discussion from the
previous email list server.

https://groups.google.com/g/parflow

### Default I/O mode changed to amps_sequential_io

Change the default I/O model to amps_sequential_io because this is the
most common I/O model being used.

### CLM Solar Zenith Angle Calculation

Add slope and aspect when determining the solar zenith angle in CLM.
A new key was added Solver.CLM.UseSlopeAspect for the inclusion of
slopes when determining solar zenith angles

### Parallel NetCDF dataset compression

Added configurable deflate (zlib) compression capabilities to the
NetCDF based writing routines. The possibilities of parallel data
compression will only work in combination with the latest NetCDF4
v4.7.4 release.

pfset NetCDF.Compression True Enable deflate based compression (default: False)
pfset NetCDF.CompressionLevel 1 Compression level (0-9) (default: 1)

This work was implemented as part of EoCoE-II project (www.eocoe.eu).

Benchmark tests show that the datasize for regular NetCDF output files
could be lowered by a factor of three, in addition, as less data is
written to disk, the reduced data size can also lower the overall I/O
footprint towards the filesystem. Therefore, depending of the selected
setup, the compression overhead can be balanced by reduced writing
times.

### Kokkos support

User instructions on how to use Kokkos backend can be found from
README-GPU.md. 

Add Kokkos accelerator backend support as an alternative to the
native ParFlow CUDA backend to support more accelerator devices. The
implementation does not rely on any CUDA-specific arguments but still
requires Unified Memory support from the accelerator devices. It
should be compatible with AMD GPUs when sufficient Unified Memory
support is available.

The performance of using CUDA through the Kokkos library is slightly
worse in comparison to the ParFlow native CUDA implementation. This is
because a general Kokkos implementation cannot leverage certain CUDA
features such as cudaMemset() for initialization or CUDA pinned
host/device memory for MPI buffers. Also, Kokkos determines grid and
block sizes for compute kernels differently.

The RMM pool allocator for Unified Memory can be used with Kokkos
(when using Kokkos CUDA backend) and improves the performance very
significantly. In the future, a Unified Memory pool allocator that
supports AMD cards is likely needed to achieve good performance.

Performance of the simulation initialization phase has
been improved significantly when using GPUs (with CUDA and Kokkos).

### Output of Van Genuchten variables

Add output for Van Genuchten values alpha, n, sres, ssat.  The new
output will be generated when the print_subsurf_data key is set.

### Python interface updates

### Update Hypre testing to v2.18.2

The version of Hypre used for testing was updated to v2.18.2.  This
matches XSDK 0.5 version requirements.

### MPI runner change

The method used to find automatically find the MPI runner (mpiexec,
srun etc) is based purely on the CMake FindMPI script.  This should 
be invisible to most users.

### Python Interface

The Beta Python interface continues to be developed.  Many
improvements and bugfixes have been made.

* Add hydrology functions
* CLM API bug fixes
* CLM ET calculation
* Allow clm_output function to return only 2D arrays
* Add irrigation to CLM variables
* dx/dy/dz support when writing PFB files
* Python testing support was added
* New feature to only show validation results for errors
* Table builder update: adding databases, cleanup
* Domain builder helper with examples, docs

### Installation of example for smoke testing

The simple single phase test default_single.tcl is installed for smoke testing an installation.

## Bug Fixes

### Segmentation fault at end of simulation run with Van Genuchten

Segmentation fault when freeing memory at the end of simulation. 

### Memory errors when rank contained no active cells

The computation of the real space z vector was running beyond
temporary array (zz) resulting in memory errors.

### PFMGOctree solver 

PFMGOctree was not inserting the surface coefficients correctly into
the matrix with overland flow enabled.

### GFortran compilaton errors

Fixed GFortran compilation errors in ifnan.F90 with later GNU releases.
Build was tested against GNU the 10.2.0 compiler suite.

### CMake CI fixes

On some systems, it is necessary for any binary compiled with mpi to
be executed with the appropriate ${mpiexec} command. Setting
PARFLOW_TEST_FORCE_MPIEXEC forces sequential tests to be executed with
the ${MPIEXEC} command with 1 rank.

### CLM initialization bug 

Fixed CLM bug causing long initialization times.

### CMake cleanup

Updated CMake to more current usage patterns and CMake minor bugfixes.

### Fixed compilation issues in sequential amps layer

The AMPS sequential layer had several bugs preventing it from
compiling. Tests are passing again with a sequential build.

## Internal/Developer Changes

### CI has moved to Google Actions

TravisCI integration for CI has been replaced with Google Actions.

### Sponsors acknowledgment 

A new file has been added (SPONSORS.md) to enable acknowledgment the
sponsors of ParFlow development.  Please feel free to submit a pull request
if you wish to add a sponsor.

## Testing framework refactoring

The testing framework has been refactored to support Python.  Directory
structure for tests has changed.

### PFModule extended to support output methods

Add support in PFModule for module output.  Two new methods were added
to the PFModule 'class' to output time variant and time invariant
data.  This allows modules to have methods on each instance for
generating output directly from the module.  Previously the approach
was to copy data to a problem data variable and output from the copy.

### Hypre SMG and PFMG

Refactored common Hypre setup code to a method to keep Hypre setup consistent.

## Known Issues

See https://github.com/parflow/parflow/issues for current bug/issue reports.
