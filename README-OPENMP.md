# Building ParFlow with OpenMP Multicore Accelerator

**NOTE 04/21/2020** As of writing OpenMP support should be considered "in beta".  Please check answer tolerances, variances of 10^-17 can be expected in saturation and pressure results.

OpenMP was implemented with the expectation of OpenMP 4.5 specification support.  Check your compiler's OpenMP support to verify support.  GCC 7.2.0 and above are confirmed compatible.

Use of OpenMP will incur overhead from spawning, suspending, resuming, and synchronizing threads; the more threads allowed the greater this overhead becomes (in terms of microseconds).
Smaller problems will likely not benefit from OpenMP, but larger problems may.  Problem configurations will also have direct impact.  Your mileage may vary from problem to problem.

## CMake

Simply add `-DPARFLOW_ACCELERATOR_BACKEND=omp` to your cmake command, and ParFlow will compile with OpenMP support.

## Configuring thread count

OpenMP acceleration is compatible with MPI.  To specify the maximum number of OpenMP threads per MPI rank/process, run the following command before running your TCL script

```shell
$ export OMP_NUM_THREADS=N
```

Where `N` is the number of threads desired.  For example, to run 28 threads total with a domain decomposition of `2 2 1`:

```shell
$ export OMP_NUM_THREADS=7
$ tclsh default_single.tcl 2 2 1
```

This will spawn 4 MPI ranks as normal, each having up to 7 OpenMP threads, for a total max thread count of 28.

When submitting jobs to a cluster or supercomputer that utilizes a resource manager, such as slurm, make sure to include this command in your job script.
For example, to run a problem on two nodes with 28 cores each, one might run

```shell
#SBATCH --nodes=2
#SBATCH --n-tasks-per-node=2

export OMP_NUM_THREADS=14
tclsh default_single.tcl 2 2 1
```

Depending on environment configuration, when using OpenMPI the use of the --map-by flag may be necessary.  OpenMP threads might otherwise be locked to one core, causing severe performance problems.

## Limitations

OpenMP is presently implemented as CPU-only.  OpenMP is confirmed to be compatible with MPI based on MPICH 3.2.1 and OpenMPI 4.0.3.
Use of GPU accelerators should be done through use of the CUDA compile option.
Refer to README-CUDA.md for build process details, support, and caveats.  OpenMP and CUDA are currently mutually exclusive and cannot be used together.

## Development

Refer to `pf_omploops.h` for OpenMP functions and loop definitions.