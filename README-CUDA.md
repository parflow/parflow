# Parflow with GPU acceleration Notes

**WARNING!** Parflow/CUDA is still beta and 
there may be issues with it.

The GPU accelerated version of Parflow uses NVIDIA Compute Unified Device Architecture (CUDA) and is therefore compatible only with NVIDIA GPU's. Currently, the minimum supported CUDA compute capability
for the hardware is 7.0 (NVIDIA Volta architecture).

The current implementation of Parflow/CUDA improves the performance 
for large problems, but may be slower for test cases or small problems 
due to some overhead associated with the GPU use.


## Configuration

Building with GPU acceleration is optional and requires that [CUDA](https://developer.nvidia.com/cuda-zone) and [RMM](https://github.com/rapidsai/rmm) are installed to the system (RMM_ROOT environment variable must point to RMM root directory). Furthermore, the used MPI version must have sufficient CUDA support, e.g., ParaStationMPI/5.4.0-1-CUDA is found to work well. The GPU acceleration is activated by specifying -DPARFLOW_ENABLE_CUDA=TRUE option to the CMake, e.g.,

```shell
CC=mpicc CXX=mpicxx FC=mpif90 cmake ../parflow -DPARFLOW_AMPS_LAYER=mpi1 -DHYPRE_ROOT=$EBROOTHYPRE -DCMAKE_C_FLAGS=-fopenmp -DHDF5_ROOT=$EBROOTHDF5 -DSILO_ROOT=$EBROOTSILO -DCMAKE_BUILD_TYPE=Release -DPARFLOW_ENABLE_TIMING=TRUE -DPARFLOW_HAVE_CLM=ON -DCMAKE_INSTALL_PREFIX=${PARFLOW_DIR} -DNETCDF_DIR=$EBROOTNETCDF -DTCL_TCLSH=${EBROOTTCL}/bin/tclsh8.6 -DPARFLOW_AMPS_SEQUENTIAL_IO=on -DPARFLOW_ENABLE_SLURM=TRUE -DPARFLOW_ENABLE_CUDA=TRUE
```

## Running Parflow with GPU acceleration

Parflow/CUDA requires that each MPI process has access to a GPU device. It is allowed for multiple processes to use the same GPU device. However, if one GPU device is used by multiple MPI processes, usage of CUDA Multi-Process Service (MPS) is recommended for
best performance.

Any existing input script can be run with Parflow/CUDA; no changes
are necessary. However, it is noted that two subsequent runs of the same input script using the same compiled Parflow/CUDA executable are not guaranteed to produce identical results. This is expected behavior due to atomic operations performed by the GPU device (i.e., the order of floating-point operations is random).
