# Building Parflow with GPU acceleration

**WARNING!** Parflow GPU support is still in beta and there may be issues with it.

The GPU acceleration is currently compatible only with NVIDIA GPUs due to CUDA being the only backend option so far. The minimum supported CUDA compute capability for the hardware is 6.0 (NVIDIA Pascal architecture).

Building with CUDA may improve the performance significantly for large problems but is often slower for test cases and small problems due to initialization overhead associated with the GPU use. Installation on Ubuntu 1404 with all dependencies excluding the GPU driver is found [here](.travis.yml).


## CMake

Building with GPU acceleration requires a [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) installation and a CUDA-Aware MPI library (if MPI is used). The pointers passed to the MPI library point to pinned GPU memory allocations, and therefore support for Unified Memory from MPI library is not required. Examples of CUDA-Aware MPI libraries include MVAPICH2-GDR 2.3.3, OpenMPI 4.0.3, and ParaStationMPI 5.4.2. Moreover, pool allocation for Unified Memory can be activated by using [RMM](https://github.com/rapidsai/rmm) library (v. 0.10) and often leads to notably better performance. The GPU acceleration is activated by specifying *DPARFLOW_ENABLE_CUDA=TRUE* option to the CMake, e.g.,

```shell
cmake ../parflow -DPARFLOW_AMPS_LAYER=cuda -DCMAKE_BUILD_TYPE=Release -DPARFLOW_ENABLE_TIMING=TRUE -DPARFLOW_HAVE_CLM=ON -DCMAKE_INSTALL_PREFIX=${PARFLOW_DIR} -DPARFLOW_ENABLE_CUDA=TRUE
```
where *DPARFLOW_AMPS_LAYER=cuda* is required to activate application-side GPU-based data packing and unpacking on pinned GPU staging buffers. Using *DPARFLOW_AMPS_LAYER=mpi1* typically results in significantly worse performance and segfaults may be experienced if the MPI library does not have full Unified Memory support for derived data types. Furthermore, RMM library can be activated by specifying the RMM root directory with *DRMM_ROOT=/path/to/rmm_root* as follows:
```shell
cmake ../parflow -DPARFLOW_AMPS_LAYER=cuda -DCMAKE_BUILD_TYPE=Release -DPARFLOW_ENABLE_TIMING=TRUE -DPARFLOW_HAVE_CLM=ON -DCMAKE_INSTALL_PREFIX=${PARFLOW_DIR} -DPARFLOW_ENABLE_CUDA=TRUE -DRMM_ROOT=/path/to/RMM
```
## Running Parflow with GPU acceleration

Running Parflow built with GPU support requires that each MPI process has access to a GPU device. Best performance is typically achieved by launching one MPI process per available GPU device. Launching more processes is not recommended and typically results in reduced performance. However, if more processes are used, performance may be improved using CUDA Multi-Process Service (MPS).

Any existing input script can be run with GPU acceleration; no changes are necessary. However, it is noted that two subsequent runs of the same input script using the same compiled executable are not guaranteed to produce identical results. This is expected behavior due to atomic operations performed by the GPU device (i.e., the order of floating-point operations may change between two subsequent runs).
