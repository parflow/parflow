/*--------------------------------------------------------------------------
 * CUDA error handling macro
 *--------------------------------------------------------------------------*/
#ifndef CUDA_ERR
#include "pfcudaerr.h"
#endif

/*--------------------------------------------------------------------------
 * CUDA lambda definition (visible for host and device functions)
 *--------------------------------------------------------------------------*/
#define GPU_LAMBDA [=] __host__ __device__

/*--------------------------------------------------------------------------
 * CUDA helper functions (must be wrapped to extern "C++" due to template use)
 *--------------------------------------------------------------------------*/
extern "C++"{ 
template <typename T>
__host__ __device__ static inline void PlusEquals(T *array_loc, T value)
{
    //Define this function depending on whether it runs on GPU or CPU
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    atomicAdd(array_loc, value);
#else
    *array_loc += value;
#endif
}
}

static void PrintDeviceProperties(){
      int device;
      CUDA_ERR(cudaGetDevice(&device));

      struct cudaDeviceProp props;
      CUDA_ERR(cudaGetDeviceProperties(&props, device));
      printf("\nGPU Model Name: %s\n", props.name);
      printf("GPU Compute Capability: %d.%d\n", props.major,props.minor);
      printf("GPU Maximum Threads Per Block: %d\n", props.maxThreadsPerBlock);
      printf("GPU Maximum Threads Per Dims: %d x %d x %d\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
      //printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
}

/*--------------------------------------------------------------------------
 * CUDA loop kernels (must be wrapped to extern "C++" due to template use)
 *--------------------------------------------------------------------------*/
extern "C++"{ 
template <typename LOOP_BODY>
__global__ void ForxyzKernel(LOOP_BODY loop_body, const int PV_ixl, const int PV_iyl, const int PV_izl,
const int PV_diff_x, const int PV_diff_y, const int PV_diff_z)
{

    int i = ((blockIdx.x*blockDim.x)+threadIdx.x);
    if(i > PV_diff_x)return;
    int j = ((blockIdx.y*blockDim.y)+threadIdx.y);
    if(j > PV_diff_y)return;
    int k = ((blockIdx.z*blockDim.z)+threadIdx.z);
    if(k > PV_diff_z)return;

    i += PV_ixl;
    j += PV_iyl;
    k += PV_izl;
    
    loop_body(i, j, k);
}
}

/*--------------------------------------------------------------------------
 * CUDA loop macro redefinitions
 *--------------------------------------------------------------------------*/
#undef GrGeomInLoopBoxes
#define GrGeomInLoopBoxes(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, loop_body)         \
  {                                                                                   \
    BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                               \
    for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                      \
    {                                                                                 \
      Box box = BoxArrayGetBox(boxes, PV_box);                                        \
      /* find octree and region intersection */                                       \
      int PV_ixl = pfmax(ix, box.lo[0]);                                              \
      int PV_iyl = pfmax(iy, box.lo[1]);                                              \
      int PV_izl = pfmax(iz, box.lo[2]);                                              \
      int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                                   \
      int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                                   \
      int PV_izu = pfmin((iz + nz - 1), box.up[2]);                                   \
                                                                                      \
      int PV_diff_x = PV_ixu - PV_ixl;                                                \
      int PV_diff_y = PV_iyu - PV_iyl;                                                \
      int PV_diff_z = PV_izu - PV_izl;                                                \
                                                                                      \
      const int BLOCKSIZE = 8;                                                        \
      dim3 grid = dim3((PV_diff_x + BLOCKSIZE) / BLOCKSIZE,                           \
        (PV_diff_y + BLOCKSIZE) / BLOCKSIZE,                                          \
        (PV_diff_z + BLOCKSIZE) / BLOCKSIZE);                                         \
      dim3 block = dim3(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);                             \
                                                                                      \
      ForxyzKernel<<<grid, block>>>(                                                  \
          GPU_LAMBDA(int i, int j, int k)loop_body,                                   \
          PV_ixl, PV_iyl, PV_izl, PV_diff_x, PV_diff_y, PV_diff_z);                   \
      CUDA_ERR( cudaPeekAtLastError() );                                              \
      CUDA_ERR( cudaDeviceSynchronize() );                                            \
    }                                                                                 \
  }
