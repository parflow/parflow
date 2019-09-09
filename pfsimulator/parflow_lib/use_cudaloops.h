/*--------------------------------------------------------------------------
 * CUDA error handling macro
 *--------------------------------------------------------------------------*/
#define CUDA_ERR( err ) (gpu_error( err, __FILE__, __LINE__ ))
static void gpu_error(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

/*--------------------------------------------------------------------------
 * CUDA lambda definition (visible for host and device functions)
 *--------------------------------------------------------------------------*/
#define GPU_LAMBDA [=] __host__  __device__

/*--------------------------------------------------------------------------
 * CUDA loop kernels (must be wrapped to extern "C++" due to template use)
 *--------------------------------------------------------------------------*/
extern "C++"{ 
template <typename T, typename LOOP_BODY>
__global__ void forxyz_kernel(T loop_data, LOOP_BODY loop_body, const int PV_ixl, const int PV_iyl, const int PV_izl,
const int PV_ixu, const int PV_iyu, const int PV_izu)
{

    int i = ((blockIdx.x*blockDim.x)+threadIdx.x) + PV_ixl;
    if(i > PV_ixu)return;
    int j = ((blockIdx.y*blockDim.y)+threadIdx.y) + PV_iyl;
    if(j > PV_iyu)return;
    int k = ((blockIdx.z*blockDim.z)+threadIdx.z) + PV_izl;
    if(k > PV_izu)return;
  
    loop_body(i, j, k, loop_data);
}
}

void printDeviceProperties(){
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
 * CUDA loop macro redefinitions
 *--------------------------------------------------------------------------*/

//#undef GrGeomInLoopBoxes
#define GrGeomInLoopBoxesGPU(grgeom, ix, iy, iz, nx, ny, nz, loop_data, loop_body) \
  {                                                                      \
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;                  \
    int *PV_visiting = NULL;                                             \
    BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                  \
    for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)         \
    {                                                                    \
      Box box = BoxArrayGetBox(boxes, PV_box);                           \
      /* find octree and region intersection */                          \
      PV_ixl = pfmax(ix, box.lo[0]);                                     \
      PV_iyl = pfmax(iy, box.lo[1]);                                     \
      PV_izl = pfmax(iz, box.lo[2]);                                     \
      PV_ixu = pfmin((ix + nx - 1), box.up[0]);                          \
      PV_iyu = pfmin((iy + ny - 1), box.up[1]);                          \
      PV_izu = pfmin((iz + nz - 1), box.up[2]);                          \
                                                                         \
      const int BLOCKSIZE = 8;                                           \
      dim3 grid = dim3((PV_ixu - PV_ixl + BLOCKSIZE) / BLOCKSIZE,        \
        (PV_iyu - PV_iyl + BLOCKSIZE) / BLOCKSIZE,                       \
        (PV_izu - PV_izl + BLOCKSIZE) / BLOCKSIZE);                      \
      dim3 block = dim3(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);                \
                                                                         \
      int gpu_count = 1;                                                 \
      /* CUDA_ERR(cudaGetDeviceCount(&gpu_count));*/                     \
      /*printf("DeviceCount: %d\n", gpu_count);*/                        \
      for (int gpu = 0; gpu < gpu_count; gpu++){                         \
        forxyz_kernel<<<grid, block>>>(loop_data, loop_body, PV_ixl,     \
        PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu);                         \
      }                                                                  \
      CUDA_ERR( cudaPeekAtLastError() );                                 \
      CUDA_ERR( cudaDeviceSynchronize() );                               \
    }                                                                    \
  }
//   #undef GrGeomInLoop
  #define GrGeomInLoopGPU(i, j, k, grgeom,                               \
                     r, ix, iy, iz, nx, ny, nz, loop_data, loop_body)    \
  {                                                                      \
    if (r == 0 && GrGeomSolidInteriorBoxes(grgeom))                      \
    {                                                                    \
      GrGeomInLoopBoxesGPU(grgeom, ix, iy, iz, nx, ny, nz, loop_data,    \
      loop_body);                                                        \
    }                                                                    \
    else                                                                 \
    {                                                                    \
      GrGeomOctree  *PV_node;                                            \
      double PV_ref = pow(2.0, r);                                       \
                                                                         \
      i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;                     \
      j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;                     \
      k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;                     \
      printf("NO GPU SUPPORT FOR  GrGeomOctreeInteriorNodeLoop, exiting...");\
      exit(1);                                                           \
      GrGeomOctreeInteriorNodeLoop(i, j, k, PV_node,                     \
                                   GrGeomSolidData(grgeom),              \
                                   GrGeomSolidOctreeBGLevel(grgeom) + r, \
                                   ix, iy, iz, nx, ny, nz,               \
                                   TRUE,                                 \
                                   loop_body);                           \
    }                                                                    \
  }