#ifdef HAVE_CUDA

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
template <typename LOOP_BODY>
__global__ void forxyz_kernel(LOOP_BODY loop_body, const int PV_ixl, const int PV_iyl, const int PV_izl,
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
#define GrGeomInLoopBoxesG(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, loop_body)\
  {                                                                      \
    int *PV_visiting = NULL;                                             \
    BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                  \
    for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)         \
    {                                                                    \
      Box box = BoxArrayGetBox(boxes, PV_box);                           \
      /* find octree and region intersection */                          \
      int PV_ixl = pfmax(ix, box.lo[0]);                                 \
      int PV_iyl = pfmax(iy, box.lo[1]);                                 \
      int PV_izl = pfmax(iz, box.lo[2]);                                 \
      int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                      \
      int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                      \
      int PV_izu = pfmin((iz + nz - 1), box.up[2]);                      \
                                                                         \
      int PV_diff_x = PV_ixu - PV_ixl;                                   \
      int PV_diff_y = PV_iyu - PV_iyl;                                   \
      int PV_diff_z = PV_izu - PV_izl;                                   \
                                                                         \
      const int BLOCKSIZE = 8;                                           \
      dim3 grid = dim3((PV_diff_x + BLOCKSIZE) / BLOCKSIZE,              \
        (PV_diff_y + BLOCKSIZE) / BLOCKSIZE,                             \
        (PV_diff_z + BLOCKSIZE) / BLOCKSIZE);                            \
      dim3 block = dim3(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);                \
                                                                         \
      int gpu_count = 1;                                                 \
      /* CUDA_ERR(cudaGetDeviceCount(&gpu_count));*/                     \
      /*printf("DeviceCount: %d\n", gpu_count);*/                        \
      for (int gpu = 0; gpu < gpu_count; gpu++){                         \
        forxyz_kernel<<<grid, block>>>(                                  \
            GPU_LAMBDA(int i, int j, int k)loop_body,                    \
            PV_ixl, PV_iyl, PV_izl, PV_diff_x, PV_diff_y, PV_diff_z);    \
      }                                                                  \
      CUDA_ERR( cudaPeekAtLastError() );                                 \
      CUDA_ERR( cudaDeviceSynchronize() );                               \
    }                                                                    \
  }
//#undef GrGeomInLoop
#define GrGeomInLoopG(i, j, k, grgeom,                                    \
                     r, ix, iy, iz, nx, ny, nz, loop_body)               \
  {                                                                      \
    if (r == 0 && GrGeomSolidInteriorBoxes(grgeom))                      \
    {                                                                    \
      GrGeomInLoopBoxesG(i, j, k, grgeom, ix, iy, iz, nx, ny, nz, loop_body);   \
    }                                                                    \
    else                                                                 \
    {                                                                    \
      GrGeomOctree  *PV_node;                                            \
      double PV_ref = pow(2.0, r);                                       \
                                                                         \
      i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;                     \
      j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;                     \
      k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;                     \
                                                                         \
      GrGeomOctreeInteriorNodeLoop(i, j, k, PV_node,                     \
                                   GrGeomSolidData(grgeom),              \
                                   GrGeomSolidOctreeBGLevel(grgeom) + r, \
                                   ix, iy, iz, nx, ny, nz,               \
                                   TRUE,                                 \
                                   loop_body);                           \
    }                                                                    \
  }
  #endif