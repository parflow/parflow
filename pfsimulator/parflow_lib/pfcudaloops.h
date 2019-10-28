#ifndef PFCUDALOOPS_H
#define PFCUDALOOPS_H

/*--------------------------------------------------------------------------
 * Include CUDA error handling header
 *--------------------------------------------------------------------------*/
#include "pfcudaerr.h"

extern "C++"{
/*--------------------------------------------------------------------------
 * CUDA lambda definition (visible for host and device functions)
 *--------------------------------------------------------------------------*/
#define GPU_LAMBDA [=] __host__ __device__

/*--------------------------------------------------------------------------
 * CUDA helper functions
 *--------------------------------------------------------------------------*/
template <typename T>
__host__ __device__ static inline void PlusEqualsSwitch(T *array_loc, T value)
{
    //Define this function depending on whether it runs on GPU or CPU
#ifdef __CUDA_ARCH__
    atomicAdd(array_loc, value);
#else
    *array_loc += value;
#endif
}

/*--------------------------------------------------------------------------
 * CUDA helper macro redefinitions
 *--------------------------------------------------------------------------*/
#undef PlusEquals
#define PlusEquals(a, b) PlusEqualsSwitch(&(a), b)

/*--------------------------------------------------------------------------
 * CUDA loop kernels
 *--------------------------------------------------------------------------*/
template <typename LAMBDA_BODY>
__global__ void BoxKernelI0(LAMBDA_BODY loop_body, 
    const int ix, const int iy, const int iz, const int nx, const int ny, const int nz)
{

    int i = ((blockIdx.x*blockDim.x)+threadIdx.x);
    if(i >= nx)return;
    int j = ((blockIdx.y*blockDim.y)+threadIdx.y);
    if(j >= ny)return;
    int k = ((blockIdx.z*blockDim.z)+threadIdx.z);
    if(k >= nz)return;

    i += ix;
    j += iy;
    k += iz;
    
    loop_body(i, j, k);
}

template <typename LAMBDA_INIT, typename LAMBDA_BODY>
__global__ void BoxKernelI1(LAMBDA_INIT loop_init, LAMBDA_BODY loop_body, 
    const int ix, const int iy, const int iz, const int nx, const int ny, const int nz)
{

    int i = ((blockIdx.x*blockDim.x)+threadIdx.x);
    if(i >= nx)return;
    int j = ((blockIdx.y*blockDim.y)+threadIdx.y);
    if(j >= ny)return;
    int k = ((blockIdx.z*blockDim.z)+threadIdx.z);
    if(k >= nz)return;

    const int i1 = loop_init(i, j, k);

    i += ix;
    j += iy;
    k += iz;
    
    loop_body(i, j, k, i1);       
}

template <typename LAMBDA_INIT1, typename LAMBDA_INIT2, typename LAMBDA_BODY>
__global__ void BoxKernelI2(LAMBDA_INIT1 loop_init1, LAMBDA_INIT2 loop_init2, LAMBDA_BODY loop_body, 
    const int ix, const int iy, const int iz, const int nx, const int ny, const int nz)
{

    int i = ((blockIdx.x*blockDim.x)+threadIdx.x);
    if(i >= nx)return;
    int j = ((blockIdx.y*blockDim.y)+threadIdx.y);
    if(j >= ny)return;
    int k = ((blockIdx.z*blockDim.z)+threadIdx.z);
    if(k >= nz)return;

    const int i1 = loop_init1(i, j, k);
    const int i2 = loop_init2(i, j, k);

    i += ix;
    j += iy;
    k += iz;
    
    loop_body(i, j, k, i1, i2);    
}

template <typename LAMBDA_INIT1, typename LAMBDA_INIT2, typename LAMBDA_INIT3, typename LAMBDA_BODY>
__global__ void BoxKernelI3(LAMBDA_INIT1 loop_init1, LAMBDA_INIT2 loop_init2, LAMBDA_INIT3 loop_init3, 
    LAMBDA_BODY loop_body, const int ix, const int iy, const int iz, const int nx, const int ny, const int nz)
{

    int i = ((blockIdx.x*blockDim.x)+threadIdx.x);
    if(i >= nx)return;
    int j = ((blockIdx.y*blockDim.y)+threadIdx.y);
    if(j >= ny)return;
    int k = ((blockIdx.z*blockDim.z)+threadIdx.z);
    if(k >= nz)return;

    const int i1 = loop_init1(i, j, k);
    const int i2 = loop_init2(i, j, k);
    const int i3 = loop_init3(i, j, k);

    i += ix;
    j += iy;
    k += iz;
    
    loop_body(i, j, k, i1, i2, i3);    
}

/*--------------------------------------------------------------------------
 * CUDA loop macro redefinitions
 *--------------------------------------------------------------------------*/
#undef BoxLoopI1
#define BoxLoopI1(i_dummy, j_dummy, k_dummy,                                        \
                  ix, iy, iz, nx, ny, nz,                                           \
                  i1, nx1, ny1, nz1, sx1, sy1, sz1,                                 \
                  loop_body)                                                        \
{                                                                                   \
  if(nx > 0 && ny > 0 && nz > 0)                                                    \
  {                                                                                 \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);     \
                                                                                    \
    const int blocksize_h = 16;                                                     \
    const int blocksize_v = 4;                                                      \
    dim3 grid = dim3(((nx - 1) + blocksize_h) / blocksize_h,                        \
        ((ny - 1) + blocksize_h) / blocksize_h,                                     \
        ((nz - 1) + blocksize_v) / blocksize_v);                                    \
    dim3 block = dim3(blocksize_h, blocksize_h, blocksize_v);                       \
                                                                                    \
    auto lambda_init =                                                              \
        GPU_LAMBDA(const int i, const int j, const int k)                           \
        {                                                                           \
            return k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                         \
            + (k * ny * nx + j * nx + i) * sx1 + i1;                                \
        };                                                                          \
    auto lambda_body =                                                              \
        GPU_LAMBDA(const int i, const int j, const int k, const int i1)             \
        loop_body;                                                                  \
                                                                                    \
    BoxKernelI1<<<grid, block>>>(                                                   \
        lambda_init, lambda_body, ix, iy, iz, nx, ny, nz);                          \
    CUDA_ERR( cudaPeekAtLastError() );                                              \
    CUDA_ERR( cudaDeviceSynchronize() );                                            \
  }                                                                                 \
}

#undef BoxLoopI2
#define BoxLoopI2(i_dummy, j_dummy, k_dummy,                                        \
                  ix, iy, iz, nx, ny, nz,                                           \
                  i1, nx1, ny1, nz1, sx1, sy1, sz1,                                 \
                  i2, nx2, ny2, nz2, sx2, sy2, sz2,                                 \
                  loop_body)                                                        \
{                                                                                   \
  if(nx > 0 && ny > 0 && nz > 0)                                                    \
  {                                                                                 \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);     \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2);     \
                                                                                    \
    const int blocksize_h = 16;                                                     \
    const int blocksize_v = 4;                                                      \
    dim3 grid = dim3(((nx - 1) + blocksize_h) / blocksize_h,                        \
        ((ny - 1) + blocksize_h) / blocksize_h,                                     \
        ((nz - 1) + blocksize_v) / blocksize_v);                                    \
    dim3 block = dim3(blocksize_h, blocksize_h, blocksize_v);                       \
                                                                                    \
    auto lambda_init1 =                                                             \
        GPU_LAMBDA(const int i, const int j, const int k)                           \
        {                                                                           \
            return k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                         \
            + (k * ny * nx + j * nx + i) * sx1 + i1;                                \
        };                                                                          \
    auto lambda_init2 =                                                             \
        GPU_LAMBDA(const int i, const int j, const int k)                           \
        {                                                                           \
            return k * PV_kinc_2 + (k * ny + j) * PV_jinc_2                         \
            + (k * ny * nx + j * nx + i) * sx2 + i2;                                \
        };                                                                          \
    auto lambda_body =                                                              \
        GPU_LAMBDA(const int i, const int j, const int k,                           \
        const int i1, const int i2)loop_body;                                       \
                                                                                    \
    BoxKernelI2<<<grid, block>>>(                                                   \
        lambda_init1, lambda_init2, lambda_body, ix, iy, iz, nx, ny, nz);           \
    CUDA_ERR( cudaPeekAtLastError() );                                              \
    CUDA_ERR( cudaDeviceSynchronize() );                                            \
  }                                                                                 \
}

#undef BoxLoopI3
#define BoxLoopI3(i_dummy, j_dummy, k_dummy,                                        \
                  ix, iy, iz, nx, ny, nz,                                           \
                  i1, nx1, ny1, nz1, sx1, sy1, sz1,                                 \
                  i2, nx2, ny2, nz2, sx2, sy2, sz2,                                 \
                  i3, nx3, ny3, nz3, sx3, sy3, sz3,                                 \
                  loop_body)                                                        \
{                                                                                   \
  if(nx > 0 && ny > 0 && nz > 0)                                                    \
  {                                                                                 \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);     \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2);     \
    DeclareInc(PV_jinc_3, PV_kinc_3, nx, ny, nz, nx3, ny3, nz3, sx3, sy3, sz3);     \
                                                                                    \
    const int blocksize_h = 16;                                                     \
    const int blocksize_v = 4;                                                      \
    dim3 grid = dim3(((nx - 1) + blocksize_h) / blocksize_h,                        \
        ((ny - 1) + blocksize_h) / blocksize_h,                                     \
        ((nz - 1) + blocksize_v) / blocksize_v);                                    \
    dim3 block = dim3(blocksize_h, blocksize_h, blocksize_v);                       \
                                                                                    \
    auto lambda_init1 =                                                             \
        GPU_LAMBDA(const int i, const int j, const int k)                           \
        {                                                                           \
            return k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                         \
            + (k * ny * nx + j * nx + i) * sx1 + i1;                                \
        };                                                                          \
    auto lambda_init2 =                                                             \
        GPU_LAMBDA(const int i, const int j, const int k)                           \
        {                                                                           \
            return k * PV_kinc_2 + (k * ny + j) * PV_jinc_2                         \
            + (k * ny * nx + j * nx + i) * sx2 + i2;                                \
        };                                                                          \
    auto lambda_init3 =                                                             \
        GPU_LAMBDA(const int i, const int j, const int k)                           \
        {                                                                           \
            return k * PV_kinc_3 + (k * ny + j) * PV_jinc_3                         \
            + (k * ny * nx + j * nx + i) * sx3 + i3;                                \
        };                                                                          \
    auto lambda_body =                                                              \
        GPU_LAMBDA(const int i, const int j, const int k,                           \
        const int i1, const int i2, const int i3)loop_body;                         \
                                                                                    \
    BoxKernelI3<<<grid, block>>>(lambda_init1, lambda_init2, lambda_init3,          \
        lambda_body, ix, iy, iz, nx, ny, nz);                                       \
    CUDA_ERR( cudaPeekAtLastError() );                                              \
    CUDA_ERR( cudaDeviceSynchronize() );                                            \
  }                                                                                 \
}

#undef GrGeomInLoopBoxes
#define GrGeomInLoopBoxes(i_dummy, j_dummy, k_dummy,                                \
                         grgeom, ix, iy, iz, nx, ny, nz, loop_body)                 \
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
    if(PV_ixl <= PV_ixu && PV_iyl <= PV_iyu && PV_izl <= PV_izu)                    \
    {                                                                               \
      int PV_diff_x = PV_ixu - PV_ixl;                                              \
      int PV_diff_y = PV_iyu - PV_iyl;                                              \
      int PV_diff_z = PV_izu - PV_izl;                                              \
                                                                                    \
      const int blocksize_h = 16;                                                   \
      const int blocksize_v = 4;                                                    \
      dim3 grid = dim3((PV_diff_x + blocksize_h) / blocksize_h,                     \
        (PV_diff_y + blocksize_h) / blocksize_h,                                    \
        (PV_diff_z + blocksize_v) / blocksize_v);                                   \
      dim3 block = dim3(blocksize_h, blocksize_h, blocksize_v);                     \
                                                                                    \
      int nx = PV_diff_x + 1;                                                       \
      int ny = PV_diff_y + 1;                                                       \
      int nz = PV_diff_z + 1;                                                       \
                                                                                    \
      BoxKernelI0<<<grid, block>>>(                                                 \
          GPU_LAMBDA(const int i, const int j, const int k)loop_body,               \
          PV_ixl, PV_iyl, PV_izl, nx, ny, nz);                                      \
      CUDA_ERR( cudaPeekAtLastError() );                                            \
      CUDA_ERR( cudaDeviceSynchronize() );                                          \
    }                                                                               \
  }                                                                                 \
}
}
#endif // PFCUDALOOPS_H
