#ifndef PFCUDALOOPS_H
#define PFCUDALOOPS_H

/*--------------------------------------------------------------------------
 * Include CUDA error handling header
 *--------------------------------------------------------------------------*/
#include "pfcudaerr.h"

extern "C++"{
/*--------------------------------------------------------------------------
 * CUDA block size definitions
 *--------------------------------------------------------------------------*/
#define BLOCKSIZE_H 16
#define BLOCKSIZE_V 4
#define BLOCKSIZE_TOTAL BLOCKSIZE_H * BLOCKSIZE_H * BLOCKSIZE_V

/*--------------------------------------------------------------------------
 * CUDA lambda definition (visible for host and device functions)
 *--------------------------------------------------------------------------*/
#define GPU_LAMBDA [=] __host__ __device__

/*--------------------------------------------------------------------------
 * CUDA helper functions
 *--------------------------------------------------------------------------*/
#define RAND48_SEED_0   (0x330e)
#define RAND48_SEED_1   (0xabcd)
#define RAND48_SEED_2   (0x1234)
#define RAND48_MULT_0   (0xe66d)
#define RAND48_MULT_1   (0xdeec)
#define RAND48_MULT_2   (0x0005)
#define RAND48_ADD      (0x000b)

__host__ __device__ __forceinline__ static void dev_dorand48(unsigned short xseed[3])
{
  unsigned long accu;
  
  unsigned short _rand48_mult[3] = {
    RAND48_MULT_0,
    RAND48_MULT_1,
    RAND48_MULT_2
  };
  unsigned short _rand48_add = RAND48_ADD;
  unsigned short temp[2];

  accu = (unsigned long)_rand48_mult[0] * (unsigned long)xseed[0] +
         (unsigned long)_rand48_add;
  temp[0] = (unsigned short)accu;               /* lower 16 bits */
  accu >>= sizeof(unsigned short) * 8;
  accu += (unsigned long)_rand48_mult[0] * (unsigned long)xseed[1] +
          (unsigned long)_rand48_mult[1] * (unsigned long)xseed[0];
  temp[1] = (unsigned short)accu;               /* middle 16 bits */
  accu >>= sizeof(unsigned short) * 8;
  accu += _rand48_mult[0] * xseed[2] + _rand48_mult[1] * xseed[1] + _rand48_mult[2] * xseed[0];
  xseed[0] = temp[0];
  xseed[1] = temp[1];
  xseed[2] = (unsigned short)accu;
}

__host__ __device__ __forceinline__ static double dev_erand48(unsigned short xseed[3])
{
  dev_dorand48(xseed);
  return ldexp((double)xseed[0], -48) +
         ldexp((double)xseed[1], -32) +
         ldexp((double)xseed[2], -16);
}

__host__ __device__ __forceinline__ static double dev_drand48(void)
{
  unsigned short _rand48_seed[3] = {
    RAND48_SEED_0,
    RAND48_SEED_1,
    RAND48_SEED_2
  };
  return dev_erand48(_rand48_seed);
}

__host__ __device__ __forceinline__ static void AtomicMin(int *address, int val2)
{
#ifdef __CUDA_ARCH__
    atomicMin(address, val2);
#else
    if(*address > val2) *address = val2;
#endif
}

__host__ __device__ __forceinline__ static void AtomicMax(int *address, int val2)
{
#ifdef __CUDA_ARCH__
    atomicMax(address, val2);
#else
    if(*address < val2) *address = val2;
#endif
}

__host__ __device__ __forceinline__ static void AtomicMin(double *address, double val2)
{
#ifdef __CUDA_ARCH__
    unsigned long long ret = __double_as_longlong(*address);
    while(val2 < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val2))) == old)
            break;
    }
    //return __longlong_as_double(ret);
#else
    if(*address > val2) *address = val2;
#endif
}

__host__ __device__ __forceinline__ static void AtomicMax(double *address, double val2)
{
#ifdef __CUDA_ARCH__
    unsigned long long ret = __double_as_longlong(*address);
    while(val2 > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val2))) == old)
            break;
    }
    //return __longlong_as_double(ret);
#else
    if(*address < val2) *address = val2;
#endif
}

template <typename T>
__host__ __device__ __forceinline__ static void AtomicAdd(T *array_loc, T value)
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
#undef pfmax_atomic
#define pfmax_atomic(a, b) AtomicMax(&(a), b)

#undef pfmin_atomic
#define pfmin_atomic(a, b) AtomicMin(&(a), b)

#undef PlusEquals
#define PlusEquals(a, b) AtomicAdd(&(a), b)

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

template <typename LAMBDA_INIT1, typename LAMBDA_INIT2, typename LAMBDA_FUN, typename T>
__global__ void DotKernelI2(LAMBDA_INIT1 loop_init1, LAMBDA_INIT2 loop_init2, LAMBDA_FUN loop_fun, 
    const T * __restrict__ a, const T * __restrict__ b, T * __restrict__ rslt,  
    const int ix, const int iy, const int iz, const int nx, const int ny, const int nz)
{
    __shared__ T temp[BLOCKSIZE_TOTAL];
    const int tid_local = blockDim.y * blockDim.x * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    temp[tid_local] = (T)0;

    int i = ((blockIdx.x*blockDim.x)+threadIdx.x);
    if(i >= nx)return;
    int j = ((blockIdx.y*blockDim.y)+threadIdx.y);
    if(j >= ny)return;
    int k = ((blockIdx.z*blockDim.z)+threadIdx.z);
    if(k >= nz)return;

    const int i1 = loop_init1(i, j, k);
    const int i2 = loop_init2(i, j, k);

    temp[tid_local] = loop_fun(a, b, i1, i2);

    __syncthreads(); 
    
    if(threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0) 
    {
      T sum = 0;
      for( int i = 0; i < BLOCKSIZE_TOTAL; i++ ) sum += temp[i];
      atomicAdd(rslt, sum);
    }
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
    const int blocksize_h = BLOCKSIZE_H;                                            \
    const int blocksize_v = BLOCKSIZE_V;                                            \
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
    const int blocksize_h = BLOCKSIZE_H;                                            \
    const int blocksize_v = BLOCKSIZE_V;                                            \
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
    const int blocksize_h = BLOCKSIZE_H;                                            \
    const int blocksize_v = BLOCKSIZE_V;                                            \
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

#undef DotLoopGPU
#define DotLoopGPU(i_dummy, j_dummy, k_dummy,                                       \
                  ix, iy, iz, nx, ny, nz,                                           \
                  i1, nx1, ny1, nz1, sx1, sy1, sz1,                                 \
                  i2, nx2, ny2, nz2, sx2, sy2, sz2,                                 \
                  xp, yp, rslt, lambda_fun)                                         \
{                                                                                   \
  if(nx > 0 && ny > 0 && nz > 0)                                                    \
  {                                                                                 \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);     \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2);     \
                                                                                    \
    const int blocksize_h = BLOCKSIZE_H;                                            \
    const int blocksize_v = BLOCKSIZE_V;                                            \
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
                                                                                    \
    DotKernelI2<<<grid, block>>>(lambda_init1, lambda_init2, lambda_fun,            \
        xp, yp, &rslt, ix, iy, iz, nx, ny, nz);                                     \
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
      const int blocksize_h = BLOCKSIZE_H;                                          \
      const int blocksize_v = BLOCKSIZE_V;                                          \
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

#undef GrGeomSurfLoopBoxes
#define GrGeomSurfLoopBoxes(i_dummy, j_dummy, k_dummy, fdir, grgeom,                \
                            ix, iy, iz, nx, ny, nz, loop_body)                      \
{                                                                                   \
  int PV_fdir[3];                                                                   \
  fdir = PV_fdir;                                                                   \
  for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)                           \
  {                                                                                 \
    switch (PV_f)                                                                   \
    {                                                                               \
    case GrGeomOctreeFaceL:                                                         \
        fdir[0] = -1; fdir[1] = 0; fdir[2] = 0;                                     \
        break;                                                                      \
    case GrGeomOctreeFaceR:                                                         \
        fdir[0] = 1; fdir[1] = 0; fdir[2] = 0;                                      \
        break;                                                                      \
    case GrGeomOctreeFaceD:                                                         \
        fdir[0] = 0; fdir[1] = -1; fdir[2] = 0;                                     \
        break;                                                                      \
    case GrGeomOctreeFaceU:                                                         \
        fdir[0] = 0; fdir[1] = 1; fdir[2] = 0;                                      \
        break;                                                                      \
    case GrGeomOctreeFaceB:                                                         \
        fdir[0] = 0; fdir[1] = 0; fdir[2] = -1;                                     \
        break;                                                                      \
    case GrGeomOctreeFaceF:                                                         \
        fdir[0] = 0; fdir[1] = 0; fdir[2] = 1;                                      \
        break;                                                                      \
    default:                                                                        \
        fdir[0] = -9999; fdir[1] = -9999; fdir[2] = -99999;                         \
        break;                                                                      \
    }                                                                               \
                                                                                    \
    BoxArray* boxes = GrGeomSolidSurfaceBoxes(grgeom, PV_f);                        \
    for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                    \
    {                                                                               \
      Box box = BoxArrayGetBox(boxes, PV_box);                                      \
      /* find octree and region intersection */                                     \
      int PV_ixl = pfmax(ix, box.lo[0]);                                            \
      int PV_iyl = pfmax(iy, box.lo[1]);                                            \
      int PV_izl = pfmax(iz, box.lo[2]);                                            \
      int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                                 \
      int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                                 \
      int PV_izu = pfmin((iz + nz - 1), box.up[2]);                                 \
                                                                                    \
      if(PV_ixl <= PV_ixu && PV_iyl <= PV_iyu && PV_izl <= PV_izu)                  \
      {                                                                             \
        int PV_diff_x = PV_ixu - PV_ixl;                                            \
        int PV_diff_y = PV_iyu - PV_iyl;                                            \
        int PV_diff_z = PV_izu - PV_izl;                                            \
                                                                                    \
        const int blocksize_h = BLOCKSIZE_H;                                        \
        const int blocksize_v = BLOCKSIZE_V;                                        \
        dim3 grid = dim3((PV_diff_x + blocksize_h) / blocksize_h,                   \
          (PV_diff_y + blocksize_h) / blocksize_h,                                  \
          (PV_diff_z + blocksize_v) / blocksize_v);                                 \
        dim3 block = dim3(blocksize_h, blocksize_h, blocksize_v);                   \
                                                                                    \
        int nx = PV_diff_x + 1;                                                     \
        int ny = PV_diff_y + 1;                                                     \
        int nz = PV_diff_z + 1;                                                     \
                                                                                    \
        const int fdir_capt0 = fdir[0];                                             \
        const int fdir_capt1 = fdir[1];                                             \
        const int fdir_capt2 = fdir[2];                                             \
                                                                                    \
        auto lambda_body =                                                          \
          GPU_LAMBDA(const int i, const int j, const int k)                         \
          {                                                                         \
            const int fdir[3] = {fdir_capt0, fdir_capt1, fdir_capt2};               \
            loop_body;                                                              \
          };                                                                        \
                                                                                    \
        BoxKernelI0<<<grid, block>>>(lambda_body,                                   \
            PV_ixl, PV_iyl, PV_izl, nx, ny, nz);                                    \
        CUDA_ERR( cudaPeekAtLastError() );                                          \
        CUDA_ERR( cudaDeviceSynchronize() );                                        \
      }                                                                             \
    }                                                                               \
  }                                                                                 \
}

#undef GrGeomPatchLoopBoxes
#define GrGeomPatchLoopBoxes(i_dummy, j_dummy, k_dummy, fdir, grgeom, patch_num,    \
                             ix, iy, iz, nx, ny, nz, loop_body)                     \
{                                                                                   \
  int PV_fdir[3];                                                                   \
  fdir = PV_fdir;                                                                   \
  for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)                           \
  {                                                                                 \
    switch (PV_f)                                                                   \
    {                                                                               \
      case GrGeomOctreeFaceL:                                                       \
        fdir[0] = -1; fdir[1] = 0; fdir[2] = 0;                                     \
        break;                                                                      \
      case GrGeomOctreeFaceR:                                                       \
        fdir[0] = 1; fdir[1] = 0; fdir[2] = 0;                                      \
        break;                                                                      \
      case GrGeomOctreeFaceD:                                                       \
        fdir[0] = 0; fdir[1] = -1; fdir[2] = 0;                                     \
        break;                                                                      \
      case GrGeomOctreeFaceU:                                                       \
        fdir[0] = 0; fdir[1] = 1; fdir[2] = 0;                                      \
        break;                                                                      \
      case GrGeomOctreeFaceB:                                                       \
        fdir[0] = 0; fdir[1] = 0; fdir[2] = -1;                                     \
        break;                                                                      \
      case GrGeomOctreeFaceF:                                                       \
        fdir[0] = 0; fdir[1] = 0; fdir[2] = 1;                                      \
        break;                                                                      \
      default:                                                                      \
        fdir[0] = -9999; fdir[1] = -9999; fdir[2] = -99999;                         \
        break;                                                                      \
    }                                                                               \
                                                                                    \
    int n_prev = 0;                                                                 \
    BoxArray* boxes = GrGeomSolidPatchBoxes(grgeom, patch_num, PV_f);               \
    for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                    \
    {                                                                               \
      Box box = BoxArrayGetBox(boxes, PV_box);                                      \
      /* find octree and region intersection */                                     \
      int PV_ixl = pfmax(ix, box.lo[0]);                                            \
      int PV_iyl = pfmax(iy, box.lo[1]);                                            \
      int PV_izl = pfmax(iz, box.lo[2]);                                            \
      int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                                 \
      int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                                 \
      int PV_izu = pfmin((iz + nz - 1), box.up[2]);                                 \
                                                                                    \
      if(PV_ixl <= PV_ixu && PV_iyl <= PV_iyu && PV_izl <= PV_izu)                  \
      {                                                                             \
        int PV_diff_x = PV_ixu - PV_ixl;                                            \
        int PV_diff_y = PV_iyu - PV_iyl;                                            \
        int PV_diff_z = PV_izu - PV_izl;                                            \
                                                                                    \
        const int blocksize_h = BLOCKSIZE_H;                                        \
        const int blocksize_v = BLOCKSIZE_V;                                        \
        dim3 grid = dim3((PV_diff_x + blocksize_h) / blocksize_h,                   \
          (PV_diff_y + blocksize_h) / blocksize_h,                                  \
          (PV_diff_z + blocksize_v) / blocksize_v);                                 \
        dim3 block = dim3(blocksize_h, blocksize_h, blocksize_v);                   \
                                                                                    \
        int nx = PV_diff_x + 1;                                                     \
        int ny = PV_diff_y + 1;                                                     \
        int nz = PV_diff_z + 1;                                                     \
                                                                                    \
        auto lambda_init =                                                          \
            GPU_LAMBDA(const int i, const int j, const int k)                       \
            {                                                                       \
                return n_prev + k * ny * nx + j * nx + i;                           \
            };                                                                      \
        n_prev += nz * ny * nx;                                                     \
                                                                                    \
        const int fdir_capt0 = fdir[0];                                             \
        const int fdir_capt1 = fdir[1];                                             \
        const int fdir_capt2 = fdir[2];                                             \
                                                                                    \
        auto lambda_body =                                                          \
             GPU_LAMBDA(const int i, const int j, const int k, int ival)            \
             {                                                                      \
               const int fdir[3] = {fdir_capt0, fdir_capt1, fdir_capt2};            \
               loop_body;                                                           \
             };                                                                     \
                                                                                    \
        BoxKernelI1<<<grid, block>>>(                                               \
            lambda_init, lambda_body, PV_ixl, PV_iyl, PV_izl, nx, ny, nz);          \
        CUDA_ERR( cudaPeekAtLastError() );                                          \
        CUDA_ERR( cudaDeviceSynchronize() );                                        \
      }                                                                             \
    }                                                                               \
  }                                                                                 \
}

}
#endif // PFCUDALOOPS_H
