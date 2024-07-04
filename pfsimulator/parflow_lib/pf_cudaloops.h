/**********************************************************************
 *
 *  Please read the LICENSE file for the GNU Lesser General Public License.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License (as published
 *  by the Free Software Foundation) version 2.1 dated February 1999.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
 *  and conditions of the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 *  USA
 ***********************************************************************/

/* @file
 * @brief Contains macros, functions, and structs for CUDA compute kernels.
 */

#ifndef PF_CUDALOOPS_H
#define PF_CUDALOOPS_H

/*--------------------------------------------------------------------------
 * Include CUDA headers
 *--------------------------------------------------------------------------*/
#include "pf_devices.h"
#include "pf_cudamalloc.h"

extern "C++"{

#include <tuple>
#include <cub/cub.cuh>

/*--------------------------------------------------------------------------
 * CUDA blocksize definitions
 *--------------------------------------------------------------------------*/

/**
 * The largest blocksize ParFlow is using, but also the largest blocksize 
 * supported by any currently available NVIDIA GPU architecture. This can 
 * also differ between different architectures. It is used for informing 
 * the compiler about how many registers should be available for the GPU 
 * kernel during the compilation. Another option is to use 
 * --maxrregcount 64 compiler flag, but NVIDIA recommends specifying 
 * this kernel-by-kernel basis by __launch_bounds__() identifier.
 */
#define BLOCKSIZE_MAX 1024

/**
 * The blocksize for the x-dimension. This is is set to 32, 
 * because the warp size for the current NVIDIA architectures is 32. 
 * Therefore, letting each thread in a warp access consecutive memory 
 * locations along the x-dimension results in best memory coalescence. 
 * It is also important that the total blocksize (the product of x, y, 
 * and z-blocksizes) is divisible by the warp size (32).
 */
#define BLOCKSIZE_X 32

/**
 * The default blocksize for the y-dimension. Blocksizes along y and 
 * z-dimensions are less important compared to the x-dimension. 
 */
#define BLOCKSIZE_Y 8

/**
 * The default blocksize for the z-dimension. Blocksizes along y and 
 * z-dimensions are less important compared to the x-dimension. 
 */
#define BLOCKSIZE_Z 4

/*--------------------------------------------------------------------------
 * CUDA lambda definition (visible for host and device functions)
 *--------------------------------------------------------------------------*/
#define GPU_LAMBDA [=] __host__ __device__

/*--------------------------------------------------------------------------
 * CUDA helper macros and functions
 *--------------------------------------------------------------------------*/
#define RAND48_SEED_0   (0x330e)
#define RAND48_SEED_1   (0xabcd)
#define RAND48_SEED_2   (0x1234)
#define RAND48_MULT_0   (0xe66d)
#define RAND48_MULT_1   (0xdeec)
#define RAND48_MULT_2   (0x0005)
#define RAND48_ADD      (0x000b)

 /** Helper struct for type comparison. @note Not for direct use! */
template <typename T>
struct function_traits
    : public function_traits<decltype(&T::operator())>
{};
// For generic types, directly use the result of the signature of its 'operator()'

 /** Helper struct for type comparison. @note Not for direct use! */
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const>
// we specialize for pointers to member function
{
    enum { arity = sizeof...(Args) };
    // arity is the number of arguments.

    typedef ReturnType result_type;

    template <size_t i>
    struct arg
    {
        typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
        // the i-th argument is equivalent to the i-th tuple element of a tuple
        // composed of those arguments.
    };
};

/** Device-callable dorand48() function for CUDA compute kernels. */
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

/** Device-callable erand48() function for CUDA compute kernels. */
__host__ __device__ __forceinline__ static double dev_erand48(unsigned short xseed[3])
{
  dev_dorand48(xseed);
  return ldexp((double)xseed[0], -48) +
         ldexp((double)xseed[1], -32) +
         ldexp((double)xseed[2], -16);
}

/** Device-callable drand48() function for CUDA compute kernels. */
__host__ __device__ __forceinline__ static double dev_drand48(void)
{
  unsigned short _rand48_seed[3] = {
    RAND48_SEED_0,
    RAND48_SEED_1,
    RAND48_SEED_2
  };
  return dev_erand48(_rand48_seed);
}

/** Device-callable RPowerR() function for CUDA compute kernels. */
template <typename T>
__host__ __device__ __forceinline__ static T RPowerR(T base, T exponent)
{
  if (base <= 0.0)
    return(0.0);

  return((T)pow((double)base, (double)exponent));
}

/**
 * Thread-safe function to find minimum value in compute kernels.
 * The function definition depends whether called from host or device code.
 *
 * @note Not for direct use!
 *
 * @param address pointer to first value [IN], pointer to min value [OUT]
 * @param val2 second value [IN]
 */
__host__ __device__ __forceinline__ static void AtomicMin(int *address, int val2)
{
#ifdef __CUDA_ARCH__
    atomicMin(address, val2);
#else
    if(*address > val2) *address = val2;
#endif
}

/**
 * Thread-safe function to find maximum value in compute kernels.
 * The function definition depends whether called from host or device code.
 *
 * @note Not for direct use!
 *
 * @param address pointer to first value [IN], pointer to max value [OUT]
 * @param val2 second value [IN]
 */
__host__ __device__ __forceinline__ static void AtomicMax(int *address, int val2)
{
#ifdef __CUDA_ARCH__
    atomicMax(address, val2);
#else
    if(*address < val2) *address = val2;
#endif
}

/**
 * Thread-safe function to find minimum value in compute kernels.
 * The function definition depends whether called from host or device code.
 *
 * @note Not for direct use!
 *
 * @param address pointer to first value [IN], pointer to min value [OUT]
 * @param val2 second value [IN]
 */
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

/**
 * Thread-safe function to find maximum value in compute kernels.
 * The function definition depends whether called from host or device code.
 *
 * @note Not for direct use!
 *
 * @param address pointer to first value [IN], pointer to max value [OUT]
 * @param val2 second value [IN]
 */
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

/**
 * Thread-safe addition assignment for compute kernels.
 * The function definition depends whether called from host or device code.
 *
 * @note Not for direct use!
 *
 * @param array_loc original value [IN], sum result [OUT]
 * @param value value to be added [IN]
 */
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

 /** Helper struct for type comparison. @note Not for direct use! */
struct SkipParallelSync {const int dummy = 0;};
#define SKIP_PARALLEL_SYNC_cuda struct SkipParallelSync sync_struct; return sync_struct;

#define PARALLEL_SYNC_cuda CUDA_ERR(cudaStreamSynchronize(0)); 

#define PlusEquals_cuda(a, b) AtomicAdd(&(a), b)

 /** Helper struct for type comparison. @note Not for direct use! */
template <typename T>
struct ReduceMaxType {T value;};
#define ReduceMax_cuda(a, b) struct ReduceMaxType<std::decay<decltype(a)>::type> reduce_struct {.value = b}; return reduce_struct;

 /** Helper struct for type comparison. @note Not for direct use! */
template <typename T>
struct ReduceMinType {T value;};
#define ReduceMin_cuda(a, b) struct ReduceMinType<std::decay<decltype(a)>::type> reduce_struct {.value = b}; return reduce_struct;

 /** Helper struct for type comparison. @note Not for direct use! */
template <typename T>
struct ReduceSumType {T value;};
#define ReduceSum_cuda(a, b) struct ReduceSumType<std::decay<decltype(a)>::type> reduce_struct {.value = b}; return reduce_struct;

/** A constant table for fdir (borrowed from OMP backend) */
static const int FDIR_TABLE[6][3] = {
  {-1, 0,  0}, // Left
  {1,  0,  0}, // Right
  {0, -1,  0}, // Down
  {0,  1,  0}, // Up
  {0,  0, -1}, // Back
  {0,  0,  1}, // Front
};

/*--------------------------------------------------------------------------
 * CUDA loop kernels
 *--------------------------------------------------------------------------*/

/**
 * @brief CUDA basic compute kernel.
 *
 * @param loop_body A lambda function that evaluates the loop body [IN/OUT]
 * @param nx The size of the first dim [IN]
 * @param ny The size of the second dim [IN]
 * @param nz The size of the third dim [IN]
 */
template <typename LambdaBody>
__global__ static void 
__launch_bounds__(BLOCKSIZE_MAX)
BoxKernel(LambdaBody loop_body, const int nx, const int ny, const int nz)
{

    const int i = ((blockIdx.x*blockDim.x)+threadIdx.x);
    if(i >= nx)return;
    const int j = ((blockIdx.y*blockDim.y)+threadIdx.y);
    if(j >= ny)return;
    const int k = ((blockIdx.z*blockDim.z)+threadIdx.z);
    if(k >= nz)return;
    
    loop_body(i, j, k);
}

/**
 * @brief CUDA compute kernel for parallel reductions.
 *
 * @param loop_fun A lambda function that evaluates the loop body [IN/OUT]
 * @param init_val The initial value of the reduction variable [IN]
 * @param rslt A pointer to the result/sum variable [OUT]
 * @param nx The size of the first dim [IN]
 * @param ny The size of the second dim [IN]
 * @param nz The size of the third dim [IN]
 */
template <typename ReduceOp, typename LambdaFun, typename T>
__global__ static void 
__launch_bounds__(BLOCKSIZE_MAX)
DotKernel(LambdaFun loop_fun, const T init_val, T * __restrict__ rslt, 
  const int nx, const int ny, const int nz)
{
    // Specialize BlockReduce for a 1D block of BLOCKSIZE_X * 1 * 1 threads on type T
#ifdef __CUDA_ARCH__
    typedef cub::BlockReduce<T, BLOCKSIZE_MAX, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 1, 1, __CUDA_ARCH__> BlockReduce;
#else
    typedef cub::BlockReduce<T, BLOCKSIZE_MAX, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 1, 1> BlockReduce;
#endif

    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int idx = ((blockIdx.x*blockDim.x)+threadIdx.x);

    const int i = idx % nx;
    const int j = (idx / nx) % ny;
    const int k = idx / (nx * ny);
    const int ntot = nx * ny * nz;

    T thread_data;
    
    // Initialize thread_data depending on reduction operation
    if(std::is_same<ReduceOp, struct ReduceSumType<T>>::value)
      thread_data = 0;
    else 
      thread_data = init_val;

    // Evaluate the loop body
    if (idx < ntot)
      thread_data = loop_fun(i, j, k).value;

    // Perform reductions
    if(std::is_same<ReduceOp, struct ReduceSumType<T>>::value)
    {
      // Compute the block-wide sum for thread0
      T aggregate = BlockReduce(temp_storage).Sum(thread_data);

      // Store aggregate
      if(threadIdx.x == 0) 
      {
        atomicAdd(rslt, aggregate);
      }
    }
    else if(std::is_same<ReduceOp, struct ReduceMaxType<T>>::value)
    {
      // Compute the block-wide sum for thread0
      T aggregate = BlockReduce(temp_storage).Reduce(thread_data, cub::Max());

      // Store aggregate
      if(threadIdx.x == 0) 
      {
        AtomicMax(rslt, aggregate);
      }

      // Write to global memory directly from all threads
      // if (idx < ntot)
      // {
      //   AtomicMax(rslt, thread_data);
      // }
    }
    else if(std::is_same<ReduceOp, struct ReduceMinType<T>>::value)
    {
      // Compute the block-wide sum for thread0
      T aggregate = BlockReduce(temp_storage).Reduce(thread_data, cub::Min());

      // Store aggregate
      if(threadIdx.x == 0) 
      {
        AtomicMin(rslt, aggregate);
      }
      
      // Write to global memory directly from all threads
      // if (idx < ntot)
      // {
      //   AtomicMin(rslt, thread_data);
      // }
    }
    else 
    {
      printf("ERROR at %s:%d: Invalid reduction identifier, likely a problem with a BoxLoopReduce body.", __FILE__, __LINE__);
    }
}

/*--------------------------------------------------------------------------
 * CUDA loop macro redefinitions
 *--------------------------------------------------------------------------*/

/**
 * @brief A macro for finding the 3D CUDA grid and block dimensions.
 * 
 * The runtime adjustment of y and z-blocksizes is based on a heuristic 
 * to improve the occupancy when launching a kernel with small grid size 
 * (does not have a very significant performance implications).
 *
 * @note Not for direct use!
 *
 * @param grid grid dimensions [OUT]
 * @param block block dimensions [OUT]
 * @param nx The size of the first dim [IN]
 * @param ny The size of the second dim [IN]
 * @param nz The size of the third dim [IN]
 * @param dyn_blocksize Runtime adjustment of y- and z-blocksizes [IN]
 */
#define FindDims(grid, block, nx, ny, nz, dyn_blocksize)                            \
{                                                                                   \
  int blocksize_x = BLOCKSIZE_X;                                                    \
  int blocksize_y = BLOCKSIZE_Y;                                                    \
  int blocksize_z = BLOCKSIZE_Z;                                                    \
  grid = dim3(((nx - 1) + blocksize_x) / blocksize_x,                               \
      ((ny - 1) + blocksize_y) / blocksize_y,                                       \
      ((nz - 1) + blocksize_z) / blocksize_z);                                      \
  while(dyn_blocksize && (grid.x*grid.y*grid.z < 80)                                \
          && ((blocksize_y * blocksize_z) >= 4 ))                                   \
  {                                                                                 \
      if ( blocksize_z >= 2 )                                                       \
          blocksize_z /= 2;                                                         \
      else                                                                          \
          blocksize_y /= 2;                                                         \
      grid = dim3(((nx - 1) + blocksize_x) / blocksize_x,                           \
      ((ny - 1) + blocksize_y) / blocksize_y,                                       \
      ((nz - 1) + blocksize_z) / blocksize_z);                                      \
  }                                                                                 \
  block = dim3(blocksize_x, blocksize_y, blocksize_z);                              \
}

/**
 * @brief A macro for checking if cell flag data array must be reallocated.
 *
 * @note Not for direct use!
 *
 * @param grgeom The geometry details [IN/OUT]
 * @param nx The size of the first dim [IN]
 * @param ny The size of the second dim [IN]
 * @param nz The size of the third dim [IN]
 */
#define CheckCellFlagAllocation(grgeom, nx, ny, nz)                                 \
{                                                                                   \
  int flagdata_size = sizeof(char) * (nz * ny * nx);                                \
  if(GrGeomSolidCellFlagDataSize(grgeom) < flagdata_size)                           \
  {                                                                                 \
    char *flagdata = (char*)_ctalloc_device(flagdata_size);                         \
                                                                                    \
    if(GrGeomSolidCellFlagDataSize(grgeom) > 0)                                     \
      CUDA_ERR(cudaMemcpy(flagdata, GrGeomSolidCellFlagData(grgeom),                \
        GrGeomSolidCellFlagDataSize(grgeom), cudaMemcpyDeviceToDevice));            \
                                                                                    \
    _tfree_device(GrGeomSolidCellFlagData(grgeom));                                 \
    GrGeomSolidCellFlagData(grgeom) = flagdata;                                     \
    GrGeomSolidCellFlagDataSize(grgeom) = flagdata_size;                            \
  }                                                                                 \
}

 /** Loop definition for CUDA. */
#define BoxLoopI1_cuda(i, j, k,                                                     \
  ix, iy, iz, nx, ny, nz,                                                           \
  i1, nx1, ny1, nz1, sx1, sy1, sz1,                                                 \
  loop_body)                                                                        \
{                                                                                   \
  if(nx > 0 && ny > 0 && nz > 0)                                                    \
  {                                                                                 \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);     \
                                                                                    \
    dim3 block, grid;                                                               \
    FindDims(grid, block, nx, ny, nz, 1);                                           \
                                                                                    \
    const auto &ref_i1 = i1;                                                        \
                                                                                    \
    auto lambda_body =                                                              \
      GPU_LAMBDA(int i, int j, int k)                                               \
      {                                                                             \
        const int i1 = k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                     \
          + (k * ny * nx + j * nx + i) * sx1 + ref_i1;                              \
                                                                                    \
        i += ix;                                                                    \
        j += iy;                                                                    \
        k += iz;                                                                    \
                                                                                    \
        loop_body;                                                                  \
      };                                                                            \
                                                                                    \
    BoxKernel<<<grid, block>>>(lambda_body, nx, ny, nz);                            \
    CUDA_ERR(cudaPeekAtLastError());                                                \
                                                                                    \
    typedef function_traits<decltype(lambda_body)> traits;                          \
    if(!std::is_same<traits::result_type, struct SkipParallelSync>::value)          \
      CUDA_ERR(cudaStreamSynchronize(0));                                           \
  }                                                                                 \
  (void)i;(void)j;(void)k;                                                          \
}

 /** Loop definition for CUDA. */
#define BoxLoopI2_cuda(i, j, k,                                                     \
  ix, iy, iz, nx, ny, nz,                                                           \
  i1, nx1, ny1, nz1, sx1, sy1, sz1,                                                 \
  i2, nx2, ny2, nz2, sx2, sy2, sz2,                                                 \
  loop_body)                                                                        \
{                                                                                   \
  if(nx > 0 && ny > 0 && nz > 0)                                                    \
  {                                                                                 \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);     \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2);     \
                                                                                    \
    dim3 block, grid;                                                               \
    FindDims(grid, block, nx, ny, nz, 1);                                           \
                                                                                    \
    const auto &ref_i1 = i1;                                                        \
    const auto &ref_i2 = i2;                                                        \
                                                                                    \
    auto lambda_body =                                                              \
      GPU_LAMBDA(int i, int j, int k)                                               \
      {                                                                             \
        const int i1 = k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                     \
          + (k * ny * nx + j * nx + i) * sx1 + ref_i1;                              \
        const int i2 = k * PV_kinc_2 + (k * ny + j) * PV_jinc_2                     \
          + (k * ny * nx + j * nx + i) * sx2 + ref_i2;                              \
                                                                                    \
        i += ix;                                                                    \
        j += iy;                                                                    \
        k += iz;                                                                    \
                                                                                    \
        loop_body;                                                                  \
      };                                                                            \
                                                                                    \
    BoxKernel<<<grid, block>>>(lambda_body, nx, ny, nz);                            \
    CUDA_ERR(cudaPeekAtLastError());                                                \
                                                                                    \
    typedef function_traits<decltype(lambda_body)> traits;                          \
    if(!std::is_same<traits::result_type, struct SkipParallelSync>::value)          \
      CUDA_ERR(cudaStreamSynchronize(0));                                           \
  }                                                                                 \
  (void)i;(void)j;(void)k;                                                          \
}

 /** Loop definition for CUDA. */
#define BoxLoopI3_cuda(i, j, k,                                                     \
  ix, iy, iz, nx, ny, nz,                                                           \
  i1, nx1, ny1, nz1, sx1, sy1, sz1,                                                 \
  i2, nx2, ny2, nz2, sx2, sy2, sz2,                                                 \
  i3, nx3, ny3, nz3, sx3, sy3, sz3,                                                 \
  loop_body)                                                                        \
{                                                                                   \
  if(nx > 0 && ny > 0 && nz > 0)                                                    \
  {                                                                                 \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);     \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2);     \
    DeclareInc(PV_jinc_3, PV_kinc_3, nx, ny, nz, nx3, ny3, nz3, sx3, sy3, sz3);     \
                                                                                    \
    dim3 block, grid;                                                               \
    FindDims(grid, block, nx, ny, nz, 1);                                           \
                                                                                    \
    const auto &ref_i1 = i1;                                                        \
    const auto &ref_i2 = i2;                                                        \
    const auto &ref_i3 = i3;                                                        \
                                                                                    \
    auto lambda_body =                                                              \
      GPU_LAMBDA(int i, int j, int k)                                               \
      {                                                                             \
        const int i1 = k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                     \
          + (k * ny * nx + j * nx + i) * sx1 + ref_i1;                              \
        const int i2 = k * PV_kinc_2 + (k * ny + j) * PV_jinc_2                     \
          + (k * ny * nx + j * nx + i) * sx2 + ref_i2;                              \
        const int i3 = k * PV_kinc_3 + (k * ny + j) * PV_jinc_3                     \
          + (k * ny * nx + j * nx + i) * sx3 + ref_i3;                              \
                                                                                    \
        i += ix;                                                                    \
        j += iy;                                                                    \
        k += iz;                                                                    \
                                                                                    \
        loop_body;                                                                  \
      };                                                                            \
                                                                                    \
    BoxKernel<<<grid, block>>>(lambda_body, nx, ny, nz);                            \
    CUDA_ERR(cudaPeekAtLastError());                                                \
                                                                                    \
    typedef function_traits<decltype(lambda_body)> traits;                          \
    if(!std::is_same<traits::result_type, struct SkipParallelSync>::value)          \
      CUDA_ERR(cudaStreamSynchronize(0));                                           \
  }                                                                                 \
  (void)i;(void)j;(void)k;                                                          \
}

 /** Loop definition for CUDA. */
#define BoxLoopReduceI1_cuda(rslt, i, j, k,                                         \
  ix, iy, iz, nx, ny, nz,                                                           \
  i1, nx1, ny1, nz1, sx1, sy1, sz1,                                                 \
  loop_body)                                                                        \
{                                                                                   \
  if(nx > 0 && ny > 0 && nz > 0)                                                    \
  {                                                                                 \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);     \
                                                                                    \
    int block = BLOCKSIZE_MAX;                                                      \
    int grid = ((nx * ny * nz - 1) + block) / block;                                \
                                                                                    \
    const auto &ref_rslt = rslt;                                                    \
    const auto &ref_i1 = i1;                                                        \
                                                                                    \
    auto lambda_body =                                                              \
      GPU_LAMBDA(int i, int j, int k)                                               \
      {                                                                             \
        auto rslt = ref_rslt;                                                       \
        const int i1 = k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                     \
          + (k * ny * nx + j * nx + i) * sx1 + ref_i1;                              \
                                                                                    \
        i += ix;                                                                    \
        j += iy;                                                                    \
        k += iz;                                                                    \
                                                                                    \
        loop_body;                                                                  \
      };                                                                            \
                                                                                    \
    decltype(rslt)*ptr_rslt = (decltype(rslt)*)_talloc_device(sizeof(decltype(rslt)));\
    MemPrefetchDeviceToHost_cuda(ptr_rslt, sizeof(decltype(rslt)), 0);              \
    *ptr_rslt = rslt;                                                               \
    MemPrefetchHostToDevice_cuda(ptr_rslt, sizeof(decltype(rslt)), 0);              \
                                                                                    \
    typedef function_traits<decltype(lambda_body)> traits;                          \
    DotKernel<traits::result_type><<<grid, block>>>(lambda_body,                    \
      rslt, ptr_rslt, nx, ny, nz);                                                  \
    CUDA_ERR(cudaPeekAtLastError());                                                \
    CUDA_ERR(cudaStreamSynchronize(0));                                             \
                                                                                    \
    MemPrefetchDeviceToHost_cuda(ptr_rslt, sizeof(decltype(rslt)), 0);              \
    rslt = *ptr_rslt;                                                               \
    _tfree_device(ptr_rslt);                                                        \
  }                                                                                 \
  (void)i;(void)j;(void)k;                                                          \
}

 /** Loop definition for CUDA. */
#define BoxLoopReduceI2_cuda(rslt, i, j, k,                                         \
  ix, iy, iz, nx, ny, nz,                                                           \
  i1, nx1, ny1, nz1, sx1, sy1, sz1,                                                 \
  i2, nx2, ny2, nz2, sx2, sy2, sz2,                                                 \
  loop_body)                                                                        \
{                                                                                   \
  if(nx > 0 && ny > 0 && nz > 0)                                                    \
  {                                                                                 \
    DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);     \
    DeclareInc(PV_jinc_2, PV_kinc_2, nx, ny, nz, nx2, ny2, nz2, sx2, sy2, sz2);     \
                                                                                    \
    int block = BLOCKSIZE_MAX;                                                      \
    int grid = ((nx * ny * nz - 1) + block) / block;                                \
                                                                                    \
    const auto &ref_rslt = rslt;                                                    \
    const auto &ref_i1 = i1;                                                        \
    const auto &ref_i2 = i2;                                                        \
                                                                                    \
    auto lambda_body =                                                              \
      GPU_LAMBDA(int i, int j, int k)                                               \
      {                                                                             \
        auto rslt = ref_rslt;                                                       \
        const int i1 = k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                     \
          + (k * ny * nx + j * nx + i) * sx1 + ref_i1;                              \
        const int i2 = k * PV_kinc_2 + (k * ny + j) * PV_jinc_2                     \
          + (k * ny * nx + j * nx + i) * sx2 + ref_i2;                              \
                                                                                    \
        i += ix;                                                                    \
        j += iy;                                                                    \
        k += iz;                                                                    \
                                                                                    \
        loop_body;                                                                  \
      };                                                                            \
                                                                                    \
    decltype(rslt)*ptr_rslt = (decltype(rslt)*)_talloc_device(sizeof(decltype(rslt)));\
    MemPrefetchDeviceToHost_cuda(ptr_rslt, sizeof(decltype(rslt)), 0);              \
    *ptr_rslt = rslt;                                                               \
    MemPrefetchHostToDevice_cuda(ptr_rslt, sizeof(decltype(rslt)), 0);              \
                                                                                    \
    typedef function_traits<decltype(lambda_body)> traits;                          \
    DotKernel<traits::result_type><<<grid, block>>>(lambda_body,                    \
      rslt, ptr_rslt, nx, ny, nz);                                                  \
    CUDA_ERR(cudaPeekAtLastError());                                                \
    CUDA_ERR(cudaStreamSynchronize(0));                                             \
                                                                                    \
    MemPrefetchDeviceToHost_cuda(ptr_rslt, sizeof(decltype(rslt)), 0);              \
    rslt = *ptr_rslt;                                                               \
    _tfree_device(ptr_rslt);                                                        \
  }                                                                                 \
  (void)i;(void)j;(void)k;                                                          \
}

 /** Loop definition for CUDA. */
#define GrGeomInLoopBoxes_cuda(i, j, k,                                             \
  grgeom, ix, iy, iz, nx, ny, nz, loop_body)                                        \
{                                                                                   \
  BoxArray* boxes = GrGeomSolidInteriorBoxes(grgeom);                               \
  int ix_bxs = BoxArrayMinCell(boxes, 0);                                           \
  int iy_bxs = BoxArrayMinCell(boxes, 1);                                           \
  int iz_bxs = BoxArrayMinCell(boxes, 2);                                           \
                                                                                    \
  int nx_bxs = BoxArrayMaxCell(boxes, 0) - ix_bxs + 1;                              \
  int ny_bxs = BoxArrayMaxCell(boxes, 1) - iy_bxs + 1;                              \
  int nz_bxs = BoxArrayMaxCell(boxes, 2) - iz_bxs + 1;                              \
                                                                                    \
  if(!(GrGeomSolidCellFlagInitialized(grgeom) & 1))                                 \
  {                                                                                 \
    CheckCellFlagAllocation(grgeom, nx_bxs, ny_bxs, nz_bxs);                        \
    char *inflag = GrGeomSolidCellFlagData(grgeom);                                 \
                                                                                    \
    for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                    \
    {                                                                               \
      Box box = BoxArrayGetBox(boxes, PV_box);                                      \
      int PV_ixl = box.lo[0];                                                       \
      int PV_iyl = box.lo[1];                                                       \
      int PV_izl = box.lo[2];                                                       \
      int PV_ixu = box.up[0];                                                       \
      int PV_iyu = box.up[1];                                                       \
      int PV_izu = box.up[2];                                                       \
                                                                                    \
      if(PV_ixl <= PV_ixu && PV_iyl <= PV_iyu && PV_izl <= PV_izu)                  \
      {                                                                             \
        int PV_nx = PV_ixu - PV_ixl + 1;                                            \
        int PV_ny = PV_iyu - PV_iyl + 1;                                            \
        int PV_nz = PV_izu - PV_izl + 1;                                            \
                                                                                    \
        dim3 block, grid;                                                           \
        FindDims(grid, block, PV_nx, PV_ny, PV_nz, 1);                              \
                                                                                    \
        Globals *globals = ::globals;                                               \
        auto lambda_body =                                                          \
          GPU_LAMBDA(int i, int j, int k)                                           \
          {                                                                         \
            i += PV_ixl;                                                            \
            j += PV_iyl;                                                            \
            k += PV_izl;                                                            \
                                                                                    \
            /* Set inflag for all cells in boxes regardless of loop limits */       \
            inflag[(k - iz_bxs) * ny_bxs * nx_bxs +                                 \
              (j - iy_bxs) * nx_bxs + (i - ix_bxs)] |= 1;                           \
                                                                                    \
            /* Only evaluate loop body if the cell is within loop limits */         \
            if(i >= ix && j >= iy && k >= iz &&                                     \
              i < ix + nx && j < iy + ny && k < iz + nz)                            \
            {                                                                       \
              loop_body;                                                            \
            }                                                                       \
          };                                                                        \
                                                                                    \
        BoxKernel<<<grid, block>>>(lambda_body, PV_nx, PV_ny, PV_nz);               \
        CUDA_ERR(cudaPeekAtLastError());                                            \
        CUDA_ERR(cudaStreamSynchronize(0));                                         \
      }                                                                             \
    }                                                                               \
    GrGeomSolidCellFlagInitialized(grgeom) |= 1;                                    \
  }                                                                                 \
  else                                                                              \
  {                                                                                 \
    int ixl_gpu = pfmax(ix, ix_bxs);                                                \
    int iyl_gpu = pfmax(iy, iy_bxs);                                                \
    int izl_gpu = pfmax(iz, iz_bxs);                                                \
    int nx_gpu = pfmin((ix + nx - 1), BoxArrayMaxCell(boxes, 0)) - ixl_gpu + 1;     \
    int ny_gpu = pfmin((iy + ny - 1), BoxArrayMaxCell(boxes, 1)) - iyl_gpu + 1;     \
    int nz_gpu = pfmin((iz + nz - 1), BoxArrayMaxCell(boxes, 2)) - izl_gpu + 1;     \
                                                                                    \
    if(nx_gpu > 0 && ny_gpu > 0 && nz_gpu > 0)                                      \
    {                                                                               \
      dim3 block, grid;                                                             \
      FindDims(grid, block, nx_gpu, ny_gpu, nz_gpu, 1);                             \
                                                                                    \
      Globals *globals = ::globals;                                                 \
      char *inflag = GrGeomSolidCellFlagData(grgeom);                               \
      auto lambda_body =                                                            \
        GPU_LAMBDA(int i, int j, int k)                                             \
        {                                                                           \
          i += ixl_gpu;                                                             \
          j += iyl_gpu;                                                             \
          k += izl_gpu;                                                             \
          if(inflag[(k - iz_bxs) * ny_bxs * nx_bxs +                                \
            (j - iy_bxs) * nx_bxs + (i - ix_bxs)] & 1)                              \
          {                                                                         \
            loop_body;                                                              \
          }                                                                         \
        };                                                                          \
                                                                                    \
      BoxKernel<<<grid, block>>>(lambda_body, nx_gpu, ny_gpu, nz_gpu);              \
      CUDA_ERR(cudaPeekAtLastError());                                              \
      CUDA_ERR(cudaStreamSynchronize(0));                                           \
    }                                                                               \
  }                                                                                 \
  (void)i;(void)j;(void)k;                                                          \
}

 /** Loop definition for CUDA. */
#define GrGeomSurfLoopBoxes_cuda(i, j, k, fdir, grgeom,                             \
  ix, iy, iz, nx, ny, nz, loop_body)                                                \
{                                                                                   \
  for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)                           \
  {                                                                                 \
    const int *fdir = FDIR_TABLE[PV_f];                                             \
                                                                                    \
    BoxArray* boxes = GrGeomSolidSurfaceBoxes(grgeom, PV_f);                        \
    int ix_bxs = BoxArrayMinCell(boxes, 0);                                         \
    int iy_bxs = BoxArrayMinCell(boxes, 1);                                         \
    int iz_bxs = BoxArrayMinCell(boxes, 2);                                         \
                                                                                    \
    int nx_bxs = BoxArrayMaxCell(boxes, 0) - ix_bxs + 1;                            \
    int ny_bxs = BoxArrayMaxCell(boxes, 1) - iy_bxs + 1;                            \
    int nz_bxs = BoxArrayMaxCell(boxes, 2) - iz_bxs + 1;                            \
                                                                                    \
    if(!(GrGeomSolidCellFlagInitialized(grgeom) & (1 << (2 + PV_f))))               \
    {                                                                               \
      CheckCellFlagAllocation(grgeom, nx_bxs, ny_bxs, nz_bxs);                      \
      char *surfflag = GrGeomSolidCellFlagData(grgeom);                             \
                                                                                    \
      for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                  \
      {                                                                             \
        Box box = BoxArrayGetBox(boxes, PV_box);                                    \
        int PV_ixl = box.lo[0];                                                     \
        int PV_iyl = box.lo[1];                                                     \
        int PV_izl = box.lo[2];                                                     \
        int PV_ixu = box.up[0];                                                     \
        int PV_iyu = box.up[1];                                                     \
        int PV_izu = box.up[2];                                                     \
                                                                                    \
        if(PV_ixl <= PV_ixu && PV_iyl <= PV_iyu && PV_izl <= PV_izu)                \
        {                                                                           \
          int PV_nx = PV_ixu - PV_ixl + 1;                                          \
          int PV_ny = PV_iyu - PV_iyl + 1;                                          \
          int PV_nz = PV_izu - PV_izl + 1;                                          \
                                                                                    \
          dim3 block, grid;                                                         \
          FindDims(grid, block, PV_nx, PV_ny, PV_nz, 1);                            \
                                                                                    \
          const int _fdir0 = fdir[0];                                               \
          const int _fdir1 = fdir[1];                                               \
          const int _fdir2 = fdir[2];                                               \
                                                                                    \
          auto lambda_body =                                                        \
            GPU_LAMBDA(int i, int j, int k)                                         \
            {                                                                       \
              i += PV_ixl;                                                          \
              j += PV_iyl;                                                          \
              k += PV_izl;                                                          \
                                                                                    \
              /* Set surfflag for all cells in boxes regardless of loop limits */   \
              surfflag[(k - iz_bxs) * ny_bxs * nx_bxs +                             \
                (j - iy_bxs) * nx_bxs + (i - ix_bxs)] |= (1 << (2 + PV_f));         \
                                                                                    \
              /* Only evaluate loop body if the cell is within loop limits */       \
              if(i >= ix && j >= iy && k >= iz &&                                   \
                i < ix + nx && j < iy + ny && k < iz + nz)                          \
              {                                                                     \
                const int fdir[3] = {_fdir0, _fdir1, _fdir2};                       \
                loop_body;                                                          \
                (void)fdir;                                                         \
              }                                                                     \
            };                                                                      \
                                                                                    \
          BoxKernel<<<grid, block>>>(lambda_body, PV_nx, PV_ny, PV_nz);             \
          CUDA_ERR(cudaPeekAtLastError());                                          \
          CUDA_ERR(cudaStreamSynchronize(0));                                       \
        }                                                                           \
      }                                                                             \
      GrGeomSolidCellFlagInitialized(grgeom) |= (1 << (2 + PV_f));                  \
    }                                                                               \
    else                                                                            \
    {                                                                               \
      int ixl_gpu = pfmax(ix, ix_bxs);                                              \
      int iyl_gpu = pfmax(iy, iy_bxs);                                              \
      int izl_gpu = pfmax(iz, iz_bxs);                                              \
      int nx_gpu = pfmin((ix + nx - 1), BoxArrayMaxCell(boxes, 0)) - ixl_gpu + 1;   \
      int ny_gpu = pfmin((iy + ny - 1), BoxArrayMaxCell(boxes, 1)) - iyl_gpu + 1;   \
      int nz_gpu = pfmin((iz + nz - 1), BoxArrayMaxCell(boxes, 2)) - izl_gpu + 1;   \
                                                                                    \
      if(nx_gpu > 0 && ny_gpu > 0 && nz_gpu > 0)                                    \
      {                                                                             \
        dim3 block, grid;                                                           \
        FindDims(grid, block, nx_gpu, ny_gpu, nz_gpu, 1);                           \
                                                                                    \
        char *surfflag = GrGeomSolidCellFlagData(grgeom);                           \
                                                                                    \
        const int _fdir0 = fdir[0];                                                 \
        const int _fdir1 = fdir[1];                                                 \
        const int _fdir2 = fdir[2];                                                 \
                                                                                    \
        auto lambda_body =                                                          \
          GPU_LAMBDA(int i, int j, int k)                                           \
          {                                                                         \
            i += ixl_gpu;                                                           \
            j += iyl_gpu;                                                           \
            k += izl_gpu;                                                           \
                                                                                    \
            if(surfflag[(k - iz_bxs) * ny_bxs * nx_bxs +                            \
              (j - iy_bxs) * nx_bxs + (i - ix_bxs)] & (1 << (2 + PV_f)))            \
            {                                                                       \
              const int fdir[3] = {_fdir0, _fdir1, _fdir2};                         \
              loop_body;                                                            \
              (void)fdir;                                                           \
            }                                                                       \
          };                                                                        \
                                                                                    \
        BoxKernel<<<grid, block>>>(lambda_body, nx_gpu, ny_gpu, nz_gpu);            \
        CUDA_ERR(cudaPeekAtLastError());                                            \
        CUDA_ERR(cudaStreamSynchronize(0));                                         \
      }                                                                             \
    }                                                                               \
  }                                                                                 \
  (void)i;(void)j;(void)k;                                                          \
}

 /** Loop definition for CUDA. */
#define GrGeomPatchLoopBoxes_cuda(i, j, k, fdir, grgeom, patch_num,                 \
  ix, iy, iz, nx, ny, nz, loop_body)                                                \
{                                                                                   \
  for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)                           \
  {                                                                                 \
    const int *fdir = FDIR_TABLE[PV_f];                                             \
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
        dim3 block, grid;                                                           \
        FindDims(grid, block, PV_diff_x + 1, PV_diff_y + 1, PV_diff_z + 1, 1);      \
                                                                                    \
        int nx = PV_diff_x + 1;                                                     \
        int ny = PV_diff_y + 1;                                                     \
        int nz = PV_diff_z + 1;                                                     \
                                                                                    \
        const int _fdir0 = fdir[0];                                                 \
        const int _fdir1 = fdir[1];                                                 \
        const int _fdir2 = fdir[2];                                                 \
                                                                                    \
        auto lambda_body =                                                          \
          GPU_LAMBDA(int i, int j, int k)                                           \
          {                                                                         \
            const int fdir[3] = {_fdir0, _fdir1, _fdir2};                           \
            int ival = n_prev + k * ny * nx + j * nx + i;                           \
                                                                                    \
            i += PV_ixl;                                                            \
            j += PV_iyl;                                                            \
            k += PV_izl;                                                            \
                                                                                    \
            loop_body;                                                              \
            (void)fdir;                                                             \
          };                                                                        \
        n_prev += nz * ny * nx;                                                     \
                                                                                    \
        BoxKernel<<<grid, block>>>(lambda_body, nx, ny, nz);                        \
        CUDA_ERR(cudaPeekAtLastError());                                            \
        CUDA_ERR(cudaStreamSynchronize(0));                                         \
      }                                                                             \
    }                                                                               \
  }                                                                                 \
  (void)i;(void)j;(void)k;                                                          \
}

 /** Loop definition for CUDA. */
#define GrGeomPatchLoopBoxesNoFdir_cuda(i, j, k, grgeom, patch_num, ovrlnd,         \
  ix, iy, iz, nx, ny, nz, locals, setup,                                            \
  f_left, f_right, f_down, f_up, f_back, f_front, finalize)                         \
{                                                                                   \
  int n_ival = 0;                                                                   \
  for (int PV_f = 0; PV_f < GrGeomOctreeNumFaces; PV_f++)                           \
  {                                                                                 \
    BoxArray* boxes = GrGeomSolidPatchBoxes(grgeom, patch_num, PV_f);               \
                                                                                    \
    int ix_bxs = BoxArrayMinCell(boxes, 0);                                         \
    int iy_bxs = BoxArrayMinCell(boxes, 1);                                         \
    int iz_bxs = BoxArrayMinCell(boxes, 2);                                         \
                                                                                    \
    int nx_bxs = BoxArrayMaxCell(boxes, 0) - ix_bxs + 1;                            \
    int ny_bxs = BoxArrayMaxCell(boxes, 1) - iy_bxs + 1;                            \
    int nz_bxs = BoxArrayMaxCell(boxes, 2) - iz_bxs + 1;                            \
                                                                                    \
    int patch_loc;                                                                  \
    if(ovrlnd)                                                                      \
      patch_loc = GrGeomSolidNumPatches(grgeom) + patch_num;                        \
    else                                                                            \
      patch_loc = patch_num;                                                        \
                                                                                    \
    int *ptr_ival = GrGeomSolidCellIval(grgeom, patch_loc, PV_f);                   \
    if(!(ptr_ival))                                                                 \
    {                                                                               \
      GrGeomSolidCellIval(grgeom, patch_loc, PV_f) =                                \
        (int*)_talloc_device(sizeof(int) * nx_bxs * ny_bxs * nz_bxs);               \
                                                                                    \
      ptr_ival = GrGeomSolidCellIval(grgeom, patch_loc, PV_f);                      \
      for (int idx = 0; idx < nx_bxs * ny_bxs * nz_bxs; idx++)                      \
        ptr_ival[idx] = -1;                                                         \
                                                                                    \
      for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++)                  \
      {                                                                             \
        Box box = BoxArrayGetBox(boxes, PV_box);                                    \
        int PV_ixl = pfmax(ix, box.lo[0]);                                          \
        int PV_iyl = pfmax(iy, box.lo[1]);                                          \
        int PV_izl = pfmax(iz, box.lo[2]);                                          \
        int PV_ixu = pfmin((ix + nx - 1), box.up[0]);                               \
        int PV_iyu = pfmin((iy + ny - 1), box.up[1]);                               \
        int PV_izu = pfmin((iz + nz - 1), box.up[2]);                               \
                                                                                    \
        for (k = PV_izl; k <= PV_izu; k++)                                          \
          for (j = PV_iyl; j <= PV_iyu; j++)                                        \
            for (i = PV_ixl; i <= PV_ixu; i++)                                      \
            {                                                                       \
              UNPACK(locals);                                                       \
              setup;                                                                \
              switch(PV_f)                                                          \
              {                                                                     \
                f_left;                                                             \
                f_right;                                                            \
                f_down;                                                             \
                f_up;                                                               \
                f_back;                                                             \
                f_front;                                                            \
              }                                                                     \
              finalize;                                                             \
              ptr_ival[(k - iz_bxs) * ny_bxs * nx_bxs + (j - iy_bxs) *              \
                nx_bxs + (i - ix_bxs)] = n_ival++;                                  \
            }                                                                       \
      }                                                                             \
    }                                                                               \
    else                                                                            \
    {                                                                               \
      int ixl_gpu = pfmax(ix, ix_bxs);                                              \
      int iyl_gpu = pfmax(iy, iy_bxs);                                              \
      int izl_gpu = pfmax(iz, iz_bxs);                                              \
      int nx_gpu = pfmin((ix + nx - 1), BoxArrayMaxCell(boxes, 0)) - ixl_gpu + 1;   \
      int ny_gpu = pfmin((iy + ny - 1), BoxArrayMaxCell(boxes, 1)) - iyl_gpu + 1;   \
      int nz_gpu = pfmin((iz + nz - 1), BoxArrayMaxCell(boxes, 2)) - izl_gpu + 1;   \
                                                                                    \
      if(nx_gpu > 0 && ny_gpu > 0 && nz_gpu > 0)                                    \
      {                                                                             \
        dim3 block, grid;                                                           \
        FindDims(grid, block, nx_gpu, ny_gpu, nz_gpu, 1);                           \
                                                                                    \
        auto lambda_body =                                                          \
          GPU_LAMBDA(int i, int j, int k)                                           \
          {                                                                         \
            i += ixl_gpu;                                                           \
            j += iyl_gpu;                                                           \
            k += izl_gpu;                                                           \
                                                                                    \
            int ival = ptr_ival[(k - iz_bxs) * ny_bxs * nx_bxs +                    \
              (j - iy_bxs) * nx_bxs + (i - ix_bxs)];                                \
            if(ival >= 0)                                                           \
            {                                                                       \
              UNPACK(locals);                                                       \
              setup;                                                                \
              switch(PV_f)                                                          \
              {                                                                     \
                f_left;                                                             \
                f_right;                                                            \
                f_down;                                                             \
                f_up;                                                               \
                f_back;                                                             \
                f_front;                                                            \
              }                                                                     \
              finalize;                                                             \
            }                                                                       \
          };                                                                        \
                                                                                    \
        BoxKernel<<<grid, block>>>(lambda_body, nx_gpu, ny_gpu, nz_gpu);            \
        CUDA_ERR(cudaPeekAtLastError());                                            \
        CUDA_ERR(cudaStreamSynchronize(0));                                         \
      }                                                                             \
    }                                                                               \
  }                                                                                 \
  (void)i;(void)j;(void)k;                                                          \
}

 /** Loop definition for CUDA. */
#define GrGeomOctreeExteriorNodeLoop_cuda(i, j, k, node, octree, level,             \
  ix, iy, iz, nx, ny, nz, val_test, loop_body)                                      \
{                                                                                   \
  int PV_i, PV_j, PV_k, PV_l;                                                       \
  int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;                               \
                                                                                    \
  PV_i = i;                                                                         \
  PV_j = j;                                                                         \
  PV_k = k;                                                                         \
                                                                                    \
  GrGeomOctreeExteriorLoop(PV_i, PV_j, PV_k, PV_l, node, octree, level, val_test,   \
  {                                                                                 \
    if ((PV_i >= ix) && (PV_i < (ix + nx)) &&                                       \
        (PV_j >= iy) && (PV_j < (iy + ny)) &&                                       \
        (PV_k >= iz) && (PV_k < (iz + nz)))                                         \
    {                                                                               \
      i = PV_i;                                                                     \
      j = PV_j;                                                                     \
      k = PV_k;                                                                     \
      loop_body;                                                                    \
    }                                                                               \
  },                                                                                \
  {                                                                                 \
    /* find octree and region intersection */                                       \
    PV_ixl = pfmax(ix, PV_i);                                                       \
    PV_iyl = pfmax(iy, PV_j);                                                       \
    PV_izl = pfmax(iz, PV_k);                                                       \
    PV_ixu = pfmin((ix + nx), (PV_i + (int)PV_inc));                                \
    PV_iyu = pfmin((iy + ny), (PV_j + (int)PV_inc));                                \
    PV_izu = pfmin((iz + nz), (PV_k + (int)PV_inc));                                \
                                                                                    \
    if(PV_ixl < PV_ixu && PV_iyl < PV_iyu && PV_izl < PV_izu)                       \
    {                                                                               \
      const int PV_diff_x = PV_ixu - PV_ixl;                                        \
      const int PV_diff_y = PV_iyu - PV_iyl;                                        \
      const int PV_diff_z = PV_izu - PV_izl;                                        \
                                                                                    \
      dim3 block;                                                                   \
      dim3 grid;                                                                    \
      FindDims(grid, block, PV_diff_x, PV_diff_y, PV_diff_z, 1);                    \
                                                                                    \
      auto lambda_body =                                                            \
        GPU_LAMBDA(int i, int j, int k)                                             \
        {                                                                           \
          i += PV_ixl;                                                              \
          j += PV_iyl;                                                              \
          k += PV_izl;                                                              \
                                                                                    \
          loop_body;                                                                \
        };                                                                          \
                                                                                    \
      (BoxKernel<<<grid, block>>>(lambda_body, PV_diff_x, PV_diff_y, PV_diff_z));   \
      CUDA_ERR(cudaPeekAtLastError());                                              \
      CUDA_ERR(cudaStreamSynchronize(0));                                           \
    }                                                                               \
    i = PV_ixu;                                                                     \
    j = PV_iyu;                                                                     \
    k = PV_izu;                                                                     \
  })                                                                                \
  (void)i;(void)j;(void)k;                                                          \
}

 /** Loop definition for CUDA. */
#define GrGeomOutLoop_cuda(i, j, k, grgeom, r,                                      \
  ix, iy, iz, nx, ny, nz, body)                                                     \
{                                                                                   \
  if(nx > 0 && ny > 0 && nz > 0)                                                    \
  {                                                                                 \
    if(!(GrGeomSolidCellFlagInitialized(grgeom) & (1 << 1)))                        \
    {                                                                               \
      CheckCellFlagAllocation(grgeom, nx, ny, nz);                                  \
      char *outflag = GrGeomSolidCellFlagData(grgeom);                              \
                                                                                    \
      GrGeomOctree  *PV_node;                                                       \
      double PV_ref = pow(2.0, r);                                                  \
                                                                                    \
      i = GrGeomSolidOctreeIX(grgeom) * (int)PV_ref;                                \
      j = GrGeomSolidOctreeIY(grgeom) * (int)PV_ref;                                \
      k = GrGeomSolidOctreeIZ(grgeom) * (int)PV_ref;                                \
      GrGeomOctreeExteriorNodeLoop(i, j, k, PV_node,                                \
                                   GrGeomSolidData(grgeom),                         \
                                   GrGeomSolidOctreeBGLevel(grgeom) + r,            \
                                   ix, iy, iz, nx, ny, nz,                          \
                                   TRUE,                                            \
      {                                                                             \
        body;                                                                       \
        outflag[(k - iz) * ny * nx + (j - iy) * nx + (i - ix)] |= (1 << 1);         \
      });                                                                           \
      GrGeomSolidCellFlagInitialized(grgeom) |= (1 << 1);                           \
    }                                                                               \
    else                                                                            \
    {                                                                               \
      dim3 block, grid;                                                             \
      FindDims(grid, block, nx, ny, nz, 1);                                         \
                                                                                    \
      char *outflag = GrGeomSolidCellFlagData(grgeom);                              \
      auto lambda_body =                                                            \
        GPU_LAMBDA(int i, int j, int k)                                             \
        {                                                                           \
          i += ix;                                                                  \
          j += iy;                                                                  \
          k += iz;                                                                  \
                                                                                    \
          if(outflag[(k - iz) * ny * nx + (j - iy) * nx + (i - ix)] & (1 << 1))     \
          {                                                                         \
            body;                                                                   \
          }                                                                         \
        };                                                                          \
                                                                                    \
      BoxKernel<<<grid, block>>>(lambda_body, nx, ny, nz);                          \
      CUDA_ERR(cudaPeekAtLastError());                                              \
      CUDA_ERR(cudaStreamSynchronize(0));                                           \
    }                                                                               \
  }                                                                                 \
  (void)i;(void)j;(void)k;                                                          \
}

}
#endif // PF_CUDALOOPS_H
