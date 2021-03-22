/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
 *  LLC. Produced at the Lawrence Livermore National Laboratory. Written
 *  by the Parflow Team (see the CONTRIBUTORS file)
 *  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
 *
 *  This file is part of Parflow. For details, see
 *  http://www.llnl.gov/casc/parflow
 *
 *  Please read the COPYRIGHT file or Our Notice and the LICENSE file
 *  for the GNU Lesser General Public License.
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
 **********************************************************************EHEADER*/
/*****************************************************************************
*
*****************************************************************************/

// extern "C"{
// extern "C++"{
// #include <Kokkos_Core.hpp>


// /**
//  * @brief CUDA error handling.
//  * 
//  * If error detected, print error message and exit.
//  *
//  * @param expr CUDA error (of type cudaError_t) [IN]
//  */
// #define CUDA_ERR(expr)                                                                 \
// {                                                                                      \
//   cudaError_t err = expr;                                                              \
//   if (err != cudaSuccess) {                                                            \
//     printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);  \
//     exit(1);                                                                           \
//   }                                                                                    \
// }

// #define BoxLoopI1_NEW(i, j, k,                                                          \
//   ix, iy, iz, nx, ny, nz,                                                           \
//   i1, nx1, ny1, nz1, sx1, sy1, sz1,                                                 \
//   loop_body)                                                                        \
// {                                                                                   \
//   if(nx > 0 && ny > 0 && nz > 0)                                                    \
//   {                                                                                 \
//     DeclareInc(PV_jinc_1, PV_kinc_1, nx, ny, nz, nx1, ny1, nz1, sx1, sy1, sz1);     \
//                                                                                     \
//     const auto &ref_i1 = i1;                                                        \
//                                                                                     \
//     auto lambda_body =                                                              \
//       KOKKOS_LAMBDA(int i, int j, int k)                                               \
//       {                                                                             \
//         const int i1 = k * PV_kinc_1 + (k * ny + j) * PV_jinc_1                     \
//           + (k * ny * nx + j * nx + i) * sx1 + ref_i1;                              \
//                                                                                     \
//         i += ix;                                                                    \
//         j += iy;                                                                    \
//         k += iz;                                                                    \
//                                                                                     \
//         loop_body;                                                                  \
//       };                                                                            \
//                                                                                     \
//     /* Rank<3> Case: Rank is provided, all other parameters are default  */         \
//     using MDPolicyType_3D = typename Kokkos::Experimental::MDRangePolicy<           \
//         Kokkos::Experimental::Rank<3> >;                                            \
//     MDPolicyType_3D mdpolicy_3d({{0, 0, 0}}, {{nx, ny, nz}});                       \
//     Kokkos::parallel_for(mdpolicy_3d, lambda_body);                                 \
//     CUDA_ERR(cudaPeekAtLastError());                                                \
//     CUDA_ERR(cudaStreamSynchronize(0));                                             \
//   }                                                                                 \
//   (void)i;(void)j;(void)k;                                                          \
// }

// int Kokkos_initd = 0;

// static inline void *_ctalloc_cuda(size_t size)
// {
//   void *ptr = NULL;  

// #ifdef PARFLOW_HAVE_RMM
//   RMM_ERR(rmmAlloc(&ptr,size,0,__FILE__,__LINE__));
// #else
//   CUDA_ERR(cudaMallocManaged((void**)&ptr, size, cudaMemAttachGlobal));
//   // CUDA_ERR(cudaHostAlloc((void**)&ptr, size, cudaHostAllocMapped));
// #endif  
//   // memset(ptr, 0, size);
//   CUDA_ERR(cudaMemset(ptr, 0, size));  
  
//   return ptr;
// }
// static inline void _tfree_cuda(void *ptr)
// {
// #ifdef PARFLOW_HAVE_RMM
//   RMM_ERR(rmmFree(ptr,0,__FILE__,__LINE__));
// #else
//   CUDA_ERR(cudaFree(ptr));
//   // CUDA_ERR(cudaFreeHost(ptr));
// #endif
// }

// void kokkos_func(){
//   if(Kokkos_initd == 0){
//     Kokkos::InitArguments args;
//     args.ndevices = 1;
//     Kokkos::initialize(args);
//   }

//   // Rank<3> Case: Rank is provided, all other parameters are default
//   using MDPolicyType_3D = typename Kokkos::Experimental::MDRangePolicy<
//       Kokkos::Experimental::Rank<3> >;
//   // Construct 3D MDRangePolicy: lower and upper bounds provided, tile dims
//   // defaulted
//   MDPolicyType_3D mdpolicy_3d({{0, 0, 0}}, {{3, 3, 3}});

//   int* a;// = new double[10];
//   cudaMallocManaged(&a, 27*sizeof(int));
//   Kokkos::parallel_for(
//       mdpolicy_3d, KOKKOS_LAMBDA(const int i, const int j, const int k) {
//         a[3 * 3 * k + 3 * j + i] = 2 * (3 * 3 * k + 3 * j + i);
//   });
//       // Kokkos::parallel_for(
//         // 27, KOKKOS_LAMBDA(const int i) {
//           // a[i] = i;
//         // });
//   CUDA_ERR(cudaDeviceSynchronize()); 
//   if(Kokkos_initd == 0){
//     for(int ii = 0; ii < 27; ii++){
//       printf("a[%d]: %d\n",ii,a[ii]);
//     }
//     Kokkos_initd = 1;
//   }
//   cudaFree(a);
// }

// }


int Kokkos_initd = 0;
#include "parflow.h"

// double *kokkos_alloc(int count){
//   return (count) ? (double*)_ctalloc_cuda(sizeof(double) * (unsigned int)(count)) : NULL;
// }
// void kokkos_free(double *ptr){
//   if (ptr) _tfree_cuda(ptr); else {};
// }


void     Axpy(
              double  alpha,
              Vector *x,
              Vector *y)
{
  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *y_sub;
  Subvector  *x_sub;

  double     *yp, *xp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;

  int i_s, i, j, k, iv;

  // kokkos_func();

  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    y_sub = VectorSubvector(y, i_s);
    x_sub = VectorSubvector(x, i_s);

    nx_v = SubvectorNX(y_sub);
    ny_v = SubvectorNY(y_sub);
    nz_v = SubvectorNZ(y_sub);

    yp = SubvectorElt(y_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);

    iv = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      yp[iv] += alpha * xp[iv];
    });
  }

  IncFLOPCount(2 * VectorSize(x));
}
// }
