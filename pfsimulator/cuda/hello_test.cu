#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"
// #include <cuda.h>
// #include <cuda_runtime.h>

__global__ void kernel (int* data, const int nt){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(tid >= nt)return;
    data[tid] = tid*tid;
    // printf( "Hello from GPU thread %d!\n",tid); 

}

extern "C"
void hello_test() {
  printf("\nStart CUDA hello_test!\n"); 
  int *data, nt = 10;
//   data = (int*)malloc(sizeof(int) * nt); 
  cudaMallocManaged(&data, sizeof(int) * nt);

  kernel<<<(nt+8)/8,8>>>(data,nt);
  cudaDeviceSynchronize();
  for(unsigned int i = 0; i < nt; i++){
    printf("data[%u]: %d\n",i,data[i]);
  }
//   free(data);
  cudaFree(data);
  printf("CUDA hello_test over!\n\n"); 
  return;
}