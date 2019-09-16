#include "cuda.h"
#include "cuda_runtime.h"

#ifndef CUDA_ERR
#define CUDA_ERR( err ) (GpuError( err, __FILE__, __LINE__ ))
static inline void GpuError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}
#endif