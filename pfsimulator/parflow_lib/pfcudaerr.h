#ifndef PFCUDAERR_H
#define PFCUDAERR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <rmm/rmm_api.h>

#define CUDA_ERR( err ) (gpuError( err, __FILE__, __LINE__ ))
static inline void gpuError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

#define RMM_ERR( err ) (rmmError( err, __FILE__, __LINE__ ))
static inline void rmmError(rmmError_t err, const char *file, int line) {
	if (err != RMM_SUCCESS) {
		printf("\n\n%s in %s at line %d\n", rmmGetErrorString(err), file, line);
		exit(1);
	}
}
#endif // PFCUDAERR_H
