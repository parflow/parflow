#ifndef PFCUDAERR_H
#define PFCUDAERR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <rmm/rmm_api.h>

#undef CUDA_ERR
#define CUDA_ERR(expr)                                                                      \
{                                                                                           \
	cudaError_t err = expr;                                                                 \
	if (err != cudaSuccess) {                                                               \
		printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);   \
		exit(1);                                                                            \
	}                                                                                       \
}

#undef RMM_ERR
#define RMM_ERR(expr)                                                                       \
{                                                                                           \
	rmmError_t err = expr;                                                                  \
	if (err != RMM_SUCCESS) {                                                               \
		printf("\n\n%s in %s at line %d\n", rmmGetErrorString(err), __FILE__, __LINE__);    \
		exit(1);                                                                            \
	}                                                                                       \
}

#endif // PFCUDAERR_H
