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

#define CUDA_ERR_ARG( err, arg1, arg2, arg3 ) (gpuErrorArg( err, __FILE__, __LINE__, arg1, arg2, arg3 ))
static inline void gpuErrorArg(cudaError_t err, const char *file, int line, int arg1, int arg2, int arg3) {
	if (err != cudaSuccess) {
		printf("\n\n%s in %s at line %d; arg1: %d, arg2: %d, arg3: %d\n", cudaGetErrorString(err), file, line, arg1, arg2, arg3);
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

// #ifndef NDEBUG
#include "nvToolsExt.h"

#undef PUSH_RANGE
#define PUSH_RANGE(name,cid) { \
	const uint32_t colors_nvtx[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff }; \
	const int num_colors_nvtx = sizeof(colors_nvtx)/sizeof(uint32_t); \
    int color_id_nvtx = cid; \
    color_id_nvtx = color_id_nvtx%num_colors_nvtx;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors_nvtx[color_id_nvtx]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#undef POP_RANGE
#define POP_RANGE nvtxRangePop();
// #endif

#endif // PFCUDAERR_H
