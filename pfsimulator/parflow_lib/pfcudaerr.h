#ifndef PFCUDAERR_H
#define PFCUDAERR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <rmm/rmm_api.h>

#define CUDA_ERR(expr)                                                                      \
{                                                                                           \
	cudaError_t err = expr;                                                                 \
	if (err != cudaSuccess) {                                                               \
		printf("\n\n%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);   \
		exit(1);                                                                            \
	}                                                                                       \
}

#define RMM_ERR(expr)                                                                       \
{                                                                                           \
	rmmError_t err = expr;                                                                  \
	if (err != RMM_SUCCESS) {                                                               \
		printf("\n\n%s in %s at line %d\n", rmmGetErrorString(err), __FILE__, __LINE__);    \
		exit(1);                                                                            \
	}                                                                                       \
}

#include "nvToolsExt.h"

#undef PUSH_RANGE
#define PUSH_RANGE(name,cid)                                                                \
{                                                                                           \
	const uint32_t colors_nvtx[] =                                                          \
	  {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff}; \
	const int num_colors_nvtx = sizeof(colors_nvtx)/sizeof(uint32_t);                       \
    int color_id_nvtx = cid;                                                                \
    color_id_nvtx = color_id_nvtx%num_colors_nvtx;                                          \
    nvtxEventAttributes_t eventAttrib = {0};                                                \
    eventAttrib.version = NVTX_VERSION;                                                     \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                       \
    eventAttrib.colorType = NVTX_COLOR_ARGB;                                                \
    eventAttrib.color = colors_nvtx[color_id_nvtx];                                         \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                                      \
    eventAttrib.message.ascii = name;                                                       \
    nvtxRangePushEx(&eventAttrib);                                                          \
}
#undef POP_RANGE
#define POP_RANGE nvtxRangePop();

#endif // PFCUDAERR_H
