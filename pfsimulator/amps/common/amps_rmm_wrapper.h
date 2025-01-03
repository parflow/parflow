#ifndef RMM_WRAPPER
#define RMM_WRAPPER

#include "parflow_config.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void amps_rmmInit();
void* amps_rmmAlloc(size_t bytes);
void amps_rmmFree(void *p);

#ifdef __cplusplus
}
#endif

#endif
