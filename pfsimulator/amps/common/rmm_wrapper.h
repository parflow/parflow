#ifndef RMM_WRAPPER
#define RMM_WRAPPER

#include "parflow_config.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void rmmInit();
void* rmmAlloc(size_t bytes);
void rmmFree(void *p);

#ifdef __cplusplus
}
#endif

#endif
