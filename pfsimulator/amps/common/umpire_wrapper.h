#ifndef UMPIRE_WRAPPER
#define UMPIRE_WRAPPER

#include "parflow_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void umpireInit();
void* umpireAlloc(size_t bytes);
void umpireFree(void *p, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif
