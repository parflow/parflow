#ifndef UMPIRE_WRAPPER
#define UMPIRE_WRAPPER

#include <stddef.h>
#include "parflow_config.h"

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#endif

void umpireInit();
void* umpireAlloc(size_t bytes);
void umpireFree(void *p);

#ifdef __cplusplus
}
#endif

#endif
