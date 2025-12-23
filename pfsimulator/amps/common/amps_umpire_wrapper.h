#ifndef UMPIRE_WRAPPER
#define UMPIRE_WRAPPER

#include <stddef.h>
#include "parflow_config.h"

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#endif

void amps_umpireInit();
void* amps_umpireAlloc(size_t bytes);
void amps_umpireFree(void *p);

#ifdef __cplusplus
}
#endif

#endif
