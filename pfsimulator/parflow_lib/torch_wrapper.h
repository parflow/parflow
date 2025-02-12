#ifndef TORCH_WRAPPER
#define TORCH_WRAPPER

#include "parflow_config.h"

#ifdef PARFLOW_HAVE_TORCH

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

double* predict_next_pressure_step(double* pp, int nx, int ny, int nz);

#ifdef __cplusplus
}
#endif

#endif

#endif
