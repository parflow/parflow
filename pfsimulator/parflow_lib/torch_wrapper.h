#ifndef TORCH_WRAPPER
#define TORCH_WRAPPER

#include "parflow_config.h"

#ifdef PARFLOW_HAVE_TORCH

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

double* predict_next_pressure_step(char* model_filepath, double* pp, int nx, int ny, int nz);
void* create_random_tensor(int rows, int cols);
void print_tensor(void* tensor_ptr);
void free_tensor(void* tensor_ptr);

#ifdef __cplusplus
}
#endif

#endif

#endif
