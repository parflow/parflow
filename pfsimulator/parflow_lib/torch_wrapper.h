#ifndef TORCH_WRAPPER
#define TORCH_WRAPPER

#include "parflow_config.h"

#ifdef PARFLOW_HAVE_TORCH

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

  void init_torch_model(char *model_filepath, int nx, int ny, int nz, double *po_dat,
			double *mann_dat, double *slopex_dat, double *slopey_dat, double *permx_dat,
			double *permy_dat, double *permz_dat, double *sres_dat, double *ssat_dat);
  double* predict_next_pressure_step(double *pp, double *et, int nx, int ny, int nz);

#ifdef __cplusplus
}
#endif

#endif

#endif
