#ifndef _SIMULATION_SNAPSHOT_H
#define _SIMULATION_SNAPSHOT_H

#include "parflow.h"


typedef struct {
  int * file_number;
  double * time;
  Vector * pressure_out;
  Vector * porosity_out;
  Vector * saturation_out;
  Vector * evap_trans;
  Vector * evap_trans_sum;

  Grid * grid;
  ProblemData * problem_data;
} SimulationSnapshot;

/**
 * GetSimulationSnapshot is used to comfortably fill a SimulationSnapshot data structure
 * in the AdvanceRichards() or the SetupRichards() context in richards_solver.c
 */
#define GetSimulationSnapshot \
  (SimulationSnapshot){ \
    &(instance_xtra->file_number), \
    &t, \
    instance_xtra->pressure, \
    NULL, \
    instance_xtra->saturation, \
    NULL, \
    NULL, \
    instance_xtra->grid, \
    problem_data \
  }
//^^
// porosity ... has to be set later
// evap_trans ... has to be set later  TODO: actually should be accessible with instance_xtra too!
// evap_trans_sum ... has to be set later
#endif
