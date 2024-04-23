/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License (as published
*  by the Free Software Foundation) version 2.1 dated February 1999.
*
*  This program is distributed in the hope that it will be useful, but
*  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
*  and conditions of the GNU General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public
*  License along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
*  USA
**********************************************************************EHEADER*/
/*****************************************************************************
*
* Routine to be invoked by WRF model.
*
*****************************************************************************/

#include "parflow.h"
#include "solver.h"

#include <string.h>

amps_ThreadLocalDcl(PFModule *, Solver_module);
amps_ThreadLocalDcl(PFModule *, solver);
amps_ThreadLocalDcl(Vector   *, evap_trans);

void wrfparflowinit_(char *input_file)
{
  Grid         *grid;
  char *seperators = " \n";
  /* Fortran char array is not NULL terminated */
  char *filename = strtok(input_file, seperators);

  /*-----------------------------------------------------------------------
   * Initialize AMPS from existing MPI state
   *-----------------------------------------------------------------------*/
  if (amps_EmbeddedInit())
  {
    amps_Printf("Error: amps_EmbeddedInit initalization failed\n");
    exit(1);
  }

  /*-----------------------------------------------------------------------
   * Set up globals structure
   *-----------------------------------------------------------------------*/
  NewGlobals(filename);

  /*-----------------------------------------------------------------------
   * Read the Users Input Deck
   *-----------------------------------------------------------------------*/
  amps_ThreadLocal(input_database) = IDB_NewDB(GlobalsInFileName);

  /*-----------------------------------------------------------------------
   * Setup log printing
   *-----------------------------------------------------------------------*/
  NewLogging();

  /*-----------------------------------------------------------------------
   * Setup timing table
   *-----------------------------------------------------------------------*/
  NewTiming();

  /* End of main includes */

  /* Begin of Solver includes */

  GlobalsNumProcsX = GetIntDefault("Process.Topology.P", 1);
  GlobalsNumProcsY = GetIntDefault("Process.Topology.Q", 1);
  GlobalsNumProcsZ = GetIntDefault("Process.Topology.R", 1);

  GlobalsNumProcs = amps_Size(amps_CommWorld);

  GlobalsBackground = ReadBackground();

  GlobalsUserGrid = ReadUserGrid();

  SetBackgroundBounds(GlobalsBackground, GlobalsUserGrid);

  GlobalsMaxRefLevel = 0;

  amps_ThreadLocal(Solver_module) = PFModuleNewModuleType(SolverImpesNewPublicXtraInvoke, SolverRichards, ("Solver"));

  amps_ThreadLocal(solver) = PFModuleNewInstance(amps_ThreadLocal(Solver_module), ());

  /* End of solver includes */

  SetupRichards(amps_ThreadLocal(solver));

  /* Create the flow grid */
  grid = CreateGrid(GlobalsUserGrid);

  /* Create the PF vector holding flux */
  amps_ThreadLocal(evap_trans) = NewVectorType(grid, 1, 1, vector_cell_centered);
  InitVectorAll(amps_ThreadLocal(evap_trans), 0.0);
}

void wrfparflowadvance_(double *current_time,
                        double *dt,
                        float * wrf_flux,
                        float * wrf_pressure,
                        float * wrf_porosity,
                        float * wrf_saturation,
                        int *   num_soil_layers,
                        int *   ghost_size_i_lower,
                        int *   ghost_size_j_lower,
                        int *   ghost_size_i_upper,
                        int *   ghost_size_j_upper)

{
  ProblemData *problem_data = GetProblemDataRichards(amps_ThreadLocal(solver));

  double stop_time = *current_time + *dt;

  Vector       *pressure_out;
  Vector       *porosity_out;
  Vector       *saturation_out;

  VectorUpdateCommHandle   *handle;

  WRF2PF(wrf_flux, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         amps_ThreadLocal(evap_trans),
         ProblemDataIndexOfDomainTop(problem_data));

  /*
   * Exchange ghost layer data for the newly set fluxes
   */
  handle = InitVectorUpdate(evap_trans, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  // SGS this is somewhat inefficient as we are allocating
  // a pf module for each timestep.
  double initial_step = *dt;
  double growth_factor = 2.0;
  double max_step = *dt;
  // SGS what should this be set to?
  double min_step = *dt * 1e-8;

  PFModule *time_step_control;

  time_step_control = NewPFModule((void*)SelectTimeStep,
                                  (void*)WRFSelectTimeStepInitInstanceXtra, \
                                  (void*)SelectTimeStepFreeInstanceXtra,    \
                                  (void*)WRFSelectTimeStepNewPublicXtra,    \
                                  (void*)WRFSelectTimeStepFreePublicXtra,   \
                                  (void*)SelectTimeStepSizeOfTempData,      \
                                  NULL, NULL);

  ThisPFModule = time_step_control;
  WRFSelectTimeStepNewPublicXtra(initial_step,
                                 growth_factor,
                                 max_step,
                                 min_step);
  ThisPFModule = NULL;

  PFModule *time_step_control_instance = PFModuleNewInstance(time_step_control, ());

  AdvanceRichards(amps_ThreadLocal(solver),
                  *current_time,
                  stop_time,
                  time_step_control_instance,
                  amps_ThreadLocal(evap_trans),
                  &pressure_out,
                  &porosity_out,
                  &saturation_out);

  PFModuleFreeInstance(time_step_control_instance);
  PFModuleFreeModule(time_step_control);

  /* TODO: SGS
   * Are these needed here?  Decided to put them in just be safe but
   * they could be unnecessary.
   */
  handle = InitVectorUpdate(pressure_out, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
  handle = InitVectorUpdate(porosity_out, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
  handle = InitVectorUpdate(saturation_out, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  PF2WRF(pressure_out, wrf_pressure, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         ProblemDataIndexOfDomainTop(problem_data));

  PF2WRF(porosity_out, wrf_porosity, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         ProblemDataIndexOfDomainTop(problem_data));

  PF2WRF(saturation_out, wrf_saturation, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         ProblemDataIndexOfDomainTop(problem_data));
}



/*
 * Copy data from a WRF array to a PF vector based on the
 * k-index data for the top of the domain.
 */
void WRF2PF(
            float * wrf_array, /* WRF array */
            int     wrf_depth, /* Depth (Z) of WRF array, X,Y are assumed
                                * to be same as PF vector subgrid */
            int     ghost_size_i_lower, /* Number of ghost cells */
            int     ghost_size_j_lower,
            int     ghost_size_i_upper,
            int     ghost_size_j_upper,
            Vector *pf_vector,
            Vector *top)
{
  Grid       *grid = VectorGrid(pf_vector);
  int sg;

  (void)ghost_size_j_upper;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    Subgrid *subgrid = GridSubgrid(grid, sg);

    int ix = SubgridIX(subgrid);
    int iy = SubgridIY(subgrid);

    int nx = SubgridNX(subgrid);
    int ny = SubgridNY(subgrid);

    int wrf_nx = nx + ghost_size_i_lower + ghost_size_i_upper;

    Subvector *subvector = VectorSubvector(pf_vector, sg);
    double *subvector_data = SubvectorData(subvector);

    Subvector *top_subvector = VectorSubvector(top, sg);
    double    *top_data = SubvectorData(top_subvector);

    int i, j, k;

    for (i = ix; i < ix + nx; i++)
    {
      for (j = iy; j < iy + ny; j++)
      {
        int top_index = SubvectorEltIndex(top_subvector, i, j, 0);

        // SGS What to do if near bottom such that
        // there are not wrf_depth values?
        int iz = (int)top_data[top_index] - (wrf_depth - 1);
        for (k = iz; k < iz + wrf_depth; k++)
        {
          int pf_index = SubvectorEltIndex(subvector, i, j, k);
          int wrf_index = (i - ix + ghost_size_i_lower) +
                          ((wrf_depth - (k - iz) - 1) * wrf_nx) +
                          ((j - iy + ghost_size_j_lower) * (wrf_nx * wrf_depth));
          subvector_data[pf_index] = (double)(wrf_array[wrf_index]);
        }
      }
    }
  }
}

/*
 * Copy data from a PF vector to a WRF array based on the
 * k-index data for the top of the domain.
 */
void PF2WRF(
            Vector *pf_vector,
            float * wrf_array, /* WRF array */
            int     wrf_depth, /* Depth (Z) of WRF array, X,Y are assumed
                                * to be same as PF vector subgrid */
            int     ghost_size_i_lower, /* Number of ghost cells */
            int     ghost_size_j_lower,
            int     ghost_size_i_upper,
            int     ghost_size_j_upper,
            Vector *top)
{
  Grid       *grid = VectorGrid(pf_vector);
  int sg;

  (void)ghost_size_j_upper;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    Subgrid *subgrid = GridSubgrid(grid, sg);

    int ix = SubgridIX(subgrid);
    int iy = SubgridIY(subgrid);

    int nx = SubgridNX(subgrid);
    int ny = SubgridNY(subgrid);

    int wrf_nx = nx + ghost_size_i_lower + ghost_size_i_upper;

    Subvector *subvector = VectorSubvector(pf_vector, sg);
    double *subvector_data = SubvectorData(subvector);

    Subvector *top_subvector = VectorSubvector(top, sg);
    double    *top_data = SubvectorData(top_subvector);

    int i, j, k;

    for (i = ix; i < ix + nx; i++)
    {
      for (j = iy; j < iy + ny; j++)
      {
        int top_index = SubvectorEltIndex(top_subvector, i, j, 0);

        // SGS What to do if near bottom such that
        // there are not wrf_depth values?
        int iz = (int)top_data[top_index] - (wrf_depth - 1);

        for (k = iz; k < iz + wrf_depth; k++)
        {
          int pf_index = SubvectorEltIndex(subvector, i, j, k);
          int wrf_index = (i - ix + ghost_size_i_lower) +
                          ((wrf_depth - (k - iz) - 1) * wrf_nx) +
                          ((j - iy + ghost_size_j_lower) * (wrf_nx * wrf_depth));
          wrf_array[wrf_index] = (float)(subvector_data[pf_index]);
        }
      }
    }
  }
}

