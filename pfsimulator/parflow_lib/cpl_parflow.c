/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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
* Routine to be invoked by NUOPC cap.
*
*****************************************************************************/

#include "parflow.h"
#include "solver.h"

#include <string.h>

amps_ThreadLocalDcl(PFModule *, Solver_module);
amps_ThreadLocalDcl(PFModule *, solver);
amps_ThreadLocalDcl(Vector   *, evap_trans);

void cplparflowinit_(int *  fcom,
                     char * input_file,
                     int *  numprocs,
                     int *  subgridcount,
                     int *  nz,
                     int *  ierror)
{
  Grid         *grid;
  char *separators = " \n";
  /* Fortran char array is not NULL terminated */
  char *filename = strtok(input_file, separators);

  /*-----------------------------------------------------------------------
   * Initialize AMPS from existing MPI state
   *-----------------------------------------------------------------------*/
  if (amps_EmbeddedInitFComm(fcom))
  {
    amps_Printf("Error: amps_EmbeddedInitFComm initialization failed\n");
    exit(1);
  }

  /* Set the destination stream for PF output/logging */
  amps_SetConsole(stdout);

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

  /* Processor Count X*Y */
  *numprocs = GlobalsNumProcs;

  /* Subgrid Count */
  *subgridcount = SubgridArraySize(GridSubgrids(grid));

  /* Total Soil Layers (NZ) */
  *nz = GetInt("ComputationalGrid.NZ");

  *ierror = 0;
}

void cplparflowadvance_(double * current_time,
                        double * dt,
                        float *  imp_flux,
                        float *  exp_pressure,
                        float *  exp_porosity,
                        float *  exp_saturation,
                        float *  exp_specific,
                        float *  exp_zmult,
                        int *    num_soil_layers,
                        int *    num_cpl_layers,
                        int *    ghost_size_i_lower,
                        int *    ghost_size_j_lower,
                        int *    ghost_size_i_upper,
                        int *    ghost_size_j_upper,
                        int *    ierror)
{
  ProblemData *problem_data = GetProblemDataRichards(amps_ThreadLocal(solver));
  Vector *solver_mask = GetMaskRichards(amps_ThreadLocal(solver));

  double stop_time = *current_time + *dt;

  Vector       *pressure_out;
  Vector       *porosity_out;
  Vector       *saturation_out;
  Vector       *specific_out;
  Vector       *zmult_out;

  VectorUpdateCommHandle   *handle;

  CPL2PF(imp_flux, *num_soil_layers, *num_cpl_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         amps_ThreadLocal(evap_trans),
         ProblemDataIndexOfDomainTop(problem_data),
         solver_mask);

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
                                  (void*)CPLSelectTimeStepInitInstanceXtra, \
                                  (void*)SelectTimeStepFreeInstanceXtra,    \
                                  (void*)CPLSelectTimeStepNewPublicXtra,    \
                                  (void*)CPLSelectTimeStepFreePublicXtra,   \
                                  (void*)SelectTimeStepSizeOfTempData,      \
                                  NULL, NULL);

  ThisPFModule = time_step_control;
  CPLSelectTimeStepNewPublicXtra(initial_step,
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

  specific_out = ProblemDataSpecificStorage(problem_data);
  zmult_out = ProblemDataZmult(problem_data);

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
  handle = InitVectorUpdate(specific_out, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
  handle = InitVectorUpdate(zmult_out, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  PF2CPL(pressure_out, exp_pressure, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         ProblemDataIndexOfDomainTop(problem_data),
         solver_mask);

  PF2CPL(porosity_out, exp_porosity, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         ProblemDataIndexOfDomainTop(problem_data),
         solver_mask);

  PF2CPL(saturation_out, exp_saturation, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         ProblemDataIndexOfDomainTop(problem_data),
         solver_mask);

  PF2CPL(specific_out, exp_specific, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         ProblemDataIndexOfDomainTop(problem_data),
         solver_mask);

  PF2CPL(zmult_out, exp_zmult, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         ProblemDataIndexOfDomainTop(problem_data),
         solver_mask);

  *ierror = 0;
}

void cplparflowexport_(float * exp_pressure,
                       float * exp_porosity,
                       float * exp_saturation,
                       float * exp_specific,
                       float * exp_zmult,
                       int *   num_soil_layers,
                       int *   num_cpl_layers,
                       int *   ghost_size_i_lower,
                       int *   ghost_size_j_lower,
                       int *   ghost_size_i_upper,
                       int *   ghost_size_j_upper,
                       int *   ierror)
{
  ProblemData *problem_data = GetProblemDataRichards(amps_ThreadLocal(solver));
  Vector *solver_mask = GetMaskRichards(amps_ThreadLocal(solver));

  Vector       *pressure_out;
  Vector       *porosity_out;
  Vector       *saturation_out;
  Vector       *specific_out;
  Vector       *zmult_out;

  VectorUpdateCommHandle   *handle;

  ExportRichards(amps_ThreadLocal(solver),
                 &pressure_out,
                 &porosity_out,
                 &saturation_out);

  specific_out = ProblemDataSpecificStorage(problem_data);
  zmult_out = ProblemDataZmult(problem_data);

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
  handle = InitVectorUpdate(specific_out, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
  handle = InitVectorUpdate(zmult_out, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  PF2CPL(pressure_out, exp_pressure, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         ProblemDataIndexOfDomainTop(problem_data),
         solver_mask);

  PF2CPL(porosity_out, exp_porosity, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         ProblemDataIndexOfDomainTop(problem_data),
         solver_mask);

  PF2CPL(saturation_out, exp_saturation, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         ProblemDataIndexOfDomainTop(problem_data),
         solver_mask);

  PF2CPL(specific_out, exp_specific, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         ProblemDataIndexOfDomainTop(problem_data),
         solver_mask);

  PF2CPL(zmult_out, exp_zmult, *num_soil_layers,
         *ghost_size_i_lower, *ghost_size_j_lower,
         *ghost_size_i_upper, *ghost_size_j_upper,
         ProblemDataIndexOfDomainTop(problem_data),
         solver_mask);

  *ierror = 0;
}

/*
 * Copy data from a import array to a PF vector based on the
 * k-index data for the top of the domain.
 */
void CPL2PF(
            float *  imp_array,          /* import array */
            int      imp_nz,             /* layers of import array, X, Y are
                                          * assumed to be the same as PF vector
                                          * subgrid */
            int      cpy_layers,         /* number of layers to copy */
            int      ghost_size_i_lower, /* Number of ghost cells */
            int      ghost_size_j_lower,
            int      ghost_size_i_upper,
            int      ghost_size_j_upper,
            Vector * pf_vector,
            Vector * top,
            Vector * mask)
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

    int imp_nx = nx + ghost_size_i_lower + ghost_size_i_upper;

    Subvector *subvector = VectorSubvector(pf_vector, sg);
    double *subvector_data = SubvectorData(subvector);

    Subvector *top_subvector = VectorSubvector(top, sg);
    double    *top_data = SubvectorData(top_subvector);

    Subvector *mask_subvector = VectorSubvector(mask, sg);
    double    *mask_data = SubvectorData(mask_subvector);

    int i, j, k;

    for (i = ix; i < ix + nx; i++)
    {
      for (j = iy; j < iy + ny; j++)
      {
        int top_index = SubvectorEltIndex(top_subvector, i, j, 0);
        int mask_index = SubvectorEltIndex(mask_subvector, i, j, 0);
        // check for cell outside watershed
        if (mask_data[mask_index] > 0)
        {
          // SGS What to do if near bottom such that
          // there are not imp_nz values?
          int iz = (int)top_data[top_index] - (cpy_layers - 1);
          for (k = iz; k < iz + cpy_layers; k++)
          {
            int pf_index = SubvectorEltIndex(subvector, i, j, k);
            int imp_index = (i - ix + ghost_size_i_lower) +
                            ((cpy_layers - (k - iz) - 1) * imp_nx) +
                            ((j - iy + ghost_size_j_lower) * (imp_nx * imp_nz));
            subvector_data[pf_index] = (double)(imp_array[imp_index]);
          }
        }
      }
    }
  }
}

/*
 * Copy data from a PF vector to an export array
 */
void PF2CPL(
            Vector * pf_vector,
            float *  exp_array, /* export array */
            int      exp_nz,             /* layers of export array, X, Y are
                                          * assumed to be the same as PF vector
                                          * subgrid */
            int      ghost_size_i_lower, /* Number of ghost cells */
            int      ghost_size_j_lower,
            int      ghost_size_i_upper,
            int      ghost_size_j_upper,
            Vector * top,
            Vector * mask)
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

    int exp_nx = nx + ghost_size_i_lower + ghost_size_i_upper;

    Subvector *subvector = VectorSubvector(pf_vector, sg);
    double *subvector_data = SubvectorData(subvector);

    Subvector *top_subvector = VectorSubvector(top, sg);
    double    *top_data = SubvectorData(top_subvector);

    Subvector *mask_subvector = VectorSubvector(mask, sg);
    double    *mask_data = SubvectorData(mask_subvector);

    int i, j, k;

    for (i = ix; i < ix + nx; i++)
    {
      for (j = iy; j < iy + ny; j++)
      {
        int top_index = SubvectorEltIndex(top_subvector, i, j, 0);
        int mask_index = SubvectorEltIndex(mask_subvector, i, j, 0);
        // check for cell outside watershed
        if (mask_data[mask_index] > 0)
        {
          // SGS What to do if near bottom such that
          // there are not exp_nz values?
          int iz = (int)top_data[top_index] - (exp_nz - 1);

          for (k = iz; k < iz + exp_nz; k++)
          {
            int pf_index = SubvectorEltIndex(subvector, i, j, k);
            int exp_index = (i - ix + ghost_size_i_lower) +
                            ((exp_nz - (k - iz) - 1) * exp_nx) +
                            ((j - iy + ghost_size_j_lower) * (exp_nx * exp_nz));
            exp_array[exp_index] = (float)(subvector_data[pf_index]);
          }
        }
        else
        {
          // fill missing export
          for (k = 0; k < exp_nz; k++)
          {
            int exp_index = (i - ix + ghost_size_i_lower) +
                            ((exp_nz - k - 1) * exp_nx) +
                            ((j - iy + ghost_size_j_lower) * (exp_nx * exp_nz));
            exp_array[exp_index] = (float)(-1.0e34);
          }
        }
      }
    }
  }
}


/*--------------------------------------------------------------------------
 * Local Decomposition
 *--------------------------------------------------------------------------*/

void cplparflowlcldecomp_(int * sg,
                          int * lowerx,
                          int * upperx,
                          int * lowery,
                          int * uppery,
                          int * ierror)
{
  Grid *grid = GetGrid2DRichards(amps_ThreadLocal(solver));
  int subgridcount = SubgridArraySize(GridSubgrids(grid));

  if (*sg < 0 || *sg > (subgridcount - 1))
  {
    *lowerx = 0;
    *upperx = 0;
    *lowery = 0;
    *uppery = 0;
    *ierror = 22;
  }
  else
  {
    Subgrid *subgrid = GridSubgrid(grid, *sg);
    *lowerx = SubgridIX(subgrid);
    *upperx = *lowerx + SubgridNX(subgrid) - 1;
    *lowery = SubgridIY(subgrid);
    *uppery = *lowery + SubgridNY(subgrid) - 1;
    *ierror = 0;
  }
}

/*--------------------------------------------------------------------------
 * Local Watershed Mask
 *--------------------------------------------------------------------------*/

void cplparflowlclmask_(int * sg,
                        int * localmask,
                        int * ierror)
{
  Grid *grid = GetGrid2DRichards(amps_ThreadLocal(solver));
  int subgridcount = SubgridArraySize(GridSubgrids(grid));
  Vector *mask = GetMaskRichards(amps_ThreadLocal(solver));

  if (*sg < 0 || *sg > (subgridcount - 1))
  {
    *ierror = 22;
  }
  else
  {
    Subgrid *subgrid = GridSubgrid(grid, *sg);
    int ix = SubgridIX(subgrid);
    int iy = SubgridIY(subgrid);
    int nx = SubgridNX(subgrid);
    int ny = SubgridNY(subgrid);

    Subvector *mask_subvector = VectorSubvector(mask, *sg);
    double    *mask_data = SubvectorData(mask_subvector);

    int i, j;

    for (i = ix; i < ix + nx; i++)
    {
      for (j = iy; j < iy + ny; j++)
      {
        int localmask_index = (i - ix) + ((j - iy) * nx);
        int mask_index = SubvectorEltIndex(mask_subvector, i, j, 0);
        if (mask_data[mask_index] > 0)
        {
          localmask[localmask_index] = 1;
        }
        else
        {
          localmask[localmask_index] = 0;
        }
      }
    }
    *ierror = 0;
  }
}

/*--------------------------------------------------------------------------
 * Local Cartesian Coordinates Center
 *--------------------------------------------------------------------------*/

void cplparflowlclxyctr_(int *   sg,
                         float * localx,
                         float * localy,
                         int *   ierror)
{
  Grid *grid = GetGrid2DRichards(amps_ThreadLocal(solver));
  int subgridcount = SubgridArraySize(GridSubgrids(grid));

  if (*sg < 0 || *sg > (subgridcount - 1))
  {
    *ierror = 22;
  }
  else
  {
    Subgrid *subgrid = GridSubgrid(grid, *sg);
    int nx = SubgridNX(subgrid);
    int ny = SubgridNY(subgrid);
    float lx = SubgridX(subgrid);
    float ly = SubgridY(subgrid);
    float dx = SubgridDX(subgrid);
    float dy = SubgridDY(subgrid);

    int i, j;

    for (i = 0; i < nx; i++)
    {
      for (j = 0; j < ny; j++)
      {
        int local_index = i + (j * nx);
        localx[local_index] = lx + (dx * (float)i);
        localy[local_index] = ly + (dy * (float)j);
      }
    }
    *ierror = 0;
  }
}

/*--------------------------------------------------------------------------
 * Local Cartesian Coordinates Edge
 *--------------------------------------------------------------------------*/

void cplparflowlclxyedg_(int *   sg,
                         float * localx,
                         float * localy,
                         int *   ierror)
{
  Grid *grid = GetGrid2DRichards(amps_ThreadLocal(solver));
  int subgridcount = SubgridArraySize(GridSubgrids(grid));

  if (*sg < 0 || *sg > (subgridcount - 1))
  {
    *ierror = 22;
  }
  else
  {
    Subgrid *subgrid = GridSubgrid(grid, *sg);
    int nx = SubgridNX(subgrid);
    int ny = SubgridNY(subgrid);
    float lx = SubgridX(subgrid);
    float ly = SubgridY(subgrid);
    float dx = SubgridDX(subgrid);
    float dy = SubgridDY(subgrid);

    int i, j;

    for (i = 0; i < nx + 1; i++)
    {
      for (j = 0; j < ny + 1; j++)
      {
        int local_index = i + (j * (nx + 1));
        localx[local_index] = lx + (dx * ((float)i - 0.5));
        localy[local_index] = ly + (dy * ((float)j - 0.5));
      }
    }
    *ierror = 0;
  }
}
