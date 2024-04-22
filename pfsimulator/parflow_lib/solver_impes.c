/*BHEADER**********************************************************************

  Copyright (c) 1995-2024, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER);
        }
      }

      /*----------------------------------------------------------------
       * Log this step
       *----------------------------------------------------------------*/

      IfLogging(1)
      {
        seq_log[number_logged] = iteration_number;
        time_log[number_logged] = t;
        dt_log[number_logged] = dt;
        dt_info_log[number_logged] = dt_info;
        if (any_file_dumped)
          dumped_log[number_logged] = file_number;
        else
          dumped_log[number_logged] = -1;
        if (recompute_pressure)
          recomp_log[number_logged] = 'y';
        else
          recomp_log[number_logged] = 'n';
        number_logged++;
      }

      if (any_file_dumped || pressure_file_dumped)
      {
        file_number++;
      }

      any_file_dumped = 0;
      pressure_file_dumped = 0;
    }
    else
    {
      if (print_press)
      {
        sprintf(file_postfix, "press");
        WritePFBinary(file_prefix, file_postfix, pressure);
      }

      if (public_xtra->write_silo_press)
      {
	strcpy(file_postfix, "");
        sprintf(file_type, "press");
        WriteSilo(file_prefix, file_type, file_postfix, pressure,
                  t, file_number, "Pressure");
      }
    }
  }
  while (still_evolving);

  /***************************************************************/
  /*                 Print the pressure and saturation           */
  /***************************************************************/

  /* Dump the pressure values at end if requested */
  if (ProblemDumpAtEnd(problem))
  {
    if (public_xtra->print_press)
    {
      sprintf(file_postfix, "press.%05d", file_number);
      WritePFBinary(file_prefix, file_postfix, pressure);
      any_file_dumped = 1;
    }

    if (public_xtra->write_silo_press)
    {
      sprintf(file_postfix, "%05d", file_number);
      sprintf(file_type, "press");
      WriteSilo(file_prefix, file_type, file_postfix, pressure,
                t, file_number, "Pressure");
      any_file_dumped = 1;
    }

    for (phase = 0; phase < ProblemNumPhases(problem); phase++)
    {
      if (print_satur)
      {
        sprintf(file_postfix, "satur.%01d.%05d", phase, file_number);
        WritePFBinary(file_prefix, file_postfix, saturations[phase]);
        any_file_dumped = 1;
      }

      if (public_xtra->write_silo_satur)
      {
        sprintf(file_postfix, "%01d.%05d", phase, file_number);
        sprintf(file_type, "satur");
        WriteSilo(file_prefix, file_type, file_postfix, saturations[phase],
                  t, file_number, "Saturation");
        any_file_dumped = 1;
      }
    }


    if (ProblemNumContaminants(problem) > 0)
    {
      indx = 0;
      for (phase = 0; phase < ProblemNumPhases(problem); phase++)
      {
        for (concen = 0; concen < ProblemNumContaminants(problem); concen++)
        {
          if (print_concen)
          {
            sprintf(file_postfix, "concen.%01d.%02d.%05d", phase, concen, file_number);
            WritePFSBinary(file_prefix, file_postfix,
                           concentrations[indx], drop_tol);

            any_file_dumped = 1;
          }

          if (public_xtra->write_silo_concen)
          {
            sprintf(file_postfix, "%01d.%02d.%05d", phase, concen, file_number);
            sprintf(file_type, "concen");
            WriteSilo(file_prefix, file_type, file_postfix, concentrations[indx],
                      t, file_number, "Concentration");
            any_file_dumped = 1;
          }

          indx++;
        }
      }
    }
  }

  if (transient)
  {
    free(phase_dt);

    if (is_multiphase)
    {
      FreeVector(z_permeability);

      FreeVector(total_z_velocity);
      FreeVector(total_y_velocity);
      FreeVector(total_x_velocity);

      FreeEvalStruct(eval_struct);
    }

    for (phase = 0; phase < ProblemNumPhases(problem); phase++)
    {
      FreeVector(phase_z_velocity[phase]);
    }
    tfree(phase_z_velocity);

    for (phase = 0; phase < ProblemNumPhases(problem); phase++)
    {
      FreeVector(phase_y_velocity[phase]);
    }
    tfree(phase_y_velocity);

    for (phase = 0; phase < ProblemNumPhases(problem); phase++)
    {
      FreeVector(phase_x_velocity[phase]);
    }
    tfree(phase_x_velocity);

    if (ProblemNumContaminants(problem) > 0)
    {
      indx = 0;
      for (phase = 0; phase < ProblemNumPhases(problem); phase++)
      {
        for (concen = 0; concen < ProblemNumContaminants(problem); concen++)
        {
          FreeVector(concentrations[indx]);
          indx++;
        }
      }
      tfree(concentrations);

      FreeVector(solidmassfactor);
    }
  }

  if (is_multiphase)
  {
    for (phase = 0; phase < ProblemNumPhases(problem); phase++)
    {
      FreeVector(saturations[phase]);
    }
  }
  tfree(saturations);
  tfree(phase_densities);


  /*-------------------------------------------------------------------
   * Free temp vectors
   *-------------------------------------------------------------------*/
  if (is_multiphase)
  {
    FreeVector(stemp);
    FreeVector(temp_mobility_x);
    FreeVector(temp_mobility_y);
    FreeVector(temp_mobility_z);
  }
  FreeVector(ctemp);

  FreeVector(total_mobility_x);
  FreeVector(total_mobility_y);
  FreeVector(total_mobility_z);
  FreeVector(pressure);

  if (!amps_Rank(amps_CommWorld))
  {
    PrintWellData(ProblemDataWellData(problem_data),
                  (WELLDATA_PRINTSTATS));
  }

  /*----------------------------------------------------------------------
   * Print log
   *----------------------------------------------------------------------*/

  IfLogging(1)
  {
    FILE*  log_file;
    int k;

    log_file = OpenLogFile("SolverImpes");

    if (transient)
    {
      fprintf(log_file, "Transient Problem Solved.\n");
      fprintf(log_file, "-------------------------\n");
      fprintf(log_file, "Sequence #       Time         \\Delta t         Dumpfile #   Recompute?\n");
      fprintf(log_file, "----------   ------------   ------------ -     ----------   ----------\n");

      for (k = 0; k < number_logged; k++)
      {
        if (dumped_log[k] == -1)
          fprintf(log_file, "  %06d     %8e   %8e %1c                       %1c\n",
                  k, time_log[k], dt_log[k], dt_info_log[k], recomp_log[k]);
        else
          fprintf(log_file, "  %06d     %8e   %8e %1c       %06d          %1c\n",
                  k, time_log[k], dt_log[k], dt_info_log[k], dumped_log[k], recomp_log[k]);
      }
    }
    else
    {
      fprintf(log_file, "Non-Transient Problem Solved.\n");
      fprintf(log_file, "-----------------------------\n");
    }

    CloseLogFile(log_file);

    tfree(seq_log);
    tfree(time_log);
    tfree(dt_log);
    tfree(dt_info_log);
    tfree(dumped_log);
    tfree(recomp_log);
  }
}

/*-------------------------------------------------------------------------
 * SolverImpesInitInstanceXtra
 *-------------------------------------------------------------------------*/

PFModule *SolverImpesInitInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;

  Problem      *problem = (public_xtra->problem);

  Grid         *grid;
  Grid         *grid2d;
  Grid         *x_grid;
  Grid         *y_grid;
  Grid         *z_grid;

  SubgridArray *new_subgrids;
  SubgridArray *all_subgrids, *new_all_subgrids;

  Subgrid      *subgrid, *new_subgrid;

  double       *temp_data, *temp_data_placeholder;
  // SGS TODO total_mobility_sz is not being set anywhere so initialized to 0 here.
  int total_mobility_sz = 0;
  int pressure_sz, velocity_sz, satur_sz = 0,
    concen_sz, temp_data_size, sz;
  int is_multiphase;

  int i;


  is_multiphase = ProblemNumPhases(problem) > 1;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-------------------------------------------------------------------
   * Create the grids
   *-------------------------------------------------------------------*/

  /* Create the flow grid */
  grid = CreateGrid(GlobalsUserGrid);

  /*sk: Create a two-dimensional grid for later use*/

  all_subgrids = GridAllSubgrids(grid);

  new_all_subgrids = NewSubgridArray();
  ForSubgridI(i, all_subgrids)
  {
    subgrid = SubgridArraySubgrid(all_subgrids, i);
    new_subgrid = DuplicateSubgrid(subgrid);
    SubgridNZ(new_subgrid) = 1;
    AppendSubgrid(new_subgrid, new_all_subgrids);
  }
  new_subgrids = GetGridSubgrids(new_all_subgrids);
  grid2d = NewGrid(new_subgrids, new_all_subgrids);
  CreateComputePkgs(grid2d);

  // SGS Debug
  globals->grid2d = grid2d;

  /* Create the x velocity grid */

  all_subgrids = GridAllSubgrids(grid);

  /***** Set up a new subgrid grown by one in the x-direction *****/

  new_all_subgrids = NewSubgridArray();
  ForSubgridI(i, all_subgrids)
  {
    subgrid = SubgridArraySubgrid(all_subgrids, i);
    new_subgrid = DuplicateSubgrid(subgrid);
    SubgridNX(new_subgrid) += 1;
    AppendSubgrid(new_subgrid, new_all_subgrids);
  }
  new_subgrids = GetGridSubgrids(new_all_subgrids);
  x_grid = NewGrid(new_subgrids, new_all_subgrids);
  CreateComputePkgs(x_grid);

  /* Create the y velocity grid */

  all_subgrids = GridAllSubgrids(grid);

  /***** Set up a new subgrid grown by one in the y-direction *****/

  new_all_subgrids = NewSubgridArray();
  ForSubgridI(i, all_subgrids)
  {
    subgrid = SubgridArraySubgrid(all_subgrids, i);
    new_subgrid = DuplicateSubgrid(subgrid);
    SubgridNY(new_subgrid) += 1;
    AppendSubgrid(new_subgrid, new_all_subgrids);
  }
  new_subgrids = GetGridSubgrids(new_all_subgrids);
  y_grid = NewGrid(new_subgrids, new_all_subgrids);
  CreateComputePkgs(y_grid);

  /* Create the z velocity grid */

  all_subgrids = GridAllSubgrids(grid);

  /***** Set up a new subgrid grown by one in the z-direction *****/

  new_all_subgrids = NewSubgridArray();
  ForSubgridI(i, all_subgrids)
  {
    subgrid = SubgridArraySubgrid(all_subgrids, i);
    new_subgrid = DuplicateSubgrid(subgrid);
    SubgridNZ(new_subgrid) += 1;
    AppendSubgrid(new_subgrid, new_all_subgrids);
  }
  new_subgrids = GetGridSubgrids(new_all_subgrids);
  z_grid = NewGrid(new_subgrids, new_all_subgrids);
  CreateComputePkgs(z_grid);

  (instance_xtra->grid) = grid;
  (instance_xtra->grid2d) = grid2d;
  (instance_xtra->x_grid) = x_grid;
  (instance_xtra->y_grid) = y_grid;
  (instance_xtra->z_grid) = z_grid;

  /*-------------------------------------------------------------------
   * Create problem_data
   *-------------------------------------------------------------------*/

  (instance_xtra->problem_data) = NewProblemData(grid, grid2d);

  /*-------------------------------------------------------------------
   * Initialize module instances
   *-------------------------------------------------------------------*/

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->discretize_pressure) =
      PFModuleNewInstanceType(DiscretizePressureInitInstanceXtraInvoke,
                              (public_xtra->discretize_pressure),
                              (problem, grid, NULL));
    (instance_xtra->diag_scale) =
      PFModuleNewInstanceType(MatrixDiagScaleInitInstanceXtraInvoke,
                              (public_xtra->diag_scale),
                              (grid));
    (instance_xtra->linear_solver) =
      PFModuleNewInstanceType(LinearSolverInitInstanceXtraInvoke,
                              (public_xtra->linear_solver),
                              (problem, grid, NULL, NULL, NULL));
    (instance_xtra->phase_velocity_face) =
      PFModuleNewInstanceType(PhaseVelocityFaceInitInstanceXtraInvoke,
                              (public_xtra->phase_velocity_face),
                              (problem, grid, x_grid, y_grid, z_grid, NULL));
    (instance_xtra->advect_concen) =
      PFModuleNewInstanceType(AdvectionConcentrationInitInstanceXtraType,
                              (public_xtra->advect_concen),
                              (problem, grid, NULL));
    (instance_xtra->set_problem_data) =
      PFModuleNewInstanceType(SetProblemDataInitInstanceXtraInvoke,
                              (public_xtra->set_problem_data),
                              (problem, grid, NULL, NULL));

    (instance_xtra->retardation) =
      PFModuleNewInstanceType(RetardationInitInstanceXtraInvoke,
                              ProblemRetardation(problem), (NULL));
    (instance_xtra->phase_mobility) =
      PFModuleNewInstance(ProblemPhaseMobility(problem), ());
    (instance_xtra->ic_phase_concen) =
      PFModuleNewInstance(ProblemICPhaseConcen(problem), ());
    (instance_xtra->phase_density) =
      PFModuleNewInstance(ProblemPhaseDensity(problem), ());


    if (is_multiphase)
    {
      (instance_xtra->permeability_face) =
        PFModuleNewInstanceType(PermeabilityFaceInitInstanceXtraInvoke,
                                (public_xtra->permeability_face),
                                (z_grid));
      (instance_xtra->total_velocity_face) =
        PFModuleNewInstanceType(TotalVelocityFaceInitInstanceXtraInvoke,
                                public_xtra->total_velocity_face,
                                (problem, grid, x_grid, y_grid, z_grid,
                                 NULL));
      (instance_xtra->advect_satur) =
        PFModuleNewInstanceType(AdvectionSaturationInitInstanceXtraInvoke,
                                (public_xtra->advect_satur),
                                (problem, grid, NULL));

      (instance_xtra->ic_phase_satur) =
        PFModuleNewInstance(ProblemICPhaseSatur(problem), ());
      (instance_xtra->bc_phase_saturation) =
        PFModuleNewInstance(ProblemBCPhaseSaturation(problem), ());
      (instance_xtra->constitutive) =
        PFModuleNewInstanceType(SaturationConstitutiveInitInstanceXtraInvoke,
                                ProblemSaturationConstitutive(problem),
                                (grid));
    }
  }
  else
  {
    PFModuleReNewInstanceType(DiscretizePressureInitInstanceXtraInvoke,
                              (instance_xtra->discretize_pressure),
                              (problem, grid, NULL));
    PFModuleReNewInstanceType(MatrixDiagScaleInitInstanceXtraInvoke,
                              (instance_xtra->diag_scale),
                              (grid));
    PFModuleReNewInstanceType(LinearSolverInitInstanceXtraInvoke,
                              (instance_xtra->linear_solver),
                              (problem, grid, NULL, NULL, NULL));
    PFModuleReNewInstanceType(PhaseVelocityFaceInitInstanceXtraInvoke,
                              (instance_xtra->phase_velocity_face),
                              (problem, grid, x_grid, y_grid, z_grid, NULL));
    PFModuleReNewInstanceType(AdvectionConcentrationInitInstanceXtraType,
                              (instance_xtra->advect_concen),
                              (problem, grid, NULL));
    PFModuleReNewInstanceType(SetProblemDataInitInstanceXtraInvoke,
                              (instance_xtra->set_problem_data),
                              (problem, grid, NULL, NULL));
    PFModuleReNewInstanceType(RetardationInitInstanceXtraInvoke,
                              (instance_xtra->retardation), (NULL));
    PFModuleReNewInstance((instance_xtra->phase_mobility), ());
    PFModuleReNewInstance((instance_xtra->ic_phase_concen), ());
    PFModuleReNewInstance((instance_xtra->phase_density), ());

    if (is_multiphase)
    {
      PFModuleReNewInstanceType(PermeabilityFaceInitInstanceXtraInvoke,
                                (instance_xtra->permeability_face),
                                (z_grid));
      PFModuleReNewInstanceType(TotalVelocityFaceInitInstanceXtraInvoke,
                                (instance_xtra->total_velocity_face),
                                (problem, grid, x_grid, y_grid, z_grid,
                                 NULL));
      PFModuleReNewInstanceType(AdvectionSaturationInitInstanceXtraInvoke,
                                (instance_xtra->advect_satur),
                                (problem, grid, NULL));
      PFModuleReNewInstance((instance_xtra->ic_phase_satur), ());
      PFModuleReNewInstance((instance_xtra->bc_phase_saturation), ());
      PFModuleReNewInstanceType(SaturationConstitutiveInitInstanceXtraInvoke,
                                (instance_xtra->constitutive), (grid));
    }
  }

  /*-------------------------------------------------------------------
   * Set up temporary data
   *-------------------------------------------------------------------*/

  if (is_multiphase)
  {
    /* compute size for total mobility computation */
    sz = 0;

    /* compute size for saturation advection */
    sz = 0;
    sz += PFModuleSizeOfTempData(instance_xtra->advect_satur);
    satur_sz = sz;
  }

  /* compute size for pressure solve */
  sz = 0;
  sz = pfmax(sz,
             PFModuleSizeOfTempData(instance_xtra->discretize_pressure));
  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->linear_solver));
  pressure_sz = sz;

  /* compute size for velocity computation */
  sz = 0;
  sz = pfmax(sz,
             PFModuleSizeOfTempData(instance_xtra->phase_velocity_face));
  if (is_multiphase)
  {
    sz = pfmax(sz,
               PFModuleSizeOfTempData(instance_xtra->total_velocity_face));
  }
  velocity_sz = sz;

  /* compute size for concentration advection */
  sz = 0;
  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->retardation));
  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->advect_concen));
  concen_sz = sz;

  /* set temp_data size to max of pressure_sz, satur_sz, and concen_sz*/
  temp_data_size = pfmax(pfmax(pressure_sz, velocity_sz), concen_sz);
  if (is_multiphase)
  {
    temp_data_size = pfmax(temp_data_size, pfmax(total_mobility_sz, satur_sz));
  }
/*     temp_data_size = total_mobility_sz + pressure_sz + velocity_sz
 *                      + satur_sz + concen_sz;  */

  /* allocate temporary data */
  temp_data = NewTempData(temp_data_size);
  (instance_xtra->temp_data) = temp_data;

  /* renew set_problem_data module */
  PFModuleReNewInstanceType(SetProblemDataInitInstanceXtraInvoke,
                            (instance_xtra->set_problem_data),
                            (NULL, NULL, NULL, temp_data));

  /* renew pressure solve modules that take temporary data */
  PFModuleReNewInstanceType(DiscretizePressureInitInstanceXtraInvoke,
                            (instance_xtra->discretize_pressure),
                            (NULL, NULL, temp_data));
/*   temp_data +=
 *          PFModuleSizeOfTempData(instance_xtra -> discretize_pressure);  */
  PFModuleReNewInstanceType(LinearSolverInitInstanceXtraInvoke,
                            (instance_xtra->linear_solver),
                            (NULL, NULL, NULL, NULL, temp_data));
/*   temp_data += PFModuleSizeOfTempData(instance_xtra -> linear_solver);  */

  /* renew velocity computation modules that take temporary data */
  PFModuleReNewInstanceType(PhaseVelocityFaceInitInstanceXtraInvoke,
                            (instance_xtra->phase_velocity_face),
                            (NULL, NULL, NULL, NULL, NULL, temp_data));
  if (is_multiphase)
  {
    PFModuleReNewInstanceType(TotalVelocityFaceInitInstanceXtraInvoke,
                              (instance_xtra->total_velocity_face),
                              (NULL, NULL, NULL, NULL, NULL, temp_data));
    /* temp_data += PFModuleSizeOfTempData(instance_xtra ->
     *                                      total_velocity_face);  */

    /* renew saturation advection modules that take temporary data */
    temp_data_placeholder = temp_data;
    PFModuleReNewInstanceType(AdvectionSaturationInitInstanceXtraInvoke,
                              (instance_xtra->advect_satur),
                              (NULL, NULL, temp_data_placeholder));
    temp_data_placeholder +=
      PFModuleSizeOfTempData(instance_xtra->advect_satur);
  }

  /* renew concentration advection modules that take temporary data */
  temp_data_placeholder = temp_data;
  PFModuleReNewInstanceType(RetardationInitInstanceXtraInvoke,
                            (instance_xtra->retardation),
                            (temp_data_placeholder));
  PFModuleReNewInstanceType(AdvectionConcentrationInitInstanceXtraType,
                            (instance_xtra->advect_concen),
                            (NULL, NULL, temp_data_placeholder));

  int size_retardation = PFModuleSizeOfTempData(instance_xtra->retardation);
  int size_advect = PFModuleSizeOfTempData(instance_xtra->advect_concen);
  temp_data_placeholder += pfmax(
                                 size_retardation,
				 size_advect
                                 );
  /* set temporary vector data used for advection */

  temp_data += temp_data_size;

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * SolverImpesFreeInstanceXtra
 *-------------------------------------------------------------------------*/

void  SolverImpesFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  Problem       *problem = (public_xtra->problem);
  int is_multiphase;

  is_multiphase = ProblemNumPhases(problem) > 1;

  if (instance_xtra)
  {
    FreeTempData((instance_xtra->temp_data));

    PFModuleFreeInstance((instance_xtra->ic_phase_concen));
    PFModuleFreeInstance((instance_xtra->phase_mobility));
    PFModuleFreeInstance((instance_xtra->retardation));

    PFModuleFreeInstance((instance_xtra->set_problem_data));
    PFModuleFreeInstance((instance_xtra->advect_concen));
    PFModuleFreeInstance((instance_xtra->phase_velocity_face));
    PFModuleFreeInstance((instance_xtra->linear_solver));
    PFModuleFreeInstance((instance_xtra->diag_scale));
    PFModuleFreeInstance((instance_xtra->discretize_pressure));

    PFModuleFreeInstance((instance_xtra->phase_density));

    if (is_multiphase)
    {
      PFModuleFreeInstance((instance_xtra->constitutive));
      PFModuleFreeInstance((instance_xtra->bc_phase_saturation));
      PFModuleFreeInstance((instance_xtra->ic_phase_satur));
      PFModuleFreeInstance((instance_xtra->advect_satur));
      PFModuleFreeInstance((instance_xtra->total_velocity_face));
      PFModuleFreeInstance((instance_xtra->permeability_face));
    }

    FreeProblemData((instance_xtra->problem_data));

    FreeGrid((instance_xtra->z_grid));
    FreeGrid((instance_xtra->y_grid));
    FreeGrid((instance_xtra->x_grid));
    FreeGrid((instance_xtra->grid2d));
    FreeGrid((instance_xtra->grid));

    tfree(instance_xtra);
  }
}

/*-------------------------------------------------------------------------
 * SolverImpesNewPublicXtra
 *-------------------------------------------------------------------------*/

PFModule   *SolverImpesNewPublicXtra(char *name)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  char          *switch_name;
  int switch_value;
  NameArray switch_na;

  char key[IDB_MAX_KEY_LEN];

  NameArray diag_solver_na;
  NameArray linear_solver_na;

  switch_na = NA_NewNameArray("False True");

  public_xtra = ctalloc(PublicXtra, 1);

  diag_solver_na = NA_NewNameArray("NoDiagScale MatDiagScale");
  sprintf(key, "%s.DiagScale", name);
  switch_name = GetStringDefault(key, "NoDiagScale");
  switch_value = NA_NameToIndexExitOnError(diag_solver_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      (public_xtra->diag_scale) = PFModuleNewModuleType(MatrixDiagScaleNewPublicXtraInvoke,
                                                        NoDiagScale, (key));
      break;
    }

    case 1:
    {
      (public_xtra->diag_scale) = PFModuleNewModuleType(MatrixDiagScaleNewPublicXtraInvoke,
                                                        MatDiagScale, (key));
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(diag_solver_na);

  linear_solver_na = NA_NewNameArray("MGSemi PPCG PCG CGHS");
  sprintf(key, "%s.Linear", name);
  switch_name = GetStringDefault(key, "PCG");
  switch_value = NA_NameToIndexExitOnError(linear_solver_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      (public_xtra->linear_solver) = PFModuleNewModuleType(LinearSolverNewPublicXtraInvoke,
                                                           MGSemi, (key));
      break;
    }

    case 1:
    {
      (public_xtra->linear_solver) = PFModuleNewModuleType(LinearSolverNewPublicXtraInvoke,
                                                           PPCG, (key));
      break;
    }

    case 2:
    {
      (public_xtra->linear_solver) = PFModuleNewModuleType(LinearSolverNewPublicXtraInvoke,
                                                           PCG, (key));
      break;
    }

    case 3:
    {
      (public_xtra->linear_solver) = PFModuleNewModuleType(LinearSolverNewPublicXtraInvoke,
                                                           CGHS, (key));
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(linear_solver_na);

  (public_xtra->discretize_pressure) =
    PFModuleNewModule(DiscretizePressure, ());

  (public_xtra->permeability_face) = PFModuleNewModule(PermeabilityFace,
                                                       ());
  (public_xtra->phase_velocity_face) = PFModuleNewModule(
                                                         PhaseVelocityFace, ());
  (public_xtra->total_velocity_face) = PFModuleNewModuleType(TotalVelocityFaceInitInstanceXtraInvoke,
                                                             TotalVelocityFace,
                                                             (NULL, NULL, NULL, NULL, NULL, NULL));

  (public_xtra->advect_satur) = PFModuleNewModule(SatGodunov, ());
  (public_xtra->advect_concen) = PFModuleNewModule(Godunov, ());
  (public_xtra->set_problem_data) = PFModuleNewModule(SetProblemData, ());

  (public_xtra->problem) = NewProblem(ImpesSolve);

  sprintf(key, "%s.SadvectOrder", name);
  public_xtra->sadvect_order = GetIntDefault(key, 2);

  sprintf(key, "%s.AdvectOrder", name);
  public_xtra->advect_order = GetIntDefault(key, 2);

  sprintf(key, "%s.CFL", name);
  public_xtra->CFL = GetDoubleDefault(key, 0.7);

  sprintf(key, "%s.MaxIter", name);
  public_xtra->max_iterations = GetIntDefault(key, 1000000);

  sprintf(key, "%s.RelTol", name);
  public_xtra->rel_tol = GetDoubleDefault(key, 1.0);

  sprintf(key, "%s.AbsTol", name);
  public_xtra->abs_tol = GetDoubleDefault(key, 1E-9);

  sprintf(key, "%s.Drop", name);
  public_xtra->drop_tol = GetDoubleDefault(key, 1E-8);

  sprintf(key, "%s.PrintSubsurf", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_subsurf_data = switch_value;

  sprintf(key, "%s.PrintPressure", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_press = switch_value;

  sprintf(key, "%s.PrintVelocities", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_velocities = switch_value;


  sprintf(key, "%s.PrintSaturation", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_satur = switch_value;

  sprintf(key, "%s.PrintConcentration", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_concen = switch_value;

  sprintf(key, "%s.PrintWells", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_wells = switch_value;

  /* Silo file writing control */
  sprintf(key, "%s.WriteSiloSubsurfData", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_subsurf_data = switch_value;

  sprintf(key, "%s.WriteSiloPressure", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_press = switch_value;

  sprintf(key, "%s.WriteSiloVelocities", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_velocities = switch_value;

  sprintf(key, "%s.WriteSiloSaturation", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_satur = switch_value;

  sprintf(key, "%s.WriteSiloConcentration", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_concen = switch_value;

  if (public_xtra->write_silo_subsurf_data ||
      public_xtra->write_silo_press ||
      public_xtra->write_silo_velocities ||
      public_xtra->write_silo_satur ||
      public_xtra->write_silo_concen
      )
  {
    WriteSiloInit(GlobalsOutFileName);
  }

  NA_FreeNameArray(switch_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * SolverImpesFreePublicXtra
 *-------------------------------------------------------------------------*/

void   SolverImpesFreePublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    FreeProblem(public_xtra->problem, ImpesSolve);

    PFModuleFreeModule(public_xtra->diag_scale);
    PFModuleFreeModule(public_xtra->linear_solver);

    PFModuleFreeModule(public_xtra->set_problem_data);
    PFModuleFreeModule(public_xtra->advect_concen);
    PFModuleFreeModule(public_xtra->advect_satur);
    PFModuleFreeModule(public_xtra->total_velocity_face);
    PFModuleFreeModule(public_xtra->phase_velocity_face);
    PFModuleFreeModule(public_xtra->permeability_face);
    PFModuleFreeModule(public_xtra->discretize_pressure);
    tfree(public_xtra);
  }
}

/*-------------------------------------------------------------------------
 * SolverImpesSizeOfTempData
 *-------------------------------------------------------------------------*/

int  SolverImpesSizeOfTempData()
{
  return 0;
}
