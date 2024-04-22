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

    /*-----------------------------------------------------------------
     * Log this step
     *-----------------------------------------------------------------*/

    IfLogging(1)
    {
      /*
       * SGS Better error handing should be added
       */
      if (instance_xtra->number_logged > public_xtra->max_iterations + 1)
      {
        amps_Printf
          ("Error: max_iterations reached, can't log anymore data\n");
        exit(1);
      }

      instance_xtra->seq_log[instance_xtra->number_logged] =
        instance_xtra->iteration_number;
      instance_xtra->time_log[instance_xtra->number_logged] = t;
      instance_xtra->dt_log[instance_xtra->number_logged] = dt;
      instance_xtra->dt_info_log[instance_xtra->number_logged] = dt_info;
      if (any_file_dumped || clm_file_dumped)
        instance_xtra->dumped_log[instance_xtra->number_logged] =
          instance_xtra->file_number;
      else
        instance_xtra->dumped_log[instance_xtra->number_logged] = -1;
      instance_xtra->recomp_log[instance_xtra->number_logged] = 'y';
      instance_xtra->number_logged++;
    }

    if (any_file_dumped || clm_file_dumped)
    {
      UpdateMetadata(this_module, GlobalsOutFileName, instance_xtra->file_number);

      instance_xtra->file_number++;
      any_file_dumped = 0;
      clm_file_dumped = 0;
    }

    if (take_more_time_steps)
    {
      take_more_time_steps =
        (instance_xtra->iteration_number < max_iterations)
        && (t < stop_time);
    }

#ifdef HAVE_SLURM
    /*
     * If at end of a dump_interval and user requests halt if
     * remaining time in job is less than user specified value.
     * Used to halt jobs gracefully when running on batch systems.
     */

    int dump_interval_execution_time_limit =
      ProblemDumpIntervalExecutionTimeLimit(problem);

    if (dump_files && dump_interval_execution_time_limit)
    {
      if (!amps_Rank(amps_CommWorld))
      {
        printf
          ("Checking execution time limit, interation = %d, remaining time = %ld (s)\n",
          instance_xtra->iteration_number, slurm_get_rem_time(0));
      }

      if (slurm_get_rem_time(0) <= dump_interval_execution_time_limit)
      {
        if (!amps_Rank(amps_CommWorld))
        {
          printf
            ("Remaining time less than supplied DumpIntervalExectionTimeLimit = %d, halting execution\n",
            dump_interval_execution_time_limit);
        }

        take_more_time_steps = 0;
      }
    }
#endif
    if(first_tstep)
    {
      BeginTiming(RichardsExclude1stTimeStepIndex);
      PUSH_NVTX("RichardsExclude1stTimeStepIndex",6)
      first_tstep = 0;
    }
  }                             /* ends do for time loop */
  while (take_more_time_steps);

  EndTiming(RichardsExclude1stTimeStepIndex);
  POP_NVTX

  /***************************************************************/
  /*                 Print the pressure and saturation           */
  /***************************************************************/

  /* Dump the pressure values at end if requested */
  if (ProblemDumpAtEnd(problem))
  {
    if (public_xtra->print_press)
    {
      sprintf(file_postfix, "press.%05d", instance_xtra->file_number);
      WritePFBinary(file_prefix, file_postfix, instance_xtra->pressure);
      any_file_dumped = 1;
    }

    if (public_xtra->write_silo_press)
    {
      sprintf(file_postfix, "%05d", instance_xtra->file_number);
      sprintf(file_type, "press");
      WriteSilo(file_prefix, file_type, file_postfix,
                instance_xtra->pressure, t, instance_xtra->file_number,
                "Pressure");
      any_file_dumped = 1;
    }

    if (print_satur)
    {
      sprintf(file_postfix, "satur.%05d", instance_xtra->file_number);
      WritePFBinary(file_prefix, file_postfix,
                    instance_xtra->saturation);
      any_file_dumped = 1;
    }

    if (public_xtra->write_silo_satur)
    {
      sprintf(file_postfix, "%05d", instance_xtra->file_number);
      sprintf(file_type, "satur");
      WriteSilo(file_prefix, file_type, file_postfix,
                instance_xtra->saturation, t, instance_xtra->file_number,
                "Saturation");
      any_file_dumped = 1;
    }

    if (public_xtra->print_evaptrans)
    {
      sprintf(file_postfix, "evaptrans.%05d",
              instance_xtra->file_number);
      WritePFBinary(file_prefix, file_postfix, evap_trans);
      any_file_dumped = 1;
    }

    if (public_xtra->write_silo_evaptrans)
    {
      sprintf(file_postfix, "%05d", instance_xtra->file_number);
      sprintf(file_type, "evaptrans");
      WriteSilo(file_prefix, file_type, file_postfix, evap_trans,
                t, instance_xtra->file_number, "EvapTrans");
      any_file_dumped = 1;
    }

    if (public_xtra->print_evaptrans_sum
        || public_xtra->write_silo_evaptrans_sum)
    {
      if (public_xtra->print_evaptrans_sum)
      {
        sprintf(file_postfix, "evaptranssum.%05d",
                instance_xtra->file_number);
        WritePFBinary(file_prefix, file_postfix, evap_trans_sum);
        any_file_dumped = 1;
      }

      if (public_xtra->write_silo_evaptrans_sum)
      {
        sprintf(file_postfix, "%05d", instance_xtra->file_number);
        sprintf(file_type, "evaptranssum");
        WriteSilo(file_prefix, file_type, file_postfix, evap_trans_sum,
                  t, instance_xtra->file_number, "EvapTransSum");
        any_file_dumped = 1;
      }

      /* reset sum after output */
      PFVConstInit(0.0, evap_trans_sum);
    }

    if (public_xtra->print_overland_sum
        || public_xtra->write_silo_overland_sum)
    {
      if (public_xtra->print_overland_sum)
      {
        sprintf(file_postfix, "overlandsum.%05d",
                instance_xtra->file_number);
        WritePFBinary(file_prefix, file_postfix, overland_sum);
        any_file_dumped = 1;
      }

      if (public_xtra->write_silo_overland_sum)
      {
        sprintf(file_postfix, "%05d", instance_xtra->file_number);
        sprintf(file_type, "overlandsum");
        WriteSilo(file_prefix, file_type, file_postfix, overland_sum,
                  t, instance_xtra->file_number, "OverlandSum");
        any_file_dumped = 1;
      }

      /* reset sum after output */
      PFVConstInit(0.0, overland_sum);
    }

    if (public_xtra->print_overland_bc_flux)
    {
      sprintf(file_postfix, "overland_bc_flux.%05d",
              instance_xtra->file_number);
      WritePFBinary(file_prefix, file_postfix,
                    instance_xtra->ovrl_bc_flx);
      any_file_dumped = 1;
    }

    if (public_xtra->write_silo_overland_bc_flux)
    {
      sprintf(file_postfix, "%05d", instance_xtra->file_number);
      sprintf(file_type, "overland_bc_flux");
      WriteSilo(file_prefix, file_type, file_postfix,
                instance_xtra->ovrl_bc_flx, t,
                instance_xtra->file_number, "OverlandBCFlux");
      any_file_dumped = 1;
    }

    // IMF: I assume this print obselete now that we have keys for EvapTrans and OverlandBCFlux?
    if (public_xtra->print_lsm_sink)
    {
      /*sk Print the sink terms from the land surface model */
      sprintf(file_postfix, "et.%05d", instance_xtra->file_number);
      WritePFBinary(file_prefix, file_postfix, evap_trans);

      /*sk Print the sink terms from the land surface model */
      sprintf(file_postfix, "obf.%05d", instance_xtra->file_number);
      WritePFBinary(file_prefix, file_postfix,
                    instance_xtra->ovrl_bc_flx);

      any_file_dumped = 1;
    }
  }

  *pressure_out = instance_xtra->pressure;
  *porosity_out = porosity;
  *saturation_out = instance_xtra->saturation;
}


void
TeardownRichards(PFModule * this_module)
{
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra *instance_xtra =
    (InstanceXtra*)PFModuleInstanceXtra(this_module);

  Problem *problem = (public_xtra->problem);
  ProblemData *problem_data = (instance_xtra->problem_data);

  int start_count = ProblemStartCount(problem);

  FinalizeMetadata(this_module, GlobalsOutFileName);

  FreeVector(instance_xtra->saturation);
  FreeVector(instance_xtra->density);
  FreeVector(instance_xtra->old_saturation);
  FreeVector(instance_xtra->old_pressure);
  FreeVector(instance_xtra->old_density);
  FreeVector(instance_xtra->pressure);
  FreeVector(instance_xtra->ovrl_bc_flx);
  FreeVector(instance_xtra->mask);

  FreeVector(instance_xtra->x_velocity);
  FreeVector(instance_xtra->y_velocity);
  FreeVector(instance_xtra->z_velocity);
  FreeVector(instance_xtra->evap_trans);

  if (instance_xtra->evap_trans_sum)
  {
    FreeVector(instance_xtra->evap_trans_sum);
  }

  if (instance_xtra->overland_sum)
  {
    FreeVector(instance_xtra->overland_sum);
  }

#ifdef HAVE_CLM
  if (instance_xtra->eflx_lh_tot)
  {
    if (instance_xtra->clm_out_grid)
    {
      FreeVector(instance_xtra->clm_out_grid);
    }

    FreeVector(instance_xtra->eflx_lh_tot);
    FreeVector(instance_xtra->eflx_lwrad_out);
    FreeVector(instance_xtra->eflx_sh_tot);
    FreeVector(instance_xtra->eflx_soil_grnd);
    FreeVector(instance_xtra->qflx_evap_tot);
    FreeVector(instance_xtra->qflx_evap_grnd);
    FreeVector(instance_xtra->qflx_evap_soi);
    FreeVector(instance_xtra->qflx_evap_veg);
    FreeVector(instance_xtra->qflx_tran_veg);
    FreeVector(instance_xtra->qflx_infl);
    FreeVector(instance_xtra->swe_out);
    FreeVector(instance_xtra->t_grnd);
    FreeVector(instance_xtra->tsoil);

    /*IMF Initialize variables for CLM irrigation output */
    FreeVector(instance_xtra->irr_flag);
    FreeVector(instance_xtra->qflx_qirr);
    FreeVector(instance_xtra->qflx_qirr_inst);
    /*IMF Initialize variables for CLM forcing fields
     * SW rad, LW rad, precip, T(air), U, V, P(air), q(air) */
    FreeVector(instance_xtra->sw_forc);
    FreeVector(instance_xtra->lw_forc);
    FreeVector(instance_xtra->prcp_forc);
    FreeVector(instance_xtra->tas_forc);
    FreeVector(instance_xtra->u_forc);
    FreeVector(instance_xtra->v_forc);
    FreeVector(instance_xtra->patm_forc);
    FreeVector(instance_xtra->qatm_forc);
    /*BH: added vegetation forcing variable & veg map */
    FreeVector(instance_xtra->lai_forc);
    FreeVector(instance_xtra->sai_forc);
    FreeVector(instance_xtra->z0m_forc);
    FreeVector(instance_xtra->displa_forc);
    FreeVector(instance_xtra->veg_map_forc);
  }


  if (public_xtra->sw1d)
  {
    tfree(public_xtra->sw1d);
    tfree(public_xtra->lw1d);
    tfree(public_xtra->prcp1d);
    tfree(public_xtra->tas1d);
    tfree(public_xtra->u1d);
    tfree(public_xtra->v1d);
    tfree(public_xtra->patm1d);
    tfree(public_xtra->qatm1d);
    /*BH: added vegetation forcing variable */
    tfree(public_xtra->lai1d);
    tfree(public_xtra->sai1d);
    tfree(public_xtra->z0m1d);
    tfree(public_xtra->displa1d);
  }
#endif

  if (!amps_Rank(amps_CommWorld))
  {
    PrintWellData(ProblemDataWellData(problem_data),
                  (WELLDATA_PRINTSTATS));
  }

  /*-----------------------------------------------------------------------
   * Print log
   *-----------------------------------------------------------------------*/

  IfLogging(1)
  {
    FILE *log_file;
    int k;

    log_file = OpenLogFile("SolverRichards");

    if (start_count >= 0)
    {
      fprintf(log_file, "Transient Problem Solved.\n");
      fprintf(log_file, "-------------------------\n");
      fprintf(log_file, "\n");
      fprintf(log_file, "Total Timesteps: %d\n",
              instance_xtra->number_logged - 1);
      fprintf(log_file, "\n");
      fprintf(log_file, "-------------------------\n");
      fprintf(log_file,
              "Sequence #       Time         \\Delta t         Dumpfile #   Recompute?\n");
      fprintf(log_file,
              "----------   ------------   ------------ -     ----------   ----------\n");

      for (k = 0; k < instance_xtra->number_logged; k++)
      {
        if (instance_xtra->dumped_log[k] == -1)
          fprintf(log_file,
                  "  %06d     %8e   %8e %1c                       %1c\n",
                  k, instance_xtra->time_log[k],
                  instance_xtra->dt_log[k],
                  instance_xtra->dt_info_log[k],
                  instance_xtra->recomp_log[k]);
        else
          fprintf(log_file,
                  "  %06d     %8e   %8e %1c       %06d          %1c\n",
                  k, instance_xtra->time_log[k],
                  instance_xtra->dt_log[k],
                  instance_xtra->dt_info_log[k],
                  instance_xtra->dumped_log[k],
                  instance_xtra->recomp_log[k]);
      }

      fprintf(log_file, "\n");
      fprintf(log_file, "Overland flow Results\n");
      for (k = 0; k < instance_xtra->number_logged; k++)        //sk start
      {
        if (instance_xtra->dumped_log[k] == -1)
          fprintf(log_file, "  %06d     %8e   %8e\n",
                  k, instance_xtra->time_log[k],
                  instance_xtra->dt_log[k]);
        else
          fprintf(log_file, "  %06d     %8e   %8e\n",
                  k, instance_xtra->time_log[k],
                  instance_xtra->dt_log[k]);
      }                         //sk end
    }
    else
    {
      fprintf(log_file, "Non-Transient Problem Solved.\n");
      fprintf(log_file, "-----------------------------\n");
    }

    CloseLogFile(log_file);

    tfree(instance_xtra->seq_log);
    tfree(instance_xtra->time_log);
    tfree(instance_xtra->dt_log);
    tfree(instance_xtra->dt_info_log);
    tfree(instance_xtra->dumped_log);
    tfree(instance_xtra->recomp_log);
  }
}

/*--------------------------------------------------------------------------
 * SolverRichardsInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule *
SolverRichardsInitInstanceXtra()
{
  PFModule *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra *instance_xtra;

  Problem *problem = (public_xtra->problem);

  Grid *grid;
  Grid *grid2d;
  Grid *x_grid;
  Grid *y_grid;
  Grid *z_grid;

#ifdef HAVE_CLM
  Grid *gridTs;
  Grid *metgrid;

  Grid *snglclm;                // NBE: New grid for CLM single file output
#endif

  SubgridArray *new_subgrids;
  SubgridArray *all_subgrids, *new_all_subgrids;
  Subgrid *subgrid, *new_subgrid;
  double *temp_data, *temp_data_placeholder;
  int concen_sz, ic_sz, velocity_sz, temp_data_size, sz;
  int nonlin_sz, parameter_sz;
  int i;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-------------------------------------------------------------------
   * Create the grids
   *-------------------------------------------------------------------*/

  /* Create the flow grid */
  grid = CreateGrid(GlobalsUserGrid);

  /*sk: Create a two-dimensional grid for later use */
  all_subgrids = GridAllSubgrids(grid);


  // SGS FIXME this is incorrect, can't loop over both at same time
  // assumes same grids in both arrays which is not correct?
  new_all_subgrids = NewSubgridArray();
  ForSubgridI(i, all_subgrids)
  {
    subgrid = SubgridArraySubgrid(all_subgrids, i);
    new_subgrid = DuplicateSubgrid(subgrid);
    SubgridIZ(new_subgrid) = 0;
    SubgridNZ(new_subgrid) = 1;
    AppendSubgrid(new_subgrid, new_all_subgrids);
  }
  new_subgrids = GetGridSubgrids(new_all_subgrids);
  grid2d = NewGrid(new_subgrids, new_all_subgrids);
  CreateComputePkgs(grid2d);

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

#ifdef HAVE_CLM
  /* IMF New grid for met forcing (nx*ny*nt) */
  /* NT specified by key CLM.MetForcing3D.NT */
  all_subgrids = GridAllSubgrids(grid);
  new_all_subgrids = NewSubgridArray();
  ForSubgridI(i, all_subgrids)
  {
    subgrid = SubgridArraySubgrid(all_subgrids, i);
    new_subgrid = DuplicateSubgrid(subgrid);
    SubgridIZ(new_subgrid) = 0;
    SubgridNZ(new_subgrid) = public_xtra->clm_metnt;
    AppendSubgrid(new_subgrid, new_all_subgrids);
  }
  new_subgrids = GetGridSubgrids(new_all_subgrids);
  metgrid = NewGrid(new_subgrids, new_all_subgrids);
  CreateComputePkgs(metgrid);
  (instance_xtra->metgrid) = metgrid;

  //NBE: Define the grid type only if it's required
  if (public_xtra->single_clm_file)
  {
    /* NBE - Create new grid for single file CLM output */
    all_subgrids = GridAllSubgrids(grid);
    new_all_subgrids = NewSubgridArray();
    ForSubgridI(i, all_subgrids)
    {
      subgrid = SubgridArraySubgrid(all_subgrids, i);
      new_subgrid = DuplicateSubgrid(subgrid);
      SubgridIZ(new_subgrid) = 0;
      SubgridNZ(new_subgrid) = 13 + public_xtra->clm_nz;
      AppendSubgrid(new_subgrid, new_all_subgrids);
    }
    new_subgrids = GetGridSubgrids(new_all_subgrids);
    snglclm = NewGrid(new_subgrids, new_all_subgrids);
    CreateComputePkgs(snglclm);
    (instance_xtra->snglclm) = snglclm;
  }

  /* IMF New grid for Tsoil (nx*ny*10) */
  all_subgrids = GridAllSubgrids(grid);
  new_all_subgrids = NewSubgridArray();
  ForSubgridI(i, all_subgrids)
  {
    subgrid = SubgridArraySubgrid(all_subgrids, i);
    new_subgrid = DuplicateSubgrid(subgrid);
    SubgridIZ(new_subgrid) = 0;
    //SubgridNZ(new_subgrid) = 10;
    SubgridNZ(new_subgrid) = public_xtra->clm_nz;       //NBE: Use variable # of soil layers
    AppendSubgrid(new_subgrid, new_all_subgrids);
  }
  new_subgrids = GetGridSubgrids(new_all_subgrids);
  gridTs = NewGrid(new_subgrids, new_all_subgrids);
  CreateComputePkgs(gridTs);
  (instance_xtra->gridTs) = gridTs;
#endif

  /*-------------------------------------------------------------------
   * Create problem_data
   *-------------------------------------------------------------------*/

  (instance_xtra->problem_data) = NewProblemData(grid, grid2d);

  /*-------------------------------------------------------------------
   * Initialize module instances
   *-------------------------------------------------------------------*/

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->advect_concen) =
      PFModuleNewInstanceType(AdvectionConcentrationInitInstanceXtraType,
                              (public_xtra->advect_concen),
                              (problem, grid, NULL));
    (instance_xtra->set_problem_data) =
      PFModuleNewInstanceType(SetProblemDataInitInstanceXtraInvoke,
                              (public_xtra->set_problem_data),
                              (problem, grid, grid2d, NULL));

    (instance_xtra->retardation) =
      PFModuleNewInstanceType(RetardationInitInstanceXtraInvoke,
                              ProblemRetardation(problem), (NULL));
    (instance_xtra->phase_rel_perm) =
      PFModuleNewInstanceType(PhaseRelPermInitInstanceXtraInvoke,
                              ProblemPhaseRelPerm(problem), (grid, NULL));
    (instance_xtra->ic_phase_concen) =
      PFModuleNewInstance(ProblemICPhaseConcen(problem), ());

    (instance_xtra->permeability_face) =
      PFModuleNewInstanceType(PermeabilityFaceInitInstanceXtraInvoke,
                              (public_xtra->permeability_face), (z_grid));

    (instance_xtra->ic_phase_pressure) =
      PFModuleNewInstanceType(ICPhasePressureInitInstanceXtraInvoke,
                              ProblemICPhasePressure(problem),
                              (problem, grid, NULL));
    (instance_xtra->problem_saturation) =
      PFModuleNewInstanceType(SaturationInitInstanceXtraInvoke,
                              ProblemSaturation(problem), (grid, NULL));
    (instance_xtra->phase_density) =
      PFModuleNewInstance(ProblemPhaseDensity(problem), ());
    (instance_xtra->select_time_step) =
      PFModuleNewInstance(ProblemSelectTimeStep(problem), ());
    (instance_xtra->l2_error_norm) =
      PFModuleNewInstance(ProblemL2ErrorNorm(problem), ());
    (instance_xtra->nonlin_solver) =
      PFModuleNewInstanceType(NonlinSolverInitInstanceXtraInvoke,
                              public_xtra->nonlin_solver,
                              (problem, grid, instance_xtra->problem_data,
                               NULL));
  }
  else
  {
    PFModuleReNewInstanceType(AdvectionConcentrationInitInstanceXtraType,
                              (instance_xtra->advect_concen),
                              (problem, grid, NULL));
    PFModuleReNewInstanceType(SetProblemDataInitInstanceXtraInvoke,
                              (instance_xtra->set_problem_data),
                              (problem, grid, grid2d, NULL));

    PFModuleReNewInstanceType(RetardationInitInstanceXtraInvoke,
                              (instance_xtra->retardation), (NULL));

    PFModuleReNewInstanceType(PhaseRelPermInitInstanceXtraInvoke,
                              (instance_xtra->phase_rel_perm), (grid,
                                                                NULL));
    PFModuleReNewInstance((instance_xtra->ic_phase_concen), ());

    PFModuleReNewInstanceType(PermeabilityFaceInitInstanceXtraInvoke,
                              (instance_xtra->permeability_face),
                              (z_grid));

    PFModuleReNewInstanceType(ICPhasePressureInitInstanceXtraInvoke,
                              (instance_xtra->ic_phase_pressure),
                              (problem, grid, NULL));
    PFModuleReNewInstanceType(SaturationInitInstanceXtraInvoke,
                              (instance_xtra->problem_saturation),
                              (grid, NULL));
    PFModuleReNewInstance((instance_xtra->phase_density), ());
    PFModuleReNewInstance((instance_xtra->select_time_step), ());
    PFModuleReNewInstance((instance_xtra->l2_error_norm), ());
    PFModuleReNewInstance((instance_xtra->nonlin_solver), ());
  }

  /*-------------------------------------------------------------------
   * Set up temporary data
   *-------------------------------------------------------------------*/

  /* May need the temp_mobility size for something later... */

  //sk: I don't have to do this for my instcances, because I allocate memory locally ?!

  /* compute size for velocity computation */
  sz = 0;
  /*   sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra -> phase_velocity_face)); */
  velocity_sz = sz;

  /* compute size for concentration advection */
  sz = 0;
  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->retardation));
  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->advect_concen));
  concen_sz = sz;

  /* compute size for pressure initial condition */
  ic_sz = PFModuleSizeOfTempData(instance_xtra->ic_phase_pressure);

  /* compute size for initial pressure guess */
  /*ig_sz = PFModuleSizeOfTempData(instance_xtra -> ig_phase_pressure); */

  /* Compute size for nonlinear solver */
  nonlin_sz = PFModuleSizeOfTempData(instance_xtra->nonlin_solver);

  /* Compute size for problem parameters */
  parameter_sz = PFModuleSizeOfTempData(instance_xtra->problem_saturation);
  parameter_sz  += PFModuleSizeOfTempData(instance_xtra->phase_rel_perm);

  /* set temp_data size to max of velocity_sz, concen_sz, and ic_sz. */
  /* The temp vector space for the nonlinear solver is added in because */
  /* at a later time advection may need to re-solve flow. */
  temp_data_size = parameter_sz
                   + pfmax(pfmax(pfmax(velocity_sz, concen_sz), nonlin_sz), ic_sz);

  /* allocate temporary data */
  temp_data = NewTempData(temp_data_size);
  (instance_xtra->temp_data) = temp_data;

  PFModuleReNewInstanceType(SaturationInitInstanceXtraInvoke,
                            (instance_xtra->problem_saturation),
                            (NULL, temp_data));
  temp_data += PFModuleSizeOfTempData(instance_xtra->problem_saturation);

  PFModuleReNewInstanceType(PhaseRelPermInitInstanceXtraInvoke,
                            (instance_xtra->phase_rel_perm),
                            (NULL, temp_data));
  temp_data += PFModuleSizeOfTempData(instance_xtra->phase_rel_perm);

  /* renew ic_phase_pressure module */
  PFModuleReNewInstanceType(ICPhasePressureInitInstanceXtraInvoke,
                            (instance_xtra->ic_phase_pressure),
                            (NULL, NULL, temp_data));

  /* renew nonlinear solver module */
  PFModuleReNewInstanceType(NonlinSolverInitInstanceXtraInvoke,
                            (instance_xtra->nonlin_solver),
                            (NULL, NULL, instance_xtra->problem_data,
                             temp_data));

  /* renew set_problem_data module */
  PFModuleReNewInstanceType(SetProblemDataInitInstanceXtraInvoke,
                            (instance_xtra->set_problem_data),
                            (NULL, NULL, NULL, temp_data));

  /* renew velocity computation modules that take temporary data */
  /*
   *  PFModuleReNewInstance((instance_xtra -> phase_velocity_face),
   *   (NULL, NULL, NULL, NULL, NULL, temp_data));
   */

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
  temp_data_placeholder += pfmax(size_retardation, size_advect);
 
  /* set temporary vector data used for advection */

  temp_data += temp_data_size;

  PFModuleInstanceXtra(this_module) = instance_xtra;

  return this_module;
}

/*--------------------------------------------------------------------------
 * SolverRichardsFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void
SolverRichardsFreeInstanceXtra()
{
  PFModule *this_module = ThisPFModule;
  InstanceXtra *instance_xtra =
    (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    FreeTempData((instance_xtra->temp_data));

    PFModuleFreeInstance((instance_xtra->ic_phase_concen));
    PFModuleFreeInstance((instance_xtra->phase_rel_perm));
    PFModuleFreeInstance((instance_xtra->retardation));

    PFModuleFreeInstance((instance_xtra->set_problem_data));
    PFModuleFreeInstance((instance_xtra->advect_concen));
    PFModuleFreeInstance((instance_xtra->ic_phase_pressure));
    PFModuleFreeInstance((instance_xtra->problem_saturation));
    PFModuleFreeInstance((instance_xtra->phase_density));
    PFModuleFreeInstance((instance_xtra->select_time_step));
    PFModuleFreeInstance((instance_xtra->l2_error_norm));
    PFModuleFreeInstance((instance_xtra->nonlin_solver));

    PFModuleFreeInstance((instance_xtra->permeability_face));

    FreeProblemData((instance_xtra->problem_data));

    FreeGrid((instance_xtra->z_grid));
    FreeGrid((instance_xtra->y_grid));
    FreeGrid((instance_xtra->x_grid));
    FreeGrid((instance_xtra->grid2d));
    FreeGrid((instance_xtra->grid));

#ifdef HAVE_CLM
    FreeGrid((instance_xtra->metgrid));
    FreeGrid((instance_xtra->gridTs));

    FreeGrid((instance_xtra->snglclm));         //NBE
#endif

    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * SolverRichardsNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule *
SolverRichardsNewPublicXtra(char *name)
{
  PFModule *this_module = ThisPFModule;
  PublicXtra *public_xtra;

  char key[IDB_MAX_KEY_LEN];

  char *switch_name;
  int switch_value;
  NameArray switch_na;
  NameArray nonlin_switch_na;
  NameArray lsm_switch_na;

#ifdef HAVE_CLM
  NameArray beta_switch_na;
  NameArray vegtype_switch_na;
  NameArray metforce_switch_na;
  NameArray irrtype_switch_na;
  NameArray irrcycle_switch_na;
  NameArray irrthresholdtype_switch_na;
#endif

  switch_na = NA_NewNameArray("False True");

  public_xtra = ctalloc(PublicXtra, 1);

  (public_xtra->permeability_face) = PFModuleNewModule(PermeabilityFace, ());
  (public_xtra->advect_concen) = PFModuleNewModule(Godunov, ());
  (public_xtra->set_problem_data) = PFModuleNewModule(SetProblemData, ());
  (public_xtra->problem) = NewProblem(RichardsSolve);

  nonlin_switch_na = NA_NewNameArray("KINSol");
  sprintf(key, "%s.NonlinearSolver", name);
  switch_name = GetStringDefault(key, "KINSol");
  switch_value = NA_NameToIndexExitOnError(nonlin_switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      (public_xtra->nonlin_solver) =
        PFModuleNewModule(KinsolNonlinSolver, ());
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(nonlin_switch_na);

  lsm_switch_na = NA_NewNameArray("none CLM");
  sprintf(key, "%s.LSM", name);
  switch_name = GetStringDefault(key, "none");
  switch_value = NA_NameToIndexExitOnError(lsm_switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      public_xtra->lsm = 0;
      break;
    }

    case 1:
    {
#ifdef HAVE_CLM
      public_xtra->lsm = 1;
#else
      InputError
        ("Error: <%s> used for key <%s> but this version of Parflow is compiled without CLM\n",
        switch_name, key);
#endif
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(lsm_switch_na);

  /* IMF: Following are only used /w CLM */
#ifdef HAVE_CLM
  sprintf(key, "%s.CLM.CLMDumpInterval", name);
  public_xtra->clm_dump_interval = GetIntDefault(key, 1);

  sprintf(key, "%s.CLM.Print1dOut", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->clm_1d_out = switch_value;

  /*BH: added an option for choosing to force vegetation (LAI,SAI,displa, z0) */
  sprintf(key, "%s.CLM.ForceVegetation", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->clm_forc_veg = switch_value;
  /*BH: end added an option for choosing to force vegetation (LAI,SAI,displa, z0) */


  sprintf(key, "%s.CLM.BinaryOutDir", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->clm_bin_out_dir = switch_value;

  sprintf(key, "%s.CLM.CLMFileDir", name);
  public_xtra->clm_file_dir = GetStringDefault(key, "");

  // NBE: Keys for the single file CLM output
  sprintf(key, "%s.CLM.SingleFile", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->single_clm_file = switch_value;

  // NBE: Different clm_nz must be hard wired, working on a way to dynamically allocate instead
  // unfortunately, the number is still hard wired in clm_varpar.f90 as of 4-12-2014

  /* IMF added key for number of layers in CLM (i.e., layers in root zone) */
  sprintf(key, "%s.CLM.RootZoneNZ", name);
  public_xtra->clm_nz = GetIntDefault(key, 10);

  /* Should match what is set in CLM for max */
  if (public_xtra->clm_nz > PF_CLM_MAX_ROOT_NZ)
  {
    char tmp_str[100];
    sprintf(tmp_str, "%d", public_xtra->clm_nz);
    InputError("Error: Invalid value <%s> for key <%s>, must be <= 20\n", tmp_str, key);
  }

  /* NBE added key to specify layer for t_soisno in clm_dynvegpar */
  sprintf(key, "%s.CLM.SoiLayer", name);
  public_xtra->clm_SoiLayer = GetIntDefault(key, 7);

  //------

  /* NBE added key to reuse a set of CLM input files for an integer
   * number of time steps */
  sprintf(key, "%s.CLM.ReuseCount", name);
  public_xtra->clm_reuse_count = GetIntDefault(key, 1);
  if (public_xtra->clm_reuse_count < 1)
  {
    public_xtra->clm_reuse_count = 1;
  }

  /* NBE - Allows disabling of the CLM output logs generated for each processor
   *  Checking of the values is manual right not in case other options are added
   */
  sprintf(key, "%s.CLM.WriteLogs", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->clm_write_logs = switch_value;

  /* NBE - Only write ONE restart file and overwrite it each time instead of writing
   * a new RST at every step/day */
  sprintf(key, "%s.CLM.WriteLastRST", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->clm_last_rst = switch_value;

  /* NBE - Option to write daily or hourly outputs from CLM */
  sprintf(key, "%s.CLM.DailyRST", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->clm_daily_rst = switch_value;


  // -------------------

  /* RMM added beta input function for clm */
  beta_switch_na = NA_NewNameArray("none Linear Cosine");
  sprintf(key, "%s.CLM.EvapBeta", name);
  switch_name = GetStringDefault(key, "Linear");
  switch_value = NA_NameToIndexExitOnError(beta_switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      public_xtra->clm_beta_function = 0;
      break;
    }

    case 1:
    {
      public_xtra->clm_beta_function = 1;
      break;
    }

    case 2:
    {
      public_xtra->clm_beta_function = 2;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(beta_switch_na);

  sprintf(key, "%s.CLM.ResSat", name);
  public_xtra->clm_res_sat = GetDoubleDefault(key, 0.1);

  /* RMM added veg sm stress input function for clm */
  vegtype_switch_na = NA_NewNameArray("none Pressure Saturation");
  sprintf(key, "%s.CLM.VegWaterStress", name);
  switch_name = GetStringDefault(key, "Saturation");
  switch_value = NA_NameToIndexExitOnError(vegtype_switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      public_xtra->clm_veg_function = 0;
      break;
    }

    case 1:
    {
      public_xtra->clm_veg_function = 1;
      break;
    }

    case 2:
    {
      public_xtra->clm_veg_function = 2;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(vegtype_switch_na);

  sprintf(key, "%s.CLM.WiltingPoint", name);
  public_xtra->clm_veg_wilting = GetDoubleDefault(key, 0.1);

  sprintf(key, "%s.CLM.FieldCapacity", name);
  public_xtra->clm_veg_fieldc = GetDoubleDefault(key, 1.0);

  /* IMF Write CLM as Silo (default=False) */
  sprintf(key, "%s.WriteSiloCLM", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_CLM = switch_value;

  /* IMF Write CLM as PFB (default=False) */
  sprintf(key, "%s.PrintCLM", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_CLM = switch_value;

  /* IMF Write CLM Binary (default=True) */
  sprintf(key, "%s.WriteCLMBinary", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_CLM_binary = switch_value;
  
/* IMF Account for slope in CLM energy budget (default=False) */
  sprintf(key, "%s.CLM.UseSlopeAspect", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->slope_accounting_CLM = switch_value;

  /* IMF Key for CLM met file path */
  sprintf(key, "%s.CLM.MetFilePath", name);
  public_xtra->clm_metpath = GetStringDefault(key, ".");

  /* IMF Key for met vars in subdirectories
   * If True  -- each variable in it's own subdirectory of MetFilePath (e.g., /Temp, /APCP, etc.)
   * If False -- all files in MetFilePath
   */
  sprintf(key, "%s.CLM.MetFileSubdir", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->clm_metsub = switch_value;

  /* IMF Key for CLM met file name...
   * for 1D forcing, is complete file name
   * for 2D/3D forcing, is base file name (w/o timestep extension) */
  /* KKu NetCDF based forcing file name would be read here */
  sprintf(key, "%s.CLM.MetFileName", name);
  public_xtra->clm_metfile = GetStringDefault(key, "narr_1hr.sc3.txt");

  /* IMF Key for CLM istep (default=1) */
  sprintf(key, "%s.CLM.IstepStart", name);
  public_xtra->clm_istep_start = GetIntDefault(key, 1);

  /* IMF Key for CLM fstep (default=1) */
  sprintf(key, "%s.CLM.FstepStart", name);
  public_xtra->clm_fstep_start = GetIntDefault(key, 1);

  /* IMF Switch for 1D (uniform) vs. 2D (distributed) met forcings */
  /* IMF Added 3D option (distributed w/ time axis -- nx*ny*nz; nz=nt) */
  /* KKu Added NetCDF meteorological forcing */
  metforce_switch_na = NA_NewNameArray("none 1D 2D 3D NC");
  sprintf(key, "%s.CLM.MetForcing", name);
  switch_name = GetStringDefault(key, "none");
  switch_value = NA_NameToIndexExitOnError(metforce_switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      public_xtra->clm_metforce = 0;
      break;
    }

    case 1:
    {
      public_xtra->clm_metforce = 1;
      break;
    }

    case 2:
    {
      public_xtra->clm_metforce = 2;
      break;
    }

    case 3:
    {
      public_xtra->clm_metforce = 3;
      break;
    }

    case 4:
    {
      public_xtra->clm_metforce = 4;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(metforce_switch_na);

  /* IMF added key for nt of 3D met files */
  sprintf(key, "%s.CLM.MetFileNT", name);
  public_xtra->clm_metnt = GetIntDefault(key, 1);

  /* IMF added irrigation type, rate, value keys for irrigating in CLM */
  /* IrrigationType -- none, Drip, Spray, Instant (default == none) */
  irrtype_switch_na = NA_NewNameArray("none Spray Drip Instant");
  sprintf(key, "%s.CLM.IrrigationType", name);
  switch_name = GetStringDefault(key, "none");
  switch_value = NA_NameToIndexExitOnError(irrtype_switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0:                     // none
    {
      public_xtra->clm_irr_type = 0;
      break;
    }

    case 1:                     // Spray
    {
      public_xtra->clm_irr_type = 1;
      break;
    }

    case 2:                     // Drip
    {
      public_xtra->clm_irr_type = 2;
      break;
    }

    case 3:                     // Instant
    {
      public_xtra->clm_irr_type = 3;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(irrtype_switch_na);

  /* KKu: Write CLM in NetCDF file */
  /* This key is added here as depenedent on irrigation type
   * an extra variable is written out*/
  sprintf(key, "NetCDF.WriteCLM");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  if (switch_value == 1)
  {
    /* KKu: Number of CLM variables + time in NetCDF file */
    if (public_xtra->clm_irr_type > 0)
    {
      public_xtra->numCLMVarTimeVariant = 15;
    }
    else
    {
      public_xtra->numCLMVarTimeVariant = 14;
    }
  }
  public_xtra->write_netcdf_clm = switch_value;

  /* IrrigationCycle -- Constant, Deficit (default == Deficit) */
  /* (Constant = irrigate based on specified time cycle [IrrigationStartTime,IrrigationEndTime];
   * Deficit  = irrigate based on soil moisture criteria [IrrigationDeficit]) */
  irrcycle_switch_na = NA_NewNameArray("Constant Deficit");
  sprintf(key, "%s.CLM.IrrigationCycle", name);
  switch_name = GetStringDefault(key, "Constant");
  switch_value = NA_NameToIndexExitOnError(irrcycle_switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      public_xtra->clm_irr_cycle = 0;
      break;
    }

    case 1:
    {
      public_xtra->clm_irr_cycle = 1;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(irrcycle_switch_na);

  /* IrrigationValue -- Application rate for Drip or Spray irrigation */
  sprintf(key, "%s.CLM.IrrigationRate", name);
  public_xtra->clm_irr_rate = GetDoubleDefault(key, 0.0);

  /* IrrigationStartTime -- Start time of daily irrigation if IrrigationCycle == Constant */
  /* IrrigationStopTime  -- Stop time of daily irrigation if IrrigationCycle == Constant  */
  /* Default == start @ 12:00gmt (7am in central US), end @ 20:00gmt (3pm in central US)  */
  /* NOTE: Times in GMT */
  sprintf(key, "%s.CLM.IrrigationStartTime", name);
  public_xtra->clm_irr_start = GetDoubleDefault(key, 12.0);
  sprintf(key, "%s.CLM.IrrigationStopTime", name);
  public_xtra->clm_irr_stop = GetDoubleDefault(key, 20.0);

  /* IrrigationThreshold -- Soil moisture threshold for irrigation if IrrigationCycle == Deficit */
  /* CLM applies irrigation whenever soil moisture < threshold */
  sprintf(key, "%s.CLM.IrrigationThreshold", name);
  public_xtra->clm_irr_threshold = GetDoubleDefault(key, 0.5);

  /* IrrigationThresholdType -- Soil moisture threshold for irrigation if IrrigationCycle == Deficit */
  /* Specifies where saturation comparison is made -- top layer, bottom layer, average over column */
  irrthresholdtype_switch_na = NA_NewNameArray("Top Bottom Column");
  sprintf(key, "%s.CLM.IrrigationThresholdType", name);
  switch_name = GetStringDefault(key, "Column");
  switch_value = NA_NameToIndexExitOnError(irrthresholdtype_switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      public_xtra->clm_irr_thresholdtype = 0;
      break;
    }

    case 1:
    {
      public_xtra->clm_irr_thresholdtype = 1;
      break;
    }

    case 2:
    {
      public_xtra->clm_irr_thresholdtype = 2;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(irrthresholdtype_switch_na);
#endif

  //CPS
  /* @RMM added switch for terrain-following grid */
  /* RMM set terrain grid (default=False) */
  sprintf(key, "%s.TerrainFollowingGrid", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->terrain_following_grid = switch_value;
  // CPS

  sprintf(key, "%s.MaxIter", name);
  public_xtra->max_iterations = GetIntDefault(key, 1000000);

  sprintf(key, "%s.MaxConvergenceFailures", name);
  public_xtra->max_convergence_failures = GetIntDefault(key, 3);

  if (public_xtra->max_convergence_failures > 9)
  {
    amps_Printf("Warning: Input variable <%s> \n", key);
    amps_Printf
      ("         is set to a large value that may cause problems\n");
    amps_Printf
      ("         with how time cycles calculations are evaluated.  Values\n");
    amps_Printf
      ("         specified via a time cycle may be on/off at the slightly\n");
    amps_Printf
      ("         wrong times times due to how Parflow discretizes time.\n");
  }

  sprintf(key, "%s.AdvectOrder", name);
  public_xtra->advect_order = GetIntDefault(key, 2);

  sprintf(key, "%s.CFL", name);
  public_xtra->CFL = GetDoubleDefault(key, 0.7);

  sprintf(key, "%s.DropTol", name);
  public_xtra->drop_tol = GetDoubleDefault(key, 1E-8);

  sprintf(key, "%s.PrintSubsurfData", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_subsurf_data = switch_value;

  sprintf(key, "%s.PrintSlopes", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_slopes = switch_value;

  sprintf(key, "%s.PrintMannings", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_mannings = switch_value;

  sprintf(key, "%s.PrintSpecificStorage", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_specific_storage = switch_value;

  sprintf(key, "%s.PrintTop", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_top = switch_value;

  sprintf(key, "%s.PrintPressure", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_press = switch_value;

  sprintf(key, "%s.PrintVelocities", name);
  switch_name = GetStringDefault(key, "False");
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

  sprintf(key, "%s.PrintDZMultiplier", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_dzmult = switch_value;

  sprintf(key, "%s.PrintMask", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_mask = switch_value;

  sprintf(key, "%s.PrintEvapTrans", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_evaptrans = switch_value;

  sprintf(key, "%s.PrintEvapTransSum", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_evaptrans_sum = switch_value;

  sprintf(key, "%s.PrintOverlandSum", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_overland_sum = switch_value;

  sprintf(key, "%s.PrintOverlandBCFlux", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_overland_bc_flux = switch_value;

  sprintf(key, "%s.PrintWells", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_wells = switch_value;

  // SGS TODO
  // Need to add this to the user manual, this is new for LSM stuff that was added.
  sprintf(key, "%s.PrintLSMSink", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->print_lsm_sink = switch_value;

#ifndef HAVE_CLM
  if (public_xtra->print_lsm_sink)
  {
    InputError("Error: setting %s to %s but do not have CLM\n",
               switch_name, key);
  }
#endif

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

  sprintf(key, "%s.WriteSiloEvapTrans", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_evaptrans = switch_value;

  sprintf(key, "%s.WriteSiloEvapTransSum", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_evaptrans_sum = switch_value;

  sprintf(key, "%s.WriteSiloOverlandSum", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_overland_sum = switch_value;

  sprintf(key, "%s.WriteSiloOverlandBCFlux", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_overland_bc_flux = switch_value;

  sprintf(key, "%s.WriteSiloDZMultiplier", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_dzmult = switch_value;
  /*
   * ---------------------------
   * NetCDF Tcl flags
   * --------------------------
   */

  /* KKu: Here we handle only TCL Write flags.
   * Rest of the tuning flags(romio hints, chunking,
   * node level IO, number of steps in NetCDF file are
   * handled in NetCDF interface */
  public_xtra->numVarTimeVariant = 0;   /*Initializing to 0 and incremented
                                         * later depending on which and how many variables
                                         * are written */
  public_xtra->numVarIni = 0;   /*Initializing to 0 and incremented
                                 * later depending on which and how many static variables
                                 * are written */
  sprintf(key, "NetCDF.WritePressure");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  if (switch_value == 1)
  {
    public_xtra->numVarTimeVariant++;
    public_xtra->numVarIni++;
  }
  public_xtra->write_netcdf_press = switch_value;


  sprintf(key, "NetCDF.WriteSaturation");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  if (switch_value == 1)
  {
    public_xtra->numVarTimeVariant++;
    public_xtra->numVarIni++;
  }
  public_xtra->write_netcdf_satur = switch_value;

  sprintf(key, "NetCDF.WriteEvapTrans");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  if (switch_value == 1)
  {
    public_xtra->numVarTimeVariant++;
  }
  public_xtra->write_netcdf_evaptrans = switch_value;

  sprintf(key, "NetCDF.WriteEvapTransSum");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  if (switch_value == 1)
  {
    public_xtra->numVarTimeVariant++;
  }
  public_xtra->write_netcdf_evaptrans_sum = switch_value;

  sprintf(key, "NetCDF.WriteOverlandSum");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  if (switch_value == 1)
  {
    public_xtra->numVarTimeVariant++;
  }
  public_xtra->write_netcdf_overland_sum = switch_value;

  sprintf(key, "NetCDF.WriteOverlandBCFlux");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  if (switch_value == 1)
  {
    public_xtra->numVarTimeVariant++;
  }
  public_xtra->write_netcdf_overland_bc_flux = switch_value;

  sprintf(key, "NetCDF.WriteMannings");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  if (switch_value == 1)
  {
    public_xtra->numVarIni++;
  }
  public_xtra->write_netcdf_mannings = switch_value;

  sprintf(key, "NetCDF.WriteSubsurface");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  if (switch_value == 1)
  {
    /*Increamenting by 5 for x, y, z permiability, porosity and specific storage */
    public_xtra->numVarIni = public_xtra->numVarIni + 5;
  }
  public_xtra->write_netcdf_subsurface = switch_value;

  sprintf(key, "NetCDF.WriteSlopes");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  if (switch_value == 1)
  {
    /*Increamenting by 2 for x, y slopes */
    public_xtra->numVarIni = public_xtra->numVarIni + 2;
  }
  public_xtra->write_netcdf_slopes = switch_value;

  sprintf(key, "NetCDF.WriteDZMultiplier");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  if (switch_value == 1)
  {
    public_xtra->numVarIni++;
  }
  public_xtra->write_netcdf_dzmult = switch_value;

  sprintf(key, "NetCDF.WriteMask");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  if (switch_value == 1)
  {
    public_xtra->numVarIni++;
  }
  public_xtra->write_netcdf_mask = switch_value;

  /* For future other vaiables, handle the TCL flags here
   * and modify the if condition below for time variable
   */

  sprintf(key, "NetCDF.EvapTransFileTransient");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->nc_evap_trans_file_transient = switch_value;

  sprintf(key, "NetCDF.EvapTrans.FileName");
  public_xtra->nc_evap_trans_filename = GetStringDefault(key, "");

  if (public_xtra->write_netcdf_press || public_xtra->write_netcdf_satur
      || public_xtra->write_netcdf_evaptrans
      || public_xtra->write_netcdf_evaptrans_sum
      || public_xtra->write_netcdf_overland_sum
      || public_xtra->write_netcdf_overland_bc_flux)

  {
    /* KKu: Incrementing one for time variable in NC file only if one of
     * the time variant variable is requested for output. This if statement
     * will grow as number of vaiant variable will be added. Could be handled
     * in a different way?
     * This variable is added extra in NetCDF file for ease of post processing
     * with tools such as CDO, NCL, python netcdf etc. */
    public_xtra->numVarTimeVariant++;
  }

  if (public_xtra->write_netcdf_press || public_xtra->write_netcdf_satur
      || public_xtra->write_netcdf_mask
      || public_xtra->write_netcdf_subsurface
      || public_xtra->write_netcdf_slopes || public_xtra->write_netcdf_dzmult)
  {
    /* KKu: Incrementing one for time variable for initial  NC file. */
    public_xtra->numVarIni++;
  }


  /*
   * ---------------------------
   * End of NetCDF Tcl flags
   * --------------------------
   */

#ifndef HAVE_CLM
  if (public_xtra->write_silo_overland_bc_flux)
  {
    InputError("Error: setting %s to %s but do not have CLM\n",
               switch_name, key);
  }
#endif

  sprintf(key, "%s.WriteSiloConcentration", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_concen = switch_value;

  sprintf(key, "%s.WriteSiloMask", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_mask = switch_value;


  sprintf(key, "%s.WriteSiloSlopes", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_slopes = switch_value;

  sprintf(key, "%s.WriteSiloMannings", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_mannings = switch_value;

  sprintf(key, "%s.WriteSiloSpecificStorage", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_specific_storage = switch_value;

  sprintf(key, "%s.WriteSiloTop", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silo_top = switch_value;


  /* Initialize silo if necessary */
  if (public_xtra->write_silo_subsurf_data ||
      public_xtra->write_silo_press ||
      public_xtra->write_silo_velocities ||
      public_xtra->write_silo_satur ||
      public_xtra->write_silo_concen ||
      public_xtra->write_silo_specific_storage ||
      public_xtra->write_silo_slopes ||
      public_xtra->write_silo_evaptrans ||
      public_xtra->write_silo_evaptrans_sum ||
      public_xtra->write_silo_mannings ||
      public_xtra->write_silo_mask ||
      public_xtra->write_silo_top ||
      public_xtra->write_silo_overland_sum ||
      public_xtra->write_silo_overland_bc_flux ||
      public_xtra->write_silo_dzmult || public_xtra->write_silo_CLM)
  {
    WriteSiloInit(GlobalsOutFileName);
  }

  /* RMM -- added control block for silo pmpio
   * writing.  Previous silo is backward compat/included and true/false
   * switches work the same as SILO and PFB.  We can change defaults eventually
   * to write silopmpio and to write a single file in the future */
  /* Silo PMPIO file writing control */
  sprintf(key, "%s.WriteSiloPMPIOSubsurfData", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_subsurf_data = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOPressure", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_press = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOVelocities", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_velocities = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOSaturation", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_satur = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOEvapTrans", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_evaptrans = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOEvapTransSum", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_evaptrans_sum = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOOverlandSum", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_overland_sum = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOOverlandBCFlux", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_overland_bc_flux = switch_value;

  sprintf(key, "%s.WriteSiloPMPIODZMultiplier", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_dzmult = switch_value;

#ifndef HAVE_CLM
  if (public_xtra->write_silopmpio_overland_bc_flux)
  {
    InputError("Error: setting %s to %s but do not have CLM\n",
               switch_name, key);
  }
#endif

  sprintf(key, "%s.WriteSiloPMPIOConcentration", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_concen = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOMask", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_mask = switch_value;


  sprintf(key, "%s.WriteSiloPMPIOSlopes", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_slopes = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOMannings", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_mannings = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOSpecificStorage", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_specific_storage = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOTop", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->write_silopmpio_top = switch_value;

  //RMM spinup key
  sprintf(key, "%s.Spinup", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->spinup = switch_value;

 //RMM surface pressure keys
 //Solver.ResetSurfacePressure “True”
  sprintf(key, "%s.ResetSurfacePressure", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->reset_surface_pressure = switch_value;

  sprintf(key, "%s.ResetSurfacePressure.ThresholdPressure", name);
  public_xtra->threshold_pressure = GetDoubleDefault(key, 0.0);

  sprintf(key, "%s.ResetSurfacePressure.ResetPressure", name);
  public_xtra->reset_pressure = GetDoubleDefault(key, 0.0);

//RMM surface predictor keys
  sprintf(key, "%s.SurfacePredictor", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->surface_predictor = switch_value;

  sprintf(key, "%s.SurfacePredictor.PressureValue", name);
  public_xtra->surface_predictor_pressure = GetDoubleDefault(key, 0.00001);

  sprintf(key, "%s.SurfacePredictor.PrintValues", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->surface_predictor_print = switch_value;



  /* @RMM read evap trans as SS file before advance richards
   * for P-E spinup type runs                                  */

  sprintf(key, "%s.EvapTransFile", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->evap_trans_file = switch_value;


  sprintf(key, "%s.EvapTransFileTransient", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->evap_trans_file_transient = switch_value;

  /* Nick's addition */
  sprintf(key, "%s.EvapTrans.FileLooping", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  public_xtra->evap_trans_file_looping = switch_value;


  /* and read file name for evap trans file  */
  sprintf(key, "%s.EvapTrans.FileName", name);
  public_xtra->evap_trans_filename = GetStringDefault(key, "");


  /* Initialize silo if necessary */
  if (public_xtra->write_silopmpio_subsurf_data ||
      public_xtra->write_silopmpio_press ||
      public_xtra->write_silopmpio_velocities ||
      public_xtra->write_silopmpio_satur ||
      public_xtra->write_silopmpio_concen ||
      public_xtra->write_silopmpio_specific_storage ||
      public_xtra->write_silopmpio_slopes ||
      public_xtra->write_silopmpio_evaptrans ||
      public_xtra->write_silopmpio_evaptrans_sum ||
      public_xtra->write_silopmpio_mannings ||
      public_xtra->write_silopmpio_mask ||
      public_xtra->write_silopmpio_top ||
      public_xtra->write_silopmpio_overland_sum ||
      public_xtra->write_silopmpio_overland_bc_flux ||
      public_xtra->write_silopmpio_dzmult || public_xtra->write_silopmpio_CLM)
  {
    WriteSiloPMPIOInit(GlobalsOutFileName);
  }

  NA_FreeNameArray(switch_na);
  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * SolverRichardsFreePublicXtra
 *--------------------------------------------------------------------------*/

void
SolverRichardsFreePublicXtra()
{
  PFModule *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    FreeProblem(public_xtra->problem, RichardsSolve);

    PFModuleFreeModule(public_xtra->set_problem_data);
    PFModuleFreeModule(public_xtra->advect_concen);
    PFModuleFreeModule(public_xtra->permeability_face);
    PFModuleFreeModule(public_xtra->nonlin_solver);
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * SolverRichardsSizeOfTempData
 *--------------------------------------------------------------------------*/

int
SolverRichardsSizeOfTempData()
{
  /* SGS temp data */

  return 0;
}

/*--------------------------------------------------------------------------
 * SolverRichards
 *--------------------------------------------------------------------------*/
void
SolverRichards()
{
  PFModule *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Problem *problem = (public_xtra->problem);
  double start_time = ProblemStartTime(problem);
  double stop_time = ProblemStopTime(problem);
  Vector *pressure_out;
  Vector *porosity_out;
  Vector *saturation_out;

  SetupRichards(this_module);

  AdvanceRichards(this_module,
                  start_time,
                  stop_time,
                  NULL,
                  NULL, &pressure_out, &porosity_out, &saturation_out);

  /*
   * Record amount of memory in use.
   */
  recordMemoryInfo();

  TeardownRichards(this_module);
}

/*
 * Getter/Setter methods
 */

ProblemData *
GetProblemDataRichards(PFModule * this_module)
{
  InstanceXtra *instance_xtra =
    (InstanceXtra*)PFModuleInstanceXtra(this_module);

  return(instance_xtra->problem_data);
}

Problem *
GetProblemRichards(PFModule * this_module)
{
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  return(public_xtra->problem);
}

PFModule *
GetICPhasePressureRichards(PFModule * this_module)
{
  InstanceXtra *instance_xtra =
    (InstanceXtra*)PFModuleInstanceXtra(this_module);

  return(instance_xtra->ic_phase_pressure);
}
