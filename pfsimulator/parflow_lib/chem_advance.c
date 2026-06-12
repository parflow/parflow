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
* Advances the geochemical problem for a time step of size dt
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "parflow.h"
#include "pf_alquimia.h"

#ifdef HAVE_ALQUIMIA

#ifdef HAVE_ALQUIMIA
#include "alquimia/alquimia_interface.h"
#include "alquimia/alquimia_util.h"
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int time_index;
  int write_restart_interval;
  int restart_file_flag;
  int dump_index;
  double time_conversion_factor;
} PublicXtra;

typedef struct {
  Problem       *problem;
  Grid          *grid;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * AdvanceChemistry
 *--------------------------------------------------------------------------*/
#ifdef HAVE_ALQUIMIA
void AdvanceChemistry(ProblemData *problem_data, AlquimiaDataPF *alquimia_data, Vector **concentrations, Vector *saturation, double dt, double t, int *any_file_dumped, int dump_files, int file_number, char *file_prefix)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  Grid          *grid = (instance_xtra->grid);
  GrGeomSolid   *gr_domain;
  Subgrid       *subgrid;
  Subvector     *por_sub, *sat_sub;
  double field_sum;
  double dt_seconds;
  char file_postfix[2048];
  double water_density = 998.0;    // density of water in kg/m**3
  double aqueous_pressure = 101325.0; // pressure in Pa.
  int is = 0;
  int i, j, k;
  int ix, iy, iz;
  int nx, ny, nz;
  int dx, dy, dz;
  int r;
  int chem_index, por_index, sat_index;
  double *por, *sat;

  BeginTiming(public_xtra->time_index);

  gr_domain = ProblemDataGrDomain(problem_data);
  SubgridArray  *subgrids = GridSubgrids(grid);
  subgrid = SubgridArraySubgrid(subgrids, is);

  dt_seconds = dt * public_xtra->time_conversion_factor;

  // copy the transported primary mobile concentrations to alquimia
  AdvectedPrimaryToChem(alquimia_data->chem_state, &alquimia_data->chem_sizes, concentrations, problem_data);

  // solve chemistry cell-by-cell
  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    double vol = dx * dy * dz;

    // RDF: assume resolution is the same in all 3 directions
    r = SubgridRX(subgrid);

    por_sub = VectorSubvector(ProblemDataPorosity(problem_data), is);
    por = SubvectorData(por_sub);

    sat_sub = VectorSubvector(saturation, is);
    sat = SubvectorData(sat_sub);

    GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
    {
      por_index = SubvectorEltIndex(por_sub, i, j, k);
      sat_index = SubvectorEltIndex(sat_sub, i, j, k);

      chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;

      alquimia_data->chem_properties[chem_index].volume = vol;
      alquimia_data->chem_properties[chem_index].saturation = sat[sat_index];

      // Set the thermodynamic state.
      alquimia_data->chem_state[chem_index].water_density = water_density;
      alquimia_data->chem_state[chem_index].temperature = 25.0;
      alquimia_data->chem_state[chem_index].porosity = por[por_index];
      alquimia_data->chem_state[chem_index].aqueous_pressure = aqueous_pressure;

      //copy pre-solution state and aux data to temp containers in case non-convergence is an issue
      CopyAlquimiaState(&alquimia_data->chem_state[chem_index], &alquimia_data->chem_state_temp);
      CopyAlquimiaAuxiliaryData(&alquimia_data->chem_aux_data[chem_index], &alquimia_data->chem_aux_data_temp);
      CopyAlquimiaProperties(&alquimia_data->chem_properties[chem_index], &alquimia_data->chem_properties_temp);


      // Solve the geochemical system
      alquimia_data->chem.ReactionStepOperatorSplit(&alquimia_data->chem_engine,
                                                    dt_seconds, &alquimia_data->chem_properties[chem_index],
                                                    &alquimia_data->chem_state[chem_index],
                                                    &alquimia_data->chem_aux_data[chem_index],
                                                    &alquimia_data->chem_status);
      if (alquimia_data->chem_status.error != 0)
      {
        amps_Printf("ReactionStepOperatorSplit() error: %s\n",
                    alquimia_data->chem_status.message);
        PARFLOW_ERROR("Geochemical engine error, exiting simulation.\n");
      }


      if (!(alquimia_data->chem_status.converged))
      {
        amps_Printf("Geochemical engine failed to converge in cell %d %d %d, cutting timestep.\n", i, j, k);

        CopyAlquimiaState(&alquimia_data->chem_state_temp, &alquimia_data->chem_state[chem_index]);
        CopyAlquimiaAuxiliaryData(&alquimia_data->chem_aux_data_temp, &alquimia_data->chem_aux_data[chem_index]);
        CopyAlquimiaProperties(&alquimia_data->chem_properties_temp, &alquimia_data->chem_properties[chem_index]);

        CutTimeStepandSolveSingleCell(alquimia_data->chem, &alquimia_data->chem_state[chem_index], &alquimia_data->chem_properties[chem_index],
                                      alquimia_data->chem_engine, &alquimia_data->chem_aux_data[chem_index], &alquimia_data->chem_status, dt_seconds);
      }


      alquimia_data->chem.GetAuxiliaryOutput(&alquimia_data->chem_engine,
                                             &alquimia_data->chem_properties[chem_index],
                                             &alquimia_data->chem_state[chem_index],
                                             &alquimia_data->chem_aux_data[chem_index],
                                             &alquimia_data->chem_aux_output[chem_index],
                                             &alquimia_data->chem_status);
      if (alquimia_data->chem_status.error != 0)
      {
        amps_Printf("GetAuxiliaryOutput() auxiliary output fetch failed: %s\n",
                    alquimia_data->chem_status.message);
        PARFLOW_ERROR("Geochemical engine error, exiting simulation.\n");
      }
    });
  }

  // copy solved primary concentrations back to PF
  ReactedPrimaryToPF(alquimia_data->chem_state, &alquimia_data->chem_sizes, concentrations, problem_data);

  // print PFB or silo files if user requested
  if (dump_files)
  {
    // transfer the new chemistry state and aux_output to PF Vectors
    ChemDataToPFVectors(alquimia_data, grid, problem_data);

    // print the concen volume
    Vector* porsat_inv = NewVectorType(grid, 1, 1, vector_cell_centered);
    PFVInvProd(saturation, ProblemDataPorosity(problem_data), porsat_inv);
    for (int concen = 0; concen < alquimia_data->chem_sizes.num_primary; concen++)
    {
      field_sum = ComputeTotalConcen(ProblemDataGrDomain(problem_data), grid, concentrations[concen], porsat_inv);
      if (!amps_Rank(amps_CommWorld))
      {
        amps_Printf("Concentration volume for component %s at time %f = %f\n", alquimia_data->chem_metadata.primary_names.data[concen], t, field_sum);
      }
    }
    FreeVector(porsat_inv);

    // print any silo or pfb files if user requested
    PrintChemistryData(alquimia_data->print_flags, &alquimia_data->chem_sizes,
                       &alquimia_data->chem_metadata, t, file_number, file_prefix,
                       any_file_dumped, concentrations,
                       alquimia_data->total_immobilePF,
                       alquimia_data->mineral_specific_surfacePF,
                       alquimia_data->mineral_volume_fractionsPF,
                       alquimia_data->surface_site_densityPF,
                       alquimia_data->cation_exchange_capacityPF,
                       alquimia_data->pH, alquimia_data->aqueous_kinetic_ratePF,
                       alquimia_data->mineral_saturation_indexPF,
                       alquimia_data->mineral_reaction_ratePF,
                       alquimia_data->primary_free_ion_concentrationPF,
                       alquimia_data->primary_activity_coeffPF,
                       alquimia_data->secondary_free_ion_concentrationPF,
                       alquimia_data->secondary_activity_coeffPF);

    public_xtra->dump_index++;

    if ((public_xtra->restart_file_flag) && (public_xtra->dump_index % public_xtra->write_restart_interval == 0)) // write chemistry checkpoint file
    {
      sprintf(file_postfix, "CHEM_CHKPT.%05d", file_number);
      WriteChemChkpt(grid, &alquimia_data->chem_sizes, alquimia_data->chem_state, alquimia_data->chem_aux_data, alquimia_data->chem_properties, file_prefix, file_postfix);
    }
  }

  EndTiming(public_xtra->time_index);
}


/*--------------------------------------------------------------------------
 * AdvanceChemistryInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *AdvanceChemistryInitInstanceXtra(Problem *problem, Grid *grid)
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `problem'
   *-----------------------------------------------------------------------*/

  if (problem != NULL)
    (instance_xtra->problem) = problem;

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
  {
    /* free old data */
    if ((instance_xtra->grid) != NULL)
    {
    }

    /* set new data */
    (instance_xtra->grid) = grid;
  }


  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * AdvanceChemistryFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  AdvanceChemistryFreeInstanceXtra()
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * AdvanceChemistryNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *AdvanceChemistryNewPublicXtra()
{
  PFModule     *this_module = ThisPFModule;
  PublicXtra   *public_xtra;
  char key[IDB_MAX_KEY_LEN];
  NameArray switch_na;
  char*          switch_name;
  int switch_value;

  switch_na = NA_NewNameArray("s sec S SECONDS Seconds seconds m min M MINUTES Minutes minutes h hr H HOURS Hours hours d D DAYS Days days y yr Y YEARS Years years");


  public_xtra = ctalloc(PublicXtra, 1);

  (public_xtra->time_index) = RegisterTiming("Chemistry Solver");

  sprintf(key, "Chemistry.ParFlowTimeUnits");
  switch_name = GetString(key);
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>. Available options are one of <s sec S SECONDS Seconds seconds m min M MINUTES Minutes minutes h hr H HOURS Hours hours d D DAYS Days days y yr Y YEARS Years years>\n", switch_name, key);
  }

  if (switch_value < 6) //PF in seconds
  {
    public_xtra->time_conversion_factor = 1.0;
  }
  else if (switch_value > 5 && switch_value < 12)   //PF in minutes
  {
    public_xtra->time_conversion_factor = 60.0;
  }
  else if (switch_value > 11 && switch_value < 18)   //PF in hours
  {
    public_xtra->time_conversion_factor = 3600.0;
  }
  else if (switch_value > 17 && switch_value < 23)   //PF in days
  {
    public_xtra->time_conversion_factor = 3600.0 * 24.0;
  }
  else if (switch_value > 22)   //PF in years
  {
    public_xtra->time_conversion_factor = 3600.0 * 24.0 * 365.00;
  }

  sprintf(key, "Chemistry.RestartFileWriteInterval");
  public_xtra->write_restart_interval = GetIntDefault(key, -1);

  if (public_xtra->write_restart_interval < 0)
  {
    public_xtra->restart_file_flag = 0;     // no restart dump interval provided, not writing files
  }
  else
  {
    public_xtra->restart_file_flag = 1;     // write restart ckpt files
  }
  public_xtra->dump_index = 0;


  NA_FreeNameArray(switch_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * AdvanceChemistryFreePublicXtra
 *--------------------------------------------------------------------------*/

void AdvanceChemistryFreePublicXtra()
{
  PFModule     *this_module = ThisPFModule;
  PublicXtra   *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * AdvanceChemistrySizeOfTempData
 *--------------------------------------------------------------------------*/

int AdvanceChemistrySizeOfTempData()
{
  int sz = 0;

  return sz;
}
#endif
#endif /* HAVE_ALQUIMIA */
