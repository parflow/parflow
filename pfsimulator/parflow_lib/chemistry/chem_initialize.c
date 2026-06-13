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
* Module for initializing the geochemical problem. This code reads the PF input
* file chemistry options, starts the alquimia interface, allocates the alquimia
* and PF data storage, and processes and assigns geochemical initial and boundary
* conditions. The data is saved in a large struct, AlquimiaDataPF. Initial
* variables are printed if the user has requested them.
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/


#include "parflow.h"
#include "pf_alquimia.h"

#ifdef HAVE_ALQUIMIA

#ifdef HAVE_ALQUIMIA
#include "alquimia/alquimia_constants.h"
#include "alquimia/alquimia_memory.h"
#include "alquimia/alquimia_util.h"
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int time_index;
  char *engine_name;
  char *chemistry_input_file;
  char *chem_restart_file;
  int chem_restart;
  int num_ic_conds;
  int num_bc_conds;
  int print_primary_mobile;
  int silo_primary_mobile;
  int print_mineral_rate;
  int silo_mineral_rate;
  int print_mineral_volfx;
  int silo_mineral_volfx;
  int print_mineral_surfarea;
  int silo_mineral_surfarea;
  int print_surf_dens;
  int silo_surf_dens;
  int print_CEC;
  int silo_CEC;
  int print_pH;
  int silo_pH;
  int print_aqueous_rate;
  int silo_aqueous_rate;
  int print_mineral_SI;
  int silo_mineral_SI;
  int print_primary_freeion;
  int silo_primary_freeion;
  int print_primary_activity;
  int silo_primary_activity;
  int print_secondary_freeion;
  int silo_secondary_freeion;
  int print_secondary_activity;
  int silo_secondary_activity;
  int print_sorbed;
  int silo_sorbed;
  PFModule *set_chem_data;
  NameArray ic_cond_na;
  NameArray bc_cond_na;
} PublicXtra;

typedef struct {
  Problem  *problem;
  Grid     *grid;
  PFModule *set_chem_data;
  PFModule *bc_concentration;
} InstanceXtra;

/*--------------------------------------------------------------------------
 * InitializeChemistry
 *--------------------------------------------------------------------------*/
#ifdef HAVE_ALQUIMIA
void InitializeChemistry(ProblemData *problem_data, AlquimiaDataPF *alquimia_data, Vector **concentrations, Vector *saturation, int *any_file_dumped, int dump_files, double t, int file_number, char* file_prefix)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  Problem       *problem = (instance_xtra->problem);
  Grid          *grid = (instance_xtra->grid);

  PFModule      *set_chem_data = (instance_xtra->set_chem_data);
  PFModule      *bc_concentration = (instance_xtra->bc_concentration);

  VectorUpdateCommHandle *handle;

  int num_cells;
  bool hands_off = true;
  double field_sum;

  BeginTiming(public_xtra->time_index);

  alquimia_data->print_flags = ctalloc(ChemPrintFlags, 1);

  // gather print flags
  alquimia_data->print_flags->print_primary_mobile = public_xtra->print_primary_mobile;
  alquimia_data->print_flags->silo_primary_mobile = public_xtra->silo_primary_mobile;
  alquimia_data->print_flags->print_mineral_rate = public_xtra->print_mineral_rate;
  alquimia_data->print_flags->silo_mineral_rate = public_xtra->silo_mineral_rate;
  alquimia_data->print_flags->print_mineral_volfx = public_xtra->print_mineral_volfx;
  alquimia_data->print_flags->silo_mineral_volfx = public_xtra->silo_mineral_volfx;
  alquimia_data->print_flags->print_mineral_surfarea = public_xtra->print_mineral_surfarea;
  alquimia_data->print_flags->silo_mineral_surfarea = public_xtra->silo_mineral_surfarea;
  alquimia_data->print_flags->print_surf_dens = public_xtra->print_surf_dens;
  alquimia_data->print_flags->silo_surf_dens = public_xtra->silo_surf_dens;
  alquimia_data->print_flags->print_CEC = public_xtra->print_CEC;
  alquimia_data->print_flags->silo_CEC = public_xtra->silo_CEC;
  alquimia_data->print_flags->print_pH = public_xtra->print_pH;
  alquimia_data->print_flags->silo_pH = public_xtra->silo_pH;
  alquimia_data->print_flags->print_aqueous_rate = public_xtra->print_aqueous_rate;
  alquimia_data->print_flags->silo_aqueous_rate = public_xtra->silo_aqueous_rate;
  alquimia_data->print_flags->print_mineral_SI = public_xtra->print_mineral_SI;
  alquimia_data->print_flags->silo_mineral_SI = public_xtra->silo_mineral_SI;
  alquimia_data->print_flags->print_primary_freeion = public_xtra->print_primary_freeion;
  alquimia_data->print_flags->silo_primary_freeion = public_xtra->silo_primary_freeion;
  alquimia_data->print_flags->print_primary_activity = public_xtra->print_primary_activity;
  alquimia_data->print_flags->silo_primary_activity = public_xtra->silo_primary_activity;
  alquimia_data->print_flags->print_secondary_freeion = public_xtra->print_secondary_freeion;
  alquimia_data->print_flags->silo_secondary_freeion = public_xtra->silo_secondary_freeion;
  alquimia_data->print_flags->print_secondary_activity = public_xtra->print_secondary_activity;
  alquimia_data->print_flags->silo_secondary_activity = public_xtra->silo_secondary_activity;
  alquimia_data->print_flags->print_sorbed = public_xtra->print_sorbed;
  alquimia_data->print_flags->silo_sorbed = public_xtra->silo_sorbed;

  // set chem data
  // this ivokes the geochemcond function and makes available a pfvector of
  // geochemcond indicators
  PFModuleInvokeType(SetChemDataInvoke, set_chem_data, (problem_data));

  // find number of active cells for this subgrid
  num_cells = SubgridNumCells(grid);

  // create alquimia interface
  AllocateAlquimiaEngineStatus(&alquimia_data->chem_status);
  CreateAlquimiaInterface(public_xtra->engine_name, &alquimia_data->chem, &alquimia_data->chem_status);
  if (alquimia_data->chem_status.error != 0)
  {
    alquimia_error("Alquimia interface creation error: %s", alquimia_data->chem_status.message);
    PARFLOW_ERROR("Geochemical engine error, exiting simulation.\n");
  }
  else if (!amps_Rank(amps_CommWorld))
  {
    amps_Printf("Successful creation of Alquimia interface\n");
  }


  // read input file, setup chem problem
  alquimia_data->chem.Setup(public_xtra->chemistry_input_file,
                            hands_off,
                            &alquimia_data->chem_engine,
                            &alquimia_data->chem_sizes,
                            &alquimia_data->chem_engine_functionality,
                            &alquimia_data->chem_status);

  if (alquimia_data->chem_status.error != 0)
  {
    alquimia_error("Alquimia interface setup error: %s", alquimia_data->chem_status.message);
    PARFLOW_ERROR("Geochemical engine error, exiting simulation.\n");
  }
  else if (!amps_Rank(amps_CommWorld))
  {
    amps_Printf("Successful setup() of Alquimia interface\n");
    PrintAlquimiaSizes(&alquimia_data->chem_sizes, stdout);
  }


  // assert that PF num_contams == chem.num_primary
  if (ProblemNumContaminants(problem) != alquimia_data->chem_sizes.num_primary)
  {
    amps_Printf("Input Error: mismatch between PF number of Contaminants.Names: <%d> and Alquimia num_primary: <%d>\n",
                ProblemNumContaminants(problem), alquimia_data->chem_sizes.num_primary);
    PARFLOW_ERROR("Geochemical engine error, exiting simulation.\n");
  }


  // Allocate space for alquimia data
  AllocateAlquimiaProblemMetaData(&alquimia_data->chem_sizes, &alquimia_data->chem_metadata);
  alquimia_data->chem_properties = ctalloc(AlquimiaProperties, num_cells);
  alquimia_data->chem_state = ctalloc(AlquimiaState, num_cells);
  alquimia_data->chem_aux_data = ctalloc(AlquimiaAuxiliaryData, num_cells);
  alquimia_data->chem_aux_output = ctalloc(AlquimiaAuxiliaryOutputData, num_cells);

  // allocate cell by cell data in alquimia data structs
  AllocateChemCells(alquimia_data, grid, problem_data);
  //allocate PF vector data for printing - todo: make allocations more flexible
  AllocatePFChemData(alquimia_data, grid);

  // get metadata - chemistry names etc
  alquimia_data->chem.GetProblemMetaData(&alquimia_data->chem_engine,
                                         &alquimia_data->chem_metadata,
                                         &alquimia_data->chem_status);
  if (alquimia_data->chem_status.error != 0)
  {
    alquimia_error("Alquimia GetProblemMetaData() error: %s", alquimia_data->chem_status.message);
    PARFLOW_ERROR("Geochemical engine error, exiting simulation.\n");
  }
  else if (!amps_Rank(amps_CommWorld))
  {
    amps_Printf("Successful GetProblemMetaData() of Alquimia interface\n");
  }

  if (public_xtra->chem_restart)
  {
    // read restart file, populate chem_state, chem_properties, chem_aux_data
    ReadChemChkpt(grid, &alquimia_data->chem_sizes, alquimia_data->chem_state, alquimia_data->chem_aux_data, alquimia_data->chem_properties, public_xtra->chem_restart_file);
  }
  else
  {
    // process geochem conds on a cell by cell basis, fill alquimia data structs
    ProcessGeochemICs(alquimia_data, grid, problem_data, public_xtra->num_ic_conds, public_xtra->ic_cond_na, saturation);
    if (alquimia_data->chem_status.error != 0)
    {
      alquimia_error("Alquimia ProcessGeochemICs() error: %s", alquimia_data->chem_status.message);
      PARFLOW_ERROR("Geochemical engine error, exiting simulation.\n");
    }
    else if (!amps_Rank(amps_CommWorld))
    {
      amps_Printf("Successful ProcessGeochemICs() of Alquimia interface\n");
    }
  }


  // process assigned boundary conditions
  ProcessGeochemBCs(alquimia_data, public_xtra->num_bc_conds, public_xtra->bc_cond_na);
  if (alquimia_data->chem_status.error != 0)
  {
    alquimia_error("Alquimia ProcessGeochemBCs() error: %s", alquimia_data->chem_status.message);
    PARFLOW_ERROR("Geochemical engine error, exiting simulation.\n");
  }
  else if (!amps_Rank(amps_CommWorld))
  {
    amps_Printf("Successful ProcessGeochemBCs() of Alquimia interface\n");
  }


  // copy alquimia data to PF Vectors for printing
  ReactedPrimaryToPF(alquimia_data->chem_state, &alquimia_data->chem_sizes, concentrations, problem_data);
  ChemDataToPFVectors(alquimia_data, grid, problem_data);

  // scatter to ghost in case boundary routine needs to copy from interior domain
  for (int concen = 0; concen < ProblemNumContaminants(problem); concen++)
  {
    handle = InitVectorUpdate(concentrations[concen], VectorUpdateGodunov);
    FinalizeVectorUpdate(handle);
  }

  // fill concen vector with assigned boundaries
  PFModuleInvokeType(BCConcentrationInvoke, bc_concentration, (problem, grid, concentrations, alquimia_data->chem_bc_state));


  // print initial concen volume
  Vector* porsat_inv = NewVectorType(grid, 1, 1, vector_cell_centered);
  PFVInvProd(saturation, ProblemDataPorosity(problem_data), porsat_inv);
  for (int concen = 0; concen < alquimia_data->chem_sizes.num_primary; concen++)
  {
    field_sum = ComputeTotalConcen(ProblemDataGrDomain(problem_data), grid, concentrations[concen], porsat_inv);
    if (!amps_Rank(amps_CommWorld))
    {
      amps_Printf("Initial concentration volume for component %s = %f\n", alquimia_data->chem_metadata.primary_names.data[concen], field_sum);
    }
  }
  FreeVector(porsat_inv);

  if (dump_files && !(public_xtra->chem_restart))
  {
    PrintChemistryData(alquimia_data->print_flags, &alquimia_data->chem_sizes,
                       &alquimia_data->chem_metadata, t, file_number, file_prefix, any_file_dumped,
                       concentrations, alquimia_data->total_immobilePF,
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
  }

  EndTiming(public_xtra->time_index);
}


/*--------------------------------------------------------------------------
 * InitializeChemistryInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *InitializeChemistryInitInstanceXtra(Problem *problem, Grid *grid)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;


  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    instance_xtra = ctalloc(InstanceXtra, 1);

    (instance_xtra->set_chem_data) =
      PFModuleNewInstanceType(SetChemDataInitInstanceXtraInvoke,
                              (public_xtra->set_chem_data),
                              (problem, grid));


    (instance_xtra->bc_concentration) =
      PFModuleNewInstance(ProblemBCConcentration(problem), ());
  }
  else
  {
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

    PFModuleReNewInstanceType(SetChemDataInitInstanceXtraInvoke,
                              (instance_xtra->set_chem_data),
                              (problem, grid));
    PFModuleReNewInstance((instance_xtra->bc_concentration), ());
  }

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
 * InitializeChemistryFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  InitializeChemistryFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    PFModuleFreeInstance((instance_xtra->set_chem_data));
    PFModuleFreeInstance(instance_xtra->bc_concentration);

    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * InitializeChemistryNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *InitializeChemistryNewPublicXtra()
{
  PFModule *this_module = ThisPFModule;
  PublicXtra *public_xtra;
  char key[IDB_MAX_KEY_LEN];
  char *ic_cond_names, *bc_cond_names;
  NameArray switch_na;
  char *switch_name;
  int switch_value;
  int print_all, write_silo_all;

  switch_na = NA_NewNameArray("False True");

  public_xtra = ctalloc(PublicXtra, 1);
  (public_xtra->set_chem_data) = PFModuleNewModule(SetChemData, ());
  (public_xtra->time_index) = RegisterTiming("Chemistry Initialization");


  sprintf(key, "Chemistry.Engine");
  public_xtra->engine_name = GetStringDefault(key, "");

  if (strcmp(public_xtra->engine_name, kAlquimiaStringCrunchFlow) != 0
      && strcmp(public_xtra->engine_name, kAlquimiaStringPFloTran) != 0)
  {
    InputError("Error: Invalid value <%s> for key <%s>. Options are 'CrunchFlow' or 'PFloTran'\n",
               public_xtra->engine_name, key);
  }

  sprintf(key, "Chemistry.InputFile");
  public_xtra->chemistry_input_file = GetString(key);


  sprintf(key, "Chemistry.RestartFromFile");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: Invalid value <%s> for key <%s>. Options are 'True' or 'False'\n",
               switch_name, key);
  }
  public_xtra->chem_restart = switch_value;

  if (public_xtra->chem_restart)
  {
    sprintf(key, "Chemistry.RestartFileName");
    public_xtra->chem_restart_file = GetString(key);
  }


  bc_cond_names = GetStringDefault("BCConcentration.GeochemCondition.Names", "");
  public_xtra->bc_cond_na = NA_NewNameArray(bc_cond_names);
  public_xtra->num_bc_conds = NA_Sizeof(public_xtra->bc_cond_na);

  ic_cond_names = GetStringDefault("GeochemCondition.Names", "");
  public_xtra->ic_cond_na = NA_NewNameArray(ic_cond_names);
  public_xtra->num_ic_conds = NA_Sizeof(public_xtra->ic_cond_na);



  sprintf(key, "Chemistry.PrintPrimaryMobile");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_primary_mobile = switch_value;

  sprintf(key, "Chemistry.WriteSiloPrimaryMobile");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_primary_mobile = switch_value;



  sprintf(key, "Chemistry.PrintMineralRate");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_mineral_rate = switch_value;

  sprintf(key, "Chemistry.WriteSiloMineralRate");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_mineral_rate = switch_value;



  sprintf(key, "Chemistry.PrintMineralVolfx");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_mineral_volfx = switch_value;

  sprintf(key, "Chemistry.WriteSiloMineralVolfx");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_mineral_volfx = switch_value;



  sprintf(key, "Chemistry.PrintMineralSurfArea");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_mineral_surfarea = switch_value;

  sprintf(key, "Chemistry.WriteSiloMineralSurfArea");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_mineral_surfarea = switch_value;



  sprintf(key, "Chemistry.PrintSurfSiteDens");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_surf_dens = switch_value;

  sprintf(key, "Chemistry.WriteSiloSurfSiteDens");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_surf_dens = switch_value;



  sprintf(key, "Chemistry.PrintCEC");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_CEC = switch_value;

  sprintf(key, "Chemistry.WriteSiloCEC");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_CEC = switch_value;



  sprintf(key, "Chemistry.PrintpH");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_pH = switch_value;

  sprintf(key, "Chemistry.WriteSilopH");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_pH = switch_value;



  sprintf(key, "Chemistry.PrintAqueousRate");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_aqueous_rate = switch_value;

  sprintf(key, "Chemistry.WriteSiloAqueousRate");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_aqueous_rate = switch_value;



  sprintf(key, "Chemistry.PrintMineralSI");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_mineral_SI = switch_value;

  sprintf(key, "Chemistry.WriteSiloMineralSI");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_mineral_SI = switch_value;



  sprintf(key, "Chemistry.PrintPrimaryFreeIon");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_primary_freeion = switch_value;

  sprintf(key, "Chemistry.WriteSiloPrimaryFreeIon");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_primary_freeion = switch_value;



  sprintf(key, "Chemistry.PrintPrimaryActivity");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_primary_activity = switch_value;

  sprintf(key, "Chemistry.WriteSiloPrimaryActivity");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_primary_activity = switch_value;



  sprintf(key, "Chemistry.PrintSecondaryFreeIon");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_secondary_freeion = switch_value;

  sprintf(key, "Chemistry.WriteSiloSecondaryFreeIon");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_secondary_freeion = switch_value;



  sprintf(key, "Chemistry.PrintSecondaryActivity");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_secondary_activity = switch_value;

  sprintf(key, "Chemistry.WriteSiloSecondaryActivity");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_secondary_activity = switch_value;



  sprintf(key, "Chemistry.PrintPrimarySorbed");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_sorbed = switch_value;

  sprintf(key, "Chemistry.WriteSiloPrimarySorbed");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->silo_sorbed = switch_value;



  sprintf(key, "Chemistry.PrintAll");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  print_all = switch_value;


  sprintf(key, "Chemistry.WriteSiloAll");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  write_silo_all = switch_value;


  if (print_all)
  {
    public_xtra->print_sorbed =
      public_xtra->print_secondary_activity =
        public_xtra->print_secondary_freeion =
          public_xtra->print_primary_activity =
            public_xtra->print_primary_freeion =
              public_xtra->print_mineral_SI =
                public_xtra->print_aqueous_rate =
                  public_xtra->print_pH =
                    public_xtra->print_CEC =
                      public_xtra->print_surf_dens =
                        public_xtra->print_mineral_surfarea =
                          public_xtra->print_mineral_volfx =
                            public_xtra->print_mineral_rate =
                              public_xtra->print_primary_mobile = print_all;
  }


  if (write_silo_all)
  {
    public_xtra->silo_sorbed =
      public_xtra->silo_secondary_activity =
        public_xtra->silo_secondary_freeion =
          public_xtra->silo_primary_activity =
            public_xtra->silo_primary_freeion =
              public_xtra->silo_mineral_SI =
                public_xtra->silo_aqueous_rate =
                  public_xtra->silo_pH =
                    public_xtra->silo_CEC =
                      public_xtra->silo_surf_dens =
                        public_xtra->silo_mineral_surfarea =
                          public_xtra->silo_mineral_volfx =
                            public_xtra->silo_mineral_rate =
                              public_xtra->silo_primary_mobile = write_silo_all;
  }


  if (public_xtra->silo_sorbed ||
      public_xtra->silo_secondary_activity ||
      public_xtra->silo_secondary_freeion ||
      public_xtra->silo_primary_activity ||
      public_xtra->silo_primary_freeion ||
      public_xtra->silo_mineral_SI ||
      public_xtra->silo_aqueous_rate ||
      public_xtra->silo_pH ||
      public_xtra->silo_CEC ||
      public_xtra->silo_surf_dens ||
      public_xtra->silo_mineral_surfarea ||
      public_xtra->silo_mineral_volfx ||
      public_xtra->silo_mineral_rate ||
      public_xtra->silo_primary_mobile
      )
  {
    WriteSiloInit(GlobalsOutFileName);
  }


  NA_FreeNameArray(switch_na);
  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * InitializeChemistryFreePublicXtra
 *--------------------------------------------------------------------------*/

void InitializeChemistryFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (public_xtra)
  {
    PFModuleFreeModule(public_xtra->set_chem_data);
    NA_FreeNameArray(public_xtra->ic_cond_na);
    NA_FreeNameArray(public_xtra->bc_cond_na);
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * InitializeChemistrySizeOfTempData
 *--------------------------------------------------------------------------*/

int InitializeChemistrySizeOfTempData()
{
  int sz = 0;

  return sz;
}
#endif
#endif /* HAVE_ALQUIMIA */
