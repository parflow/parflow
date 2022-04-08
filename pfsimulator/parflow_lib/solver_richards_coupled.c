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

/****************************************************************************
 *
 * Top level
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include "parflow.h"
#include "parflow_netcdf.h"
#include "metadata.h"
#include "cJSON.h"
#include "../amps/oas3/oas3_coupler.h"

#ifdef HAVE_SLURM
#include <slurm/slurm.h>
#endif

#include <unistd.h>
#include <string.h>
#include <float.h>
#include <limits.h>

#define PF_CLM_MAX_ROOT_NZ 20

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  PFModule *permeability_face;
  PFModule *advect_concen;
  PFModule *set_problem_data;
  PFModule *nonlin_solver;

  Problem *problem;

  int advect_order;
  double CFL;
  double drop_tol;
  int max_iterations;
  int max_convergence_failures; /* maximum number of convergence failures that are allowed */
  int lsm;                      /* land surface model */
  int terrain_following_grid;   /* @RMM flag for terrain following grid in NL fn eval, sets sslopes=toposl */
  int variable_dz;              /* @RMM flag for variable dz-multipliers */

  int print_subsurf_data;       /* print permeability/porosity? */
  int print_press;              /* print pressures? */
  int print_slopes;             /* print slopes? */
  int print_mannings;           /* print mannings? */
  int print_specific_storage;   /* print spec storage? */
  int print_top;                /* print top? */
  int print_velocities;         /* print velocities? */
  int print_satur;              /* print saturations? */
  int print_mask;               /* print mask? */
  int print_concen;             /* print concentrations? */
  int print_wells;              /* print well data? */
  int print_dzmult;             /* print dz multiplier? */
  int print_evaptrans;          /* print evaptrans? */
  int print_evaptrans_sum;      /* print evaptrans_sum? */
  int print_overland_sum;       /* print overland_sum? */
  int print_overland_bc_flux;   /* print overland outflow boundary condition flux? */
  int write_silo_subsurf_data;  /* write permeability/porosity? */
  int write_silo_press;         /* write pressures? */
  int write_silo_velocities;    /* write velocities? */
  int write_silo_satur;         /* write saturations? */
  int write_silo_concen;        /* write concentrations? */
  int write_silo_mask;          /* write mask? */
  int write_silo_evaptrans;     /* write evaptrans? */
  int write_silo_evaptrans_sum; /* write evaptrans sum? */
  int write_silo_slopes;        /* write slopes? */
  int write_silo_mannings;      /* write mannings? */
  int write_silo_specific_storage;      /* write specific storage? */
  int write_silo_top;           /* write top? */
  int write_silo_overland_sum;  /* write sum of overland outflow? */
  int write_silo_overland_bc_flux;      /* write overland outflow boundary condition flux? */
  int write_silo_dzmult;        /* write dz multiplier */
  int write_silopmpio_subsurf_data;     /* write permeability/porosity as PMPIO? */
  int write_silopmpio_press;    /* write pressures as PMPIO? */
  int write_silopmpio_velocities;       /* write velocities as PMPIO? */
  int write_silopmpio_satur;    /* write saturations as PMPIO? */
  int write_silopmpio_concen;   /* write concentrations as PMPIO? */
  int write_silopmpio_mask;     /* write mask as PMPIO? */
  int write_silopmpio_evaptrans;        /* write evaptrans as PMPIO? */
  int write_silopmpio_evaptrans_sum;    /* write evaptrans sum as PMPIO? */
  int write_silopmpio_slopes;   /* write slopes as PMPIO? */
  int write_silopmpio_mannings; /* write mannings as PMPIO? */
  int write_silopmpio_specific_storage; /* write specific storage as PMPIO? */
  int write_silopmpio_top;      /* write top as PMPIO? */
  int write_silopmpio_overland_sum;     /* write sum of overland outflow as PMPIO? */
  int write_silopmpio_overland_bc_flux; /* write overland outflow boundary condition flux as PMPIO? */
  int write_silopmpio_dzmult;   /* write dz multiplier as PMPIO? */
  int spinup;                   /* spinup flag, remove ponded water */
  int evap_trans_file;          /* read evap_trans as a SS file before advance richards */
  int evap_trans_file_transient;        /* read evap_trans as a transient file before advance richards timestep */
  char *evap_trans_filename;    /* File name for evap trans */
  int evap_trans_file_looping;  /* Loop over the flux files if we run out */



  int print_lsm_sink;           /* print LSM sink term? */
  int write_silo_CLM;           /* write CLM output as silo? */
  int write_silopmpio_CLM;      /* write CLM output as silo as PMPIO? */
  int print_CLM;                /* print CLM output as PFB? */
  int write_CLM_binary;         /* write binary output (**default**)? */
  int slope_accounting_CLM;     /* account for slopes in energy budget */

  int single_clm_file;          /* NBE: Write all CLM outputs into a single multi-layer PFB */

  /* KKu netcdf output flags */
  int write_netcdf_press;       /* write pressures? */
  int write_netcdf_satur;       /* write saturations? */
  int write_netcdf_evaptrans;   /* write evaptrans? */
  int write_netcdf_evaptrans_sum;       /* write evaptrans_sum? */
  int write_netcdf_overland_sum;        /* write overland_sum? */
  int write_netcdf_overland_bc_flux;    /* write overland_bc_flux? */
  int write_netcdf_mask;        /* write mask? */
  int write_netcdf_mannings;    /* write mask? */
  int write_netcdf_subsurface;  /* write subsurface? */
  int write_netcdf_slopes;      /* write subsurface? */
  int write_netcdf_dzmult;      /* write subsurface? */
  int numVarTimeVariant;        /*This variable is added to keep track of number of
                                 * time variant variable in NetCDF file */
  int numVarIni;                /*This variable is added to keep track of number of
                                 * time invariant variable in NetCDF file */
  int write_netcdf_clm;         /* Write CLM in NetCDF file? */
  int numCLMVarTimeVariant;     /* Number of CLM variables to be written in NetCDF file */

  int nc_evap_trans_file_transient;     /* read NetCDF evap_trans as a transient file before advance richards timestep */
  char *nc_evap_trans_filename; /* NetCDF File name for evap trans */
} PublicXtra;

typedef struct {
  PFModule *permeability_face;
  PFModule *advect_concen;
  PFModule *set_problem_data;

  PFModule *retardation;
  PFModule *phase_rel_perm;
  PFModule *ic_phase_pressure;
  PFModule *ic_phase_concen;
  PFModule *problem_saturation;
  PFModule *phase_density;
  PFModule *select_time_step;
  PFModule *l2_error_norm;
  PFModule *nonlin_solver;

  Grid *grid;
  Grid *grid2d;
  Grid *x_grid;
  Grid *y_grid;
  Grid *z_grid;

  ProblemData *problem_data;

  double *temp_data;

  /****************************************************************************
   * Local variables that need to be kept around
   *****************************************************************************/
  Vector *pressure;
  Vector *saturation;
  Vector *density;
  Vector *old_density;
  Vector *old_saturation;
  Vector *old_pressure;
  Vector *mask;

  Vector *evap_trans_sum;       /* running sum of evaporation and transpiration */
  Vector *overland_sum;
  Vector *ovrl_bc_flx;          /* vector containing outflow at the boundary */
  Vector *dz_mult;              /* vector containing dz multplier values for all cells */
  Vector *x_velocity;           /* vector containing x-velocity face values */
  Vector *y_velocity;           /* vector containing y-velocity face values */
  Vector *z_velocity;           /* vector containing z-velocity face values */

  double *time_log;
  double *dt_log;
  int *seq_log;
  int *dumped_log;
  char *recomp_log;
  char *dt_info_log;

  int file_number;
  int number_logged;
  int iteration_number;
  double dump_index;
} InstanceXtra;

void
SetupRichardsCoupled(PFModule * this_module)
{
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra *instance_xtra =
    (InstanceXtra*)PFModuleInstanceXtra(this_module);
  Problem *problem = (public_xtra->problem);
  PFModule *ic_phase_pressure = (instance_xtra->ic_phase_pressure);
  PFModule *phase_density = (instance_xtra->phase_density);
  PFModule *problem_saturation = (instance_xtra->problem_saturation);

  int print_press = (public_xtra->print_press);
  int print_satur = (public_xtra->print_satur);
  int print_wells = (public_xtra->print_wells);
  int print_velocities = (public_xtra->print_velocities);       //jjb

  ProblemData *problem_data = (instance_xtra->problem_data);
  PFModule *set_problem_data = (instance_xtra->set_problem_data);
  Grid *grid = (instance_xtra->grid);
  Grid *grid2d = (instance_xtra->grid2d);
  Grid *x_grid = (instance_xtra->x_grid);       //jjb
  Grid *y_grid = (instance_xtra->y_grid);       //jjb
  Grid *z_grid = (instance_xtra->z_grid);       //jjb

  double start_time = ProblemStartTime(problem);
  double stop_time = ProblemStopTime(problem);
  int start_count = ProblemStartCount(problem);

  char file_prefix[2048], file_type[2048], file_postfix[2048];
  char nc_postfix[2048];

  int take_more_time_steps;

  double t;
  double dt = 0.0;
  double gravity = ProblemGravity(problem);

  double dtmp;

  VectorUpdateCommHandle *handle;

  int any_file_dumped;

  t = start_time;
  dt = 0.0e0;

  NewMetadata(this_module);
  MetadataAddParflowDomainInfo(js_domains, this_module, grid);

  IfLogging(1)
  {
    int max_iterations = (public_xtra->max_iterations);

    instance_xtra->seq_log = talloc(int, max_iterations + 1);
    instance_xtra->time_log = talloc(double, max_iterations + 1);
    instance_xtra->dt_log = talloc(double, max_iterations + 1);
    instance_xtra->dt_info_log = talloc(char, max_iterations + 1);
    instance_xtra->dumped_log = talloc(int, max_iterations + 1);
    instance_xtra->recomp_log = talloc(char, max_iterations + 1);
    instance_xtra->number_logged = 0;
  }

  sprintf(file_prefix, "%s", GlobalsOutFileName);

  /* Do turning bands (and other stuff maybe) */
  PFModuleInvokeType(SetProblemDataInvoke, set_problem_data, (problem_data));
  ComputeTop(problem, problem_data);

  /* @RMM set subsurface slopes to topographic slopes if we have terrain following grid
   * turned on.  We might later make this an geometry or input file option but for now
   * it's just copying one vector into another */
  if (public_xtra->terrain_following_grid)
  {
    Copy(ProblemDataTSlopeX(problem_data),
         ProblemDataSSlopeX(problem_data));
    Copy(ProblemDataTSlopeY(problem_data),
         ProblemDataSSlopeY(problem_data));
    handle =
      InitVectorUpdate(ProblemDataSSlopeX(problem_data), VectorUpdateAll);
    FinalizeVectorUpdate(handle);
    handle =
      InitVectorUpdate(ProblemDataSSlopeY(problem_data), VectorUpdateAll);
    FinalizeVectorUpdate(handle);
  }

  /* @IMF -- set DZ multiplier from ProblemDataZmult */
  instance_xtra->dz_mult = ProblemDataZmult(problem_data);

  /* Write subsurface data */
  if (public_xtra->print_subsurf_data)
  {
    strcpy(file_postfix, "perm_x");
    WritePFBinary(file_prefix, file_postfix,
                  ProblemDataPermeabilityX(problem_data));

    strcpy(file_postfix, "perm_y");
    WritePFBinary(file_prefix, file_postfix,
                  ProblemDataPermeabilityY(problem_data));

    strcpy(file_postfix, "perm_z");
    WritePFBinary(file_prefix, file_postfix,
                  ProblemDataPermeabilityZ(problem_data));

    strcpy(file_postfix, "porosity");
    WritePFBinary(file_prefix, file_postfix,
                  ProblemDataPorosity(problem_data));

    // IMF -- added specific storage to subsurface bundle
    strcpy(file_postfix, "specific_storage");
    WritePFBinary(file_prefix, file_postfix,
                  ProblemDataSpecificStorage(problem_data));

    PFModuleOutputStaticType(SaturationOutputStaticInvoke, ProblemSaturation(problem), (file_prefix, problem_data));
    
    // Now add metadata entries:
    static const char* permeability_filenames[] = {
      "perm_x", "perm_y", "perm_z"
    };
    static const char* porosity_filenames[] = {
      "porosity"
    };
    static const char* storage_filenames[] = {
      "specific_storage"
    };
    MetadataAddStaticField(
                           js_inputs, file_prefix, "permeability", NULL, "cell", "subsurface",
                           sizeof(permeability_filenames) / sizeof(permeability_filenames[0]),
                           permeability_filenames);
    MetadataAddStaticField(
                           js_inputs, file_prefix, "porosity", NULL, "cell", "subsurface",
                           sizeof(porosity_filenames) / sizeof(porosity_filenames[0]),
                           porosity_filenames);
    MetadataAddStaticField(
                           js_inputs, file_prefix, "specific storage", "1/m", "cell", "subsurface",
                           sizeof(storage_filenames) / sizeof(storage_filenames[0]),
                           storage_filenames);
  }


  if (public_xtra->write_silo_subsurf_data)
  {
    strcpy(file_postfix, "");
    strcpy(file_type, "perm_x");
    WriteSilo(file_prefix, file_type, file_postfix,
              ProblemDataPermeabilityX(problem_data), t, 0,
              "PermeabilityX");

    strcpy(file_type, "perm_y");
    WriteSilo(file_prefix, file_type, file_postfix,
              ProblemDataPermeabilityY(problem_data), t, 0,
              "PermeabilityY");

    strcpy(file_type, "perm_z");
    WriteSilo(file_prefix, file_type, file_postfix,
              ProblemDataPermeabilityZ(problem_data), t, 0,
              "PermeabilityZ");

    strcpy(file_type, "porosity");
    WriteSilo(file_prefix, file_type, file_postfix,
              ProblemDataPorosity(problem_data), t, 0, "Porosity");

    // IMF -- added specific storage to subsurface bundle
    strcpy(file_type, "specific_storage");
    WriteSilo(file_prefix, file_type, file_postfix,
              ProblemDataSpecificStorage(problem_data), t, 0,
              "SpecificStorage");
  }

  if (public_xtra->write_silopmpio_subsurf_data)
  {
    strcpy(file_postfix, "");
    strcpy(file_type, "perm_x");
    WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                   ProblemDataPermeabilityX(problem_data), t, 0,
                   "PermeabilityX");

    strcpy(file_type, "perm_y");
    WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                   ProblemDataPermeabilityY(problem_data), t, 0,
                   "PermeabilityY");

    strcpy(file_type, "perm_z");
    WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                   ProblemDataPermeabilityZ(problem_data), t, 0,
                   "PermeabilityZ");

    strcpy(file_type, "porosity");
    WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                   ProblemDataPorosity(problem_data), t, 0, "Porosity");

    // IMF -- added specific storage to subsurface bundle
    strcpy(file_type, "specific_storage");
    WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                   ProblemDataSpecificStorage(problem_data), t, 0,
                   "SpecificStorage");
  }


  if (public_xtra->print_slopes)
  {
    strcpy(file_postfix, "slope_x");
    WritePFBinary(file_prefix, file_postfix,
                  ProblemDataTSlopeX(problem_data));

    strcpy(file_postfix, "slope_y");
    WritePFBinary(file_prefix, file_postfix,
                  ProblemDataTSlopeY(problem_data));

    static const char* slope_filenames[] = {
      "slope_x", "slope_y"
    };
    MetadataAddStaticField(
                           js_inputs, file_prefix, "slope", NULL, "cell", "surface",
                           sizeof(slope_filenames) / sizeof(slope_filenames[0]),
                           slope_filenames);
  }

  if (public_xtra->write_silo_slopes)
  {
    strcpy(file_postfix, "");
    strcpy(file_type, "slope_x");
    WriteSilo(file_prefix, file_type, file_postfix,
              ProblemDataTSlopeX(problem_data), t, 0, "SlopeX");

    strcpy(file_type, "slope_y");
    WriteSilo(file_prefix, file_type, file_postfix,
              ProblemDataTSlopeY(problem_data), t, 0, "SlopeY");
  }

  if (public_xtra->write_silopmpio_slopes)
  {
    strcpy(file_postfix, "");
    strcpy(file_type, "slope_x");
    WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                   ProblemDataTSlopeX(problem_data), t, 0, "SlopeX");

    strcpy(file_type, "slope_y");
    WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                   ProblemDataTSlopeY(problem_data), t, 0, "SlopeY");
  }

  if (public_xtra->print_mannings)
  {
    strcpy(file_postfix, "mannings");
    WritePFBinary(file_prefix, file_postfix,
                  ProblemDataMannings(problem_data));

    static const char* mannings_filenames[] = {
      "mannings"
    };
    MetadataAddStaticField(
                           js_inputs, file_prefix, "mannings", "s/m^(1/3)", "cell", "surface",
                           sizeof(mannings_filenames) / sizeof(mannings_filenames[0]),
                           mannings_filenames);
  }

  if (public_xtra->write_silo_mannings)
  {
    strcpy(file_postfix, "");
    strcpy(file_type, "mannings");
    WriteSilo(file_prefix, file_type, file_postfix,
              ProblemDataMannings(problem_data), t, 0, "Mannings");
  }

  if (public_xtra->write_silopmpio_mannings)
  {
    strcpy(file_postfix, "");
    strcpy(file_type, "mannings");
    WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                   ProblemDataMannings(problem_data), t, 0, "Mannings");
  }

  if (public_xtra->print_dzmult)
  {
    strcpy(file_postfix, "dz_mult");
    WritePFBinary(file_prefix, file_postfix, instance_xtra->dz_mult);

    static const char* dzmult_filenames[] = {
      "dzmult"
    };
    MetadataAddStaticField(
                           js_inputs, file_prefix, "dz multiplier", NULL, "cell", "subsurface",
                           sizeof(dzmult_filenames) / sizeof(dzmult_filenames[0]),
                           dzmult_filenames);
  }

  if (public_xtra->write_silo_dzmult)
  {
    strcpy(file_postfix, "");
    strcpy(file_type, "dz_mult");
    WriteSilo(file_prefix, file_type, file_postfix, instance_xtra->dz_mult,
              t, 0, "DZ_Multiplier");
  }

  if (public_xtra->write_silopmpio_dzmult)
  {
    strcpy(file_postfix, "");
    strcpy(file_type, "dz_mult");
    WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                   instance_xtra->dz_mult, t, 0, "DZ_Multiplier");
  }

  // IMF --
  // Lumped specific storage w/ subsurf bundle,
  // Left keys for individual printing for backward compatibility
  if (public_xtra->print_specific_storage)
  {
    strcpy(file_postfix, "specific_storage");
    WritePFBinary(file_prefix, file_postfix,
                  ProblemDataSpecificStorage(problem_data));

    // We may already have added this above. Do not re-add
    if (!MetaDataHasField(js_inputs, "specific storage"))
    {
      static const char* storage_filenames[] = {
        "specific_storage"
      };
      MetadataAddStaticField(
                             js_inputs, file_prefix, "specific storage", "1/m", "cell", "subsurface",
                             sizeof(storage_filenames) / sizeof(storage_filenames[0]),
                             storage_filenames);
    }
  }

  if (public_xtra->write_silo_specific_storage)
  {
    strcpy(file_postfix, "");
    strcpy(file_type, "specific_storage");
    WriteSilo(file_prefix, file_type, file_postfix,
              ProblemDataSpecificStorage(problem_data), t, 0,
              "SpecificStorage");
  }

  if (public_xtra->print_top)
  {
    printf("PrintTop -- not yet implemented\n");
    // strcpy(file_postfix, "top");
    // WritePFBinary(file_prefix, file_postfix, XXXXXXXXXXXXXXXX(problem_data));
  }

  if (public_xtra->write_silo_top)
  {
    printf("WriteSiloTop -- not yet implemented\n");
    // strcpy(file_postfix, "");
    // strcpy(file_type, "top");
    // WriteSilo(file_prefix, file_type, file_postfix, XXXXXXXXXXXXXXXXXXXXX(problem_data),
    //          t, 0, "Top");
  }

  if (!amps_Rank(amps_CommWorld))
  {
    PrintWellData(ProblemDataWellData(problem_data),
                  (WELLDATA_PRINTPHYSICAL | WELLDATA_PRINTVALUES));
  }

  /* Check to see if pressure solves are requested */
  /* start_count < 0 implies that subsurface data ONLY is requested */
  /*    Thus, we do not want to allocate memory or initialize storage for */
  /*    other variables.  */
  if (start_count < 0)
  {
    take_more_time_steps = 0;
  }
  else
  {
    take_more_time_steps = 1;
  }

  instance_xtra->iteration_number = instance_xtra->file_number = start_count;
  instance_xtra->dump_index = 1.0;

  if (((t >= stop_time)
       || (instance_xtra->iteration_number > public_xtra->max_iterations))
      && (take_more_time_steps == 1))
  {
    take_more_time_steps = 0;
    print_press = 0;
    print_satur = 0;
    print_wells = 0;
    print_velocities = 0;       //jjb
  }

  if (take_more_time_steps)
  {
    /*-------------------------------------------------------------------
     * Allocate and set up initial values
     *-------------------------------------------------------------------*/

    /* SGS FIXME why are these here and not created in instance_xtra ? */
    instance_xtra->pressure =
      NewVectorType(grid, 1, 1, vector_cell_centered);
    InitVectorAll(instance_xtra->pressure, -FLT_MAX);

    instance_xtra->saturation =
      NewVectorType(grid, 1, 1, vector_cell_centered);
    InitVectorAll(instance_xtra->saturation, -FLT_MAX);

    instance_xtra->density =
      NewVectorType(grid, 1, 1, vector_cell_centered);
    InitVectorAll(instance_xtra->density, 0.0);

    instance_xtra->old_pressure =
      NewVectorType(grid, 1, 1, vector_cell_centered);
    InitVectorAll(instance_xtra->old_pressure, 0.0);

    instance_xtra->old_saturation =
      NewVectorType(grid, 1, 1, vector_cell_centered);
    InitVectorAll(instance_xtra->old_saturation, 0.0);

    instance_xtra->old_density =
      NewVectorType(grid, 1, 1, vector_cell_centered);
    InitVectorAll(instance_xtra->old_density, 0.0);

    /*sk Initialize Overland flow boundary fluxes */
    instance_xtra->ovrl_bc_flx =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->ovrl_bc_flx, 0.0);

    if (public_xtra->write_silo_overland_sum
        || public_xtra->print_overland_sum
        || public_xtra->write_silopmpio_overland_sum
        || public_xtra->write_netcdf_overland_sum)
    {
      instance_xtra->overland_sum =
        NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
      InitVectorAll(instance_xtra->overland_sum, 0.0);
    }

    instance_xtra->mask = NewVectorType(grid, 1, 1, vector_cell_centered);
    InitVectorAll(instance_xtra->mask, 0.0);

    instance_xtra->evap_trans_sum =
      NewVectorType(grid, 1, 0, vector_cell_centered);
    InitVectorAll(instance_xtra->evap_trans_sum, 0.0);

    /* intialize vel vectors - jjb */
    instance_xtra->x_velocity =
      NewVectorType(x_grid, 1, 1, vector_side_centered_x);
    InitVectorAll(instance_xtra->x_velocity, 0.0);

    instance_xtra->y_velocity =
      NewVectorType(y_grid, 1, 1, vector_side_centered_y);
    InitVectorAll(instance_xtra->y_velocity, 0.0);

    instance_xtra->z_velocity =
      NewVectorType(z_grid, 1, 2, vector_side_centered_z);
    InitVectorAll(instance_xtra->z_velocity, 0.0);

    /* Set initial pressures and pass around ghost data to start */
    PFModuleInvokeType(ICPhasePressureInvoke,
                       ic_phase_pressure,
                       (instance_xtra->pressure, instance_xtra->mask,
                        problem_data, problem));

    handle = InitVectorUpdate(instance_xtra->pressure, VectorUpdateAll);
    FinalizeVectorUpdate(handle);

    /* Set initial densities and pass around ghost data to start */
    PFModuleInvokeType(PhaseDensityInvoke,
                       phase_density,
                       (0, instance_xtra->pressure, instance_xtra->density,
                        &dtmp, &dtmp, CALCFCN));

    handle = InitVectorUpdate(instance_xtra->density, VectorUpdateAll);
    FinalizeVectorUpdate(handle);

    /* Set initial saturations */
    PFModuleInvokeType(SaturationInvoke, problem_saturation,
                       (instance_xtra->saturation, instance_xtra->pressure,
                        instance_xtra->density, gravity, problem_data,
                        CALCFCN));

    handle = InitVectorUpdate(instance_xtra->pressure, VectorUpdateAll);
    FinalizeVectorUpdate(handle);


    /*****************************************************************/
    /*          Print out any of the requested initial data          */
    /*****************************************************************/

    any_file_dumped = 0;

    /*-------------------------------------------------------------------
     * Print out the initial well data?
     *-------------------------------------------------------------------*/

    if (print_wells)
    {
      WriteWells(file_prefix,
                 problem,
                 ProblemDataWellData(problem_data),
                 t, WELLDATA_WRITEHEADER);
    }
    sprintf(nc_postfix, "%05d", instance_xtra->file_number);
    if (public_xtra->write_netcdf_press || public_xtra->write_netcdf_satur
        || public_xtra->write_netcdf_mannings
        || public_xtra->write_netcdf_subsurface
        || public_xtra->write_netcdf_slopes
        || public_xtra->write_netcdf_mask
        || public_xtra->write_netcdf_dzmult)
    {
      WritePFNC(file_prefix, nc_postfix, t, instance_xtra->pressure,
                public_xtra->numVarTimeVariant, "time", 1, true,
                public_xtra->numVarIni);
    }
    /*-----------------------------------------------------------------
     * Print out the initial pressures?
     *-----------------------------------------------------------------*/

    if (print_press)
    {
      sprintf(file_postfix, "press.%05d", instance_xtra->file_number);
      WritePFBinary(file_prefix, file_postfix, instance_xtra->pressure);
      any_file_dumped = 1;

      static const char* press_filenames[] = {
        "press"
      };
      MetadataAddDynamicField(
                              js_outputs, file_prefix, t, 0, "pressure", "m", "cell", "subsurface",
                              sizeof(press_filenames) / sizeof(press_filenames[0]),
                              press_filenames);
    }

    if (public_xtra->write_silo_press)
    {
      sprintf(file_postfix, "%05d", instance_xtra->file_number);
      strcpy(file_type, "press");
      WriteSilo(file_prefix, file_type, file_postfix,
                instance_xtra->pressure, t, instance_xtra->file_number,
                "Pressure");
      any_file_dumped = 1;
    }

    if (public_xtra->write_silopmpio_press)
    {
      sprintf(file_postfix, "%05d", instance_xtra->file_number);
      strcpy(file_type, "press");
      WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                     instance_xtra->pressure, t,
                     instance_xtra->file_number, "Pressure");
      any_file_dumped = 1;
    }
    if (public_xtra->write_netcdf_press)
    {
      sprintf(file_postfix, "press.%05d", instance_xtra->file_number);
      sprintf(nc_postfix, "%05d", instance_xtra->file_number);
      WritePFNC(file_prefix, nc_postfix, t, instance_xtra->pressure,
                public_xtra->numVarTimeVariant, "pressure", 3, true,
                public_xtra->numVarIni);
      any_file_dumped = 1;
    }
    /*-----------------------------------------------------------------
     * Print out the initial saturations?
     *-----------------------------------------------------------------*/

    if (print_satur)
    {
      sprintf(file_postfix, "satur.%05d", instance_xtra->file_number);
      WritePFBinary(file_prefix, file_postfix,
                    instance_xtra->saturation);
      any_file_dumped = 1;

      static const char* satur_filenames[] = {
        "satur"
      };
      MetadataAddDynamicField(
                              js_outputs, file_prefix, t, 0, "saturation", NULL, "cell", "subsurface",
                              sizeof(satur_filenames) / sizeof(satur_filenames[0]),
                              satur_filenames);
    }

    if (public_xtra->write_silo_satur)
    {
      sprintf(file_postfix, "%05d", instance_xtra->file_number);
      strcpy(file_type, "satur");
      WriteSilo(file_prefix, file_type, file_postfix,
                instance_xtra->saturation, t, instance_xtra->file_number,
                "Saturation");
      any_file_dumped = 1;
    }

    if (public_xtra->write_silopmpio_satur)
    {
      sprintf(file_postfix, "%05d", instance_xtra->file_number);
      strcpy(file_type, "satur");
      WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                     instance_xtra->saturation, t,
                     instance_xtra->file_number, "Saturation");
      any_file_dumped = 1;
    }
    if (public_xtra->write_netcdf_satur)
    {
      sprintf(file_postfix, "satur.%05d", instance_xtra->file_number);
      sprintf(nc_postfix, "%05d", instance_xtra->file_number);
      WritePFNC(file_prefix, nc_postfix, t, instance_xtra->saturation,
                public_xtra->numVarTimeVariant, "saturation", 3, true,
                public_xtra->numVarIni);
      any_file_dumped = 1;
    }

    /*-----------------------------------------------------------------
     * Print out Mannings in NetCDF?
     *-----------------------------------------------------------------*/
    if (public_xtra->write_netcdf_mannings)
    {
      sprintf(nc_postfix, "%05d", instance_xtra->file_number);
      WritePFNC(file_prefix, nc_postfix, t,
                ProblemDataMannings(problem_data),
                public_xtra->numVarTimeVariant, "mannings", 2, true,
                public_xtra->numVarIni);
      any_file_dumped = 1;
    }

    /*-----------------------------------------------------------------
     * Print out Subsurface data in NetCDF?
     *-----------------------------------------------------------------*/
    if (public_xtra->write_netcdf_subsurface)
    {
      sprintf(nc_postfix, "%05d", instance_xtra->file_number);
      WritePFNC(file_prefix, nc_postfix, t,
                ProblemDataPermeabilityX(problem_data),
                public_xtra->numVarTimeVariant, "perm_x", 3, true,
                public_xtra->numVarIni);
      WritePFNC(file_prefix, nc_postfix, t,
                ProblemDataPermeabilityY(problem_data),
                public_xtra->numVarTimeVariant, "perm_y", 3, true,
                public_xtra->numVarIni);
      WritePFNC(file_prefix, nc_postfix, t,
                ProblemDataPermeabilityZ(problem_data),
                public_xtra->numVarTimeVariant, "perm_z", 3, true,
                public_xtra->numVarIni);
      WritePFNC(file_prefix, nc_postfix, t,
                ProblemDataPorosity(problem_data),
                public_xtra->numVarTimeVariant, "porosity", 3, true,
                public_xtra->numVarIni);
      WritePFNC(file_prefix, nc_postfix, t,
                ProblemDataSpecificStorage(problem_data),
                public_xtra->numVarTimeVariant, "specific_storage", 3,
                true, public_xtra->numVarIni);
      any_file_dumped = 1;
    }

    /*-----------------------------------------------------------------
     * Print out Slopes in NetCDF?
     *-----------------------------------------------------------------*/
    if (public_xtra->write_netcdf_slopes)
    {
      sprintf(nc_postfix, "%05d", instance_xtra->file_number);
      WritePFNC(file_prefix, nc_postfix, t,
                ProblemDataTSlopeX(problem_data),
                public_xtra->numVarTimeVariant, "slopex", 2, true,
                public_xtra->numVarIni);
      WritePFNC(file_prefix, nc_postfix, t,
                ProblemDataTSlopeY(problem_data),
                public_xtra->numVarTimeVariant, "slopey", 2, true,
                public_xtra->numVarIni);
      any_file_dumped = 1;
    }

    /*-----------------------------------------------------------------
     * Print out dz multipliers in NetCDF?
     *-----------------------------------------------------------------*/
    if (public_xtra->write_netcdf_dzmult)
    {
      sprintf(nc_postfix, "%05d", instance_xtra->file_number);
      WritePFNC(file_prefix, nc_postfix, t, instance_xtra->dz_mult,
                public_xtra->numVarTimeVariant, "DZ_Multiplier", 3, true,
                public_xtra->numVarIni);
      any_file_dumped = 1;
    }

    /*-----------------------------------------------------------------
     * Print out mask?
     *-----------------------------------------------------------------*/

    if (public_xtra->print_mask)
    {
      strcpy(file_postfix, "mask");
      WritePFBinary(file_prefix, file_postfix, instance_xtra->mask);
      any_file_dumped = 1;

      static const char* mask_filenames[] = {
        "mask"
      };
      MetadataAddStaticField(
                             js_inputs, file_prefix, "mask", NULL, "cell", "subsurface",
                             sizeof(mask_filenames) / sizeof(mask_filenames[0]),
                             mask_filenames);
    }

    if (public_xtra->write_netcdf_mask)
    {
      sprintf(nc_postfix, "%05d", instance_xtra->file_number);
      WritePFNC(file_prefix, nc_postfix, t, instance_xtra->mask,
                public_xtra->numVarTimeVariant, "mask", 3, true,
                public_xtra->numVarIni);
      any_file_dumped = 1;
    }


    if (public_xtra->write_silo_mask)
    {
      strcpy(file_postfix, "");
      strcpy(file_type, "mask");
      WriteSilo(file_prefix, file_type, file_postfix,
                instance_xtra->mask, t, instance_xtra->file_number,
                "Mask");
      any_file_dumped = 1;
    }

    if (public_xtra->write_silopmpio_mask)
    {
      strcpy(file_postfix, "");
      strcpy(file_type, "mask");
      WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                     instance_xtra->mask, t, instance_xtra->file_number,
                     "Mask");
      any_file_dumped = 1;
    }

    /* print initial velocities??? jjb */
    if (print_velocities)
    {
      sprintf(file_postfix, "velx.%05d", instance_xtra->file_number);
      WritePFBinary(file_prefix, file_postfix,
                    instance_xtra->x_velocity);
      static const char* velx_filenames[] = {
        "velx"
      };
      MetadataAddDynamicField(
                              js_outputs, file_prefix, t, 0, "x-velocity", "m/s", "x-face", "subsurface",
                              sizeof(velx_filenames) / sizeof(velx_filenames[0]),
                              velx_filenames);

      sprintf(file_postfix, "vely.%05d", instance_xtra->file_number);
      WritePFBinary(file_prefix, file_postfix,
                    instance_xtra->y_velocity);
      static const char* vely_filenames[] = {
        "vely"
      };
      MetadataAddDynamicField(
                              js_outputs, file_prefix, t, 0, "y-velocity", "m/s", "y-face", "subsurface",
                              sizeof(vely_filenames) / sizeof(vely_filenames[0]),
                              vely_filenames);

      sprintf(file_postfix, "velz.%05d", instance_xtra->file_number);
      WritePFBinary(file_prefix, file_postfix,
                    instance_xtra->z_velocity);
      static const char* velz_filenames[] = {
        "velz"
      };
      MetadataAddDynamicField(
                              js_outputs, file_prefix, t, 0, "z-velocity", "m/s", "z-face", "subsurface",
                              sizeof(velz_filenames) / sizeof(velz_filenames[0]),
                              velz_filenames);

      any_file_dumped = 1;
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
        printf
          ("Error: max_iterations reached, can't log anymore data\n");
        exit(1);
      }

      instance_xtra->seq_log[instance_xtra->number_logged] =
        instance_xtra->iteration_number;
      instance_xtra->time_log[instance_xtra->number_logged] = t;
      instance_xtra->dt_log[instance_xtra->number_logged] = dt;
      instance_xtra->dt_info_log[instance_xtra->number_logged] = 'i';
      if (any_file_dumped)
      {
        instance_xtra->dumped_log[instance_xtra->number_logged] =
          instance_xtra->file_number;
      }
      else
      {
        instance_xtra->dumped_log[instance_xtra->number_logged] = -1;
      }
      instance_xtra->recomp_log[instance_xtra->number_logged] = 'n';
      instance_xtra->number_logged++;
    }

    if (any_file_dumped)
    {
      instance_xtra->file_number++;
    }
  }                             /* End if take_more_time_steps */
}

void
AdvanceRichardsCoupled(PFModule * this_module, double start_time,      /* Starting time */
                       double stop_time,       /* Stopping time */
                       PFModule * time_step_control,   /* Use this module to control timestep if supplied */
                       Vector * evap_trans,    /* Flux from land surface model */
                       Vector ** pressure_out,         /* Output vars */
                       Vector ** porosity_out, Vector ** saturation_out)
{
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra *instance_xtra =
    (InstanceXtra*)PFModuleInstanceXtra(this_module);
  Problem *problem = (public_xtra->problem);

  int max_iterations = (public_xtra->max_iterations);
  int print_satur = (public_xtra->print_satur);
  int print_wells = (public_xtra->print_wells);

  PFModule *problem_saturation = (instance_xtra->problem_saturation);
  PFModule *phase_density = (instance_xtra->phase_density);
  PFModule *select_time_step = (instance_xtra->select_time_step);
  PFModule *l2_error_norm = (instance_xtra->l2_error_norm);
  PFModule *nonlin_solver = (instance_xtra->nonlin_solver);

  ProblemData *problem_data = (instance_xtra->problem_data);

  int start_count = ProblemStartCount(problem);
  double dump_interval = ProblemDumpInterval(problem);

  Vector *porosity = ProblemDataPorosity(problem_data);
  Vector *evap_trans_sum = instance_xtra->evap_trans_sum;
  Vector *overland_sum = instance_xtra->overland_sum;   /* sk: Vector of outflow at the boundary */

  Grid *grid = (instance_xtra->grid);
  Subgrid *subgrid;
  Subvector *p_sub, *s_sub, *et_sub, *m_sub;
  double *pp, *sp, *et, *ms;
  double sw_lat = .0;
  double sw_lon = .0;

  int any_file_dumped;
  int clm_file_dumped;
  int dump_files = 0;
  int retval;
  int converged;
  int take_more_time_steps;
  int conv_failures;
  int max_failures = public_xtra->max_convergence_failures;

  double t;
  double dt = 0.0;
  double ct = 0.0;
  double cdt = 0.0;
  double print_dt;
  double dtmp, err_norm;
  double gravity = ProblemGravity(problem);

  VectorUpdateCommHandle *handle;

  char dt_info;
  char file_prefix[2048], file_type[2048], file_postfix[2048];
  char nc_postfix[2048];

  int first_tstep = 1;

  sprintf(file_prefix, "%s", GlobalsOutFileName);

  //CPS oasis definition phase
  int nlon = GetInt("ComputationalGrid.NX");
  int nlat = GetInt("ComputationalGrid.NY");
  double pfl_step = GetDouble("TimeStep.Value");
  double pfl_stop = GetDouble("TimingInfo.StopTime");

  int is;
  ForSubgridI(is, GridSubgrids(grid))
  {
    double dx, dy;
    int nx, ny, ix, iy;

    subgrid = GridSubgrid(grid, is);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);

    CALL_oas_pfl_define(nx, ny, dx, dy, ix, iy, sw_lon, sw_lat, nlon, nlat,
                        pfl_step, pfl_stop);
  }
  amps_Sync(amps_CommWorld);

  /***********************************************************************/
  /*                                                                     */
  /*                Begin the main computational section                 */
  /*                                                                     */
  /***********************************************************************/

  // Initialize ct in either case
  ct = start_time;
  t = start_time;
  if (time_step_control)
  {
    PFModuleInvokeType(SelectTimeStepInvoke, time_step_control,
                       (&cdt, &dt_info, t, problem, problem_data));
  }
  else
  {
    PFModuleInvokeType(SelectTimeStepInvoke, select_time_step,
                       (&cdt, &dt_info, t, problem, problem_data));
  }
  dt = cdt;

  /*
   * Check to see if pressure solves are requested
   * start_count < 0 implies that subsurface data ONLY is requested
   * Thus, we do not want to allocate memory or initialize storage for
   * other variables.
   */

  if (start_count < 0)
  {
    take_more_time_steps = 0;
  }
  else
  {
    take_more_time_steps = 1;
  }

  do                            /* while take_more_time_steps */
  {
    if (t == ct)
    {
      ct += cdt;

      //CPS oasis exchange
      ForSubgridI(is, GridSubgrids(grid))
      {
        int ix, iy, nx, ny, nz, nx_f, ny_f;

        subgrid = GridSubgrid(grid, is);


        p_sub = VectorSubvector(instance_xtra->pressure, is);
        s_sub = VectorSubvector(instance_xtra->saturation, is);
        et_sub = VectorSubvector(evap_trans, is);
        m_sub = VectorSubvector(instance_xtra->mask, is);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);
        nx_f = SubvectorNX(et_sub);
        ny_f = SubvectorNY(et_sub);


        sp = SubvectorData(s_sub);
        pp = SubvectorData(p_sub);
        et = SubvectorData(et_sub);
        ms = SubvectorData(m_sub);

        //CPS       amps_Printf("Calling oasis send/receive for time  %3.1f \n", t);
        CALL_send_fld2_clm(pp, sp, ms, ix, iy, nx, ny, nz, nx_f, ny_f,
                           t);
        amps_Sync(amps_CommWorld);
        CALL_receive_fld2_clm(et, ms, ix, iy, nx, ny, nz, nx_f, ny_f, t);
      }
      amps_Sync(amps_CommWorld);
      handle = InitVectorUpdate(evap_trans, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      // IMF: Added to include CLM dumps in file_number updating.
      clm_file_dumped = 0;
    }                           //Endif to check whether an entire dt is complete

    converged = 1;
    conv_failures = 0;




    do                          /* while not converged */
    {
      /*
       * Record amount of memory in use.
       */

      recordMemoryInfo();

      /*******************************************************************/
      /*                  Compute time step                              */
      /*******************************************************************/
      if (converged)
      {
        if (time_step_control)
        {
          PFModuleInvokeType(SelectTimeStepInvoke, time_step_control,
                             (&dt, &dt_info, t, problem,
                              problem_data));
        }
        else
        {
          PFModuleInvokeType(SelectTimeStepInvoke, select_time_step,
                             (&dt, &dt_info, t, problem,
                              problem_data));
        }

        PFVCopy(instance_xtra->density, instance_xtra->old_density);
        PFVCopy(instance_xtra->saturation,
                instance_xtra->old_saturation);
        PFVCopy(instance_xtra->pressure, instance_xtra->old_pressure);
      }
      else                      /* Not converged, so decrease time step */
      {
        t = t - dt;

        double new_dt = 0.5 * dt;

        // If time increment is too small don't try to cut in half.
        {
          double test_time = t + new_dt;
          double diff_time = test_time - t;

          if (diff_time > TIME_EPSILON)
          {
            dt = new_dt;
          }
          else
          {
            PARFLOW_ERROR
              ("Time increment is too small; solver has failed\n");
          }
        }

        PFVCopy(instance_xtra->old_density, instance_xtra->density);
        PFVCopy(instance_xtra->old_saturation,
                instance_xtra->saturation);
        PFVCopy(instance_xtra->old_pressure, instance_xtra->pressure);
      }                         // End set t and dt based on convergence

      // CPS added to fix oasis exchange break due to parflow time stepping reduction
      // Note ct is time we want to advance to at this point
      if (t + dt > ct)
      {
        double new_dt = ct - t;

        // If time increment is too small we have a problem. Just halt
        {
          double test_time = t + new_dt;
          double diff_time = test_time - t;

          if (diff_time > TIME_EPSILON)
          {
            dt = new_dt;
          }
          else
          {
            PARFLOW_ERROR
              ("Time increment is too small; OASIS wants a small timestep\n");
            break;
          }
        }
      }

      /*--------------------------------------------------------------
       * If we are printing out results, then determine if we need
       * to print them after this time step.
       *
       * If we are dumping output at real time intervals, the value
       * of dt may be changed.  If this happens, we want to
       * compute/evolve all values.  We also set `dump_info' to `p'
       * to indicate that the dump interval decided the time step for
       * this iteration.
       *--------------------------------------------------------------*/

      // Print ParFlow output?
      dump_files = 0;
      if (dump_interval > 0)
      {
        print_dt =
          ProblemStartTime(problem) +
          instance_xtra->dump_index * dump_interval - t;

        if ((dt + TIME_EPSILON) > print_dt)
        {
          /*
           * if the difference is small don't try to compute
           * at print_dt, just use dt.  This will
           * output slightly off in time but avoids
           * extremely small dt values.
           */
          if (fabs(dt - print_dt) > TIME_EPSILON)
          {
            dt = print_dt;
          }
          dt_info = 'p';

          dump_files = 1;
        }
      }
      else if (dump_interval < 0)
      {
        if ((instance_xtra->iteration_number %
             (-(int)dump_interval)) == 0)
        {
          dump_files = 1;
        }
      }
      else
      {
        dump_files = 0;
      }

      /*--------------------------------------------------------------
       * If this is the last iteration, set appropriate variables.
       *--------------------------------------------------------------*/
      if ((t + dt) >= stop_time)
      {
        double new_dt = stop_time - t;

        double test_time = t + new_dt;
        double diff_time = test_time - t;

        if (diff_time > TIME_EPSILON)
        {
          dt = new_dt;
        }
        else
        {
          // PARFLOW_ERROR("Time increment is too small for last iteration\n");
          amps_Printf
            ("Time increment is too small for last iteration \n");
          //@RMM had to get rid of the error trap, was driving me crazy that it doesn't complete the log file
        }

        dt = new_dt;

        dt_info = 'f';
      }

      t += dt;


      /*******************************************************************/
      /*          Solve the nonlinear system for this time step          */
      /*******************************************************************/

      retval = PFModuleInvokeType(NonlinSolverInvoke, nonlin_solver,
                                  (instance_xtra->pressure,
                                   instance_xtra->density,
                                   instance_xtra->old_density,
                                   instance_xtra->saturation,
                                   instance_xtra->old_saturation,
                                   t, dt,
                                   problem_data,
                                   instance_xtra->old_pressure,
                                   evap_trans,
                                   instance_xtra->ovrl_bc_flx,
                                   instance_xtra->x_velocity,
                                   instance_xtra->y_velocity,
                                   instance_xtra->z_velocity));

      if (retval != 0)
      {
        converged = 0;
        conv_failures++;
      }
      else
      {
        converged = 1;
      }

      if (conv_failures >= max_failures)
      {
        take_more_time_steps = 0;
        if (!amps_Rank(amps_CommWorld))
        {
          amps_Printf("Error: Time step failed for time %12.4e.\n",
                      t);
          amps_Printf("Shutting down.\n");
        }
      }
    }                           /* Ends do for convergence of time step loop */
    while ((!converged) && (conv_failures < max_failures));

    instance_xtra->iteration_number++;

    /***************************************************************
    *         spinup - remove excess pressure at land surface     *
    ***************************************************************/
    //int spinup = 1;
    if (public_xtra->spinup == 1)
    {
      GrGeomSolid *gr_domain = ProblemDataGrDomain(problem_data);

      int i, j, k, r, is;
      int ix, iy, iz;
      int nx, ny, nz;
      int ip;
      // JLW add declarations for use without CLM
      Subvector *p_sub_sp;
      double *pp_sp;

      Subgrid *subgrid;
      Grid *grid = VectorGrid(evap_trans_sum);

      ForSubgridI(is, GridSubgrids(grid))
      {
        subgrid = GridSubgrid(grid, is);
        p_sub_sp = VectorSubvector(instance_xtra->pressure, is);

        r = SubgridRX(subgrid);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        pp_sp = SubvectorData(p_sub_sp);

        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ip = SubvectorEltIndex(p_sub_sp, i, j, k);
          // printf(" %d %d %d %d  \n",i,j,k,ip);
          // printf(" pp[ip] %10.3f \n",pp[ip]);
          // printf(" NZ: %d \n",nz);
          if (k == (nz - 1))
          {
            //   printf(" %d %d %d %d  \n",i,j,k,ip);
            //   printf(" pp[ip] %10.3f \n",pp[ip]);

            if (pp_sp[ip] > 0.0)
            {
              printf(" pressure-> 0 %d %d %d %10.3f \n", i, j, k,
                     pp_sp[ip]); pp_sp[ip] = 0.0;
            }
          }
        }
                     );
      }
    }

    /* velocity updates - not sure these are necessary jjb */
    handle = InitVectorUpdate(instance_xtra->x_velocity, VectorUpdateAll);
    FinalizeVectorUpdate(handle);

    handle = InitVectorUpdate(instance_xtra->y_velocity, VectorUpdateAll);
    FinalizeVectorUpdate(handle);

    handle = InitVectorUpdate(instance_xtra->z_velocity, VectorUpdateAll);
    FinalizeVectorUpdate(handle);


    /* Calculate densities and saturations for the new pressure. */
    PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                       (0, instance_xtra->pressure, instance_xtra->density,
                        &dtmp, &dtmp, CALCFCN));
    handle = InitVectorUpdate(instance_xtra->density, VectorUpdateAll);
    FinalizeVectorUpdate(handle);

    PFModuleInvokeType(SaturationInvoke, problem_saturation,
                       (instance_xtra->saturation, instance_xtra->pressure,
                        instance_xtra->density, gravity, problem_data,
                        CALCFCN));

    /***************************************************************
     * Compute running sum of evap trans for water balance
     **************************************************************/
    if (public_xtra->write_silo_evaptrans_sum
        || public_xtra->print_evaptrans_sum
        || public_xtra->write_netcdf_evaptrans_sum)
    {
      EvapTransSum(problem_data, dt, evap_trans_sum, evap_trans);
    }

    /***************************************************************
     * Compute running sum of overland outflow for water balance
     **************************************************************/
    if (public_xtra->write_silo_overland_sum
        || public_xtra->print_overland_sum
        || public_xtra->write_netcdf_overland_sum)
    {
      OverlandSum(problem_data,
                  instance_xtra->pressure,
                  dt, instance_xtra->overland_sum);
    }

    /***************************************************************/
    /*                 Print the pressure and saturation           */
    /***************************************************************/

    /* Dump the pressure, saturation, surface fluxes at this time-step */
    any_file_dumped = 0;
    if (dump_files)
    {
      sprintf(nc_postfix, "%05d", instance_xtra->file_number);
      /*KKU: Writing Current time variable value to NC file */
      if (public_xtra->write_netcdf_press
          || public_xtra->write_netcdf_satur
          || public_xtra->write_netcdf_evaptrans
          || public_xtra->write_netcdf_evaptrans_sum
          || public_xtra->write_netcdf_overland_sum
          || public_xtra->write_netcdf_overland_bc_flux)
      {
        WritePFNC(file_prefix, nc_postfix, t, instance_xtra->pressure,
                  public_xtra->numVarTimeVariant, "time", 1, false,
                  public_xtra->numVarIni);
      }

      instance_xtra->dump_index++;

      if (public_xtra->print_press)
      {
        sprintf(file_postfix, "press.%05d",
                instance_xtra->file_number);
        WritePFBinary(file_prefix, file_postfix,
                      instance_xtra->pressure);
        any_file_dumped = 1;

        // Update with new timesteps
        MetadataAddDynamicField(
                                js_outputs, file_prefix, t, instance_xtra->file_number,
                                "pressure", "m", "cell", "subsurface", 0, NULL);
      }

      if (public_xtra->write_silo_press)
      {
        sprintf(file_postfix, "%05d", instance_xtra->file_number);
        sprintf(file_type, "press");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->pressure, t,
                  instance_xtra->file_number, "Pressure");
        any_file_dumped = 1;
      }

      if (public_xtra->write_silopmpio_press)
      {
        sprintf(file_postfix, "%05d", instance_xtra->file_number);
        sprintf(file_type, "press");
        WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                       instance_xtra->pressure, t,
                       instance_xtra->file_number, "Pressure");
        any_file_dumped = 1;
      }
      if (public_xtra->write_netcdf_press)
      {
        sprintf(file_postfix, "press.%05d",
                instance_xtra->file_number);
        sprintf(nc_postfix, "%05d", instance_xtra->file_number);
        WritePFNC(file_prefix, nc_postfix, t, instance_xtra->pressure,
                  public_xtra->numVarTimeVariant, "pressure", 3, false,
                  public_xtra->numVarIni);
        any_file_dumped = 1;
      }

      if (public_xtra->print_velocities)        //jjb
      {
        sprintf(file_postfix, "velx.%05d", instance_xtra->file_number);
        WritePFBinary(file_prefix, file_postfix,
                      instance_xtra->x_velocity);
        // Update with new timesteps
        MetadataAddDynamicField(
                                js_outputs, file_prefix, t, instance_xtra->file_number,
                                "x-velocity", "m/s", "x-face", "subsurface", 0, NULL);

        sprintf(file_postfix, "vely.%05d", instance_xtra->file_number);
        WritePFBinary(file_prefix, file_postfix,
                      instance_xtra->y_velocity);
        // Update with new timesteps
        MetadataAddDynamicField(
                                js_outputs, file_prefix, t, instance_xtra->file_number,
                                "y-velocity", "m/s", "y-face", "subsurface", 0, NULL);

        sprintf(file_postfix, "velz.%05d", instance_xtra->file_number);
        WritePFBinary(file_prefix, file_postfix,
                      instance_xtra->z_velocity);
        // Update with new timesteps
        MetadataAddDynamicField(
                                js_outputs, file_prefix, t, instance_xtra->file_number,
                                "z-velocity", "m/s", "z-face", "subsurface", 0, NULL);

        any_file_dumped = 1;

      }


      if (public_xtra->print_satur)
      {
        sprintf(file_postfix, "satur.%05d",
                instance_xtra->file_number);
        WritePFBinary(file_prefix, file_postfix,
                      instance_xtra->saturation);
        any_file_dumped = 1;

        // Update with new timesteps
        MetadataAddDynamicField(
                                js_outputs, file_prefix, t, instance_xtra->file_number,
                                "saturation", "1/m", "cell", "subsurface", 0, NULL);
      }

      if (public_xtra->write_silo_satur)
      {
        sprintf(file_postfix, "%05d", instance_xtra->file_number);
        sprintf(file_type, "satur");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->saturation, t,
                  instance_xtra->file_number, "Saturation");
        any_file_dumped = 1;
      }

      if (public_xtra->write_silopmpio_satur)
      {
        sprintf(file_postfix, "%05d", instance_xtra->file_number);
        sprintf(file_type, "satur");
        WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                       instance_xtra->saturation, t,
                       instance_xtra->file_number, "Saturation");
        any_file_dumped = 1;
      }
      if (public_xtra->write_netcdf_satur)
      {
        sprintf(file_postfix, "satur.%05d",
                instance_xtra->file_number);
        sprintf(nc_postfix, "%05d", instance_xtra->file_number);
        WritePFNC(file_prefix, nc_postfix, t,
                  instance_xtra->saturation,
                  public_xtra->numVarTimeVariant, "saturation", 3,
                  false, public_xtra->numVarIni);
        any_file_dumped = 1;
      }

      if (public_xtra->print_evaptrans)
      {
        sprintf(file_postfix, "evaptrans.%05d",
                instance_xtra->file_number);
        WritePFBinary(file_prefix, file_postfix, evap_trans);
        any_file_dumped = 1;

        // Update with new timesteps
        /* No initial call to add the field
         * MetadataAddDynamicField(
         * js_outputs, file_prefix, t, instance_xtra->file_number,
         * "evapotranspiration", "mm", "cell", "subsurface", 0, NULL);
         */
      }


      if (public_xtra->write_silo_evaptrans)
      {
        sprintf(file_postfix, "%05d", instance_xtra->file_number);
        sprintf(file_type, "evaptrans");
        WriteSilo(file_prefix, file_type, file_postfix, evap_trans,
                  t, instance_xtra->file_number, "EvapTrans");
        any_file_dumped = 1;
      }

      if (public_xtra->write_silopmpio_evaptrans)
      {
        sprintf(file_postfix, "%05d", instance_xtra->file_number);
        sprintf(file_type, "evaptrans");
        WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                       evap_trans, t, instance_xtra->file_number,
                       "EvapTrans");
        any_file_dumped = 1;
      }

      if (public_xtra->write_netcdf_evaptrans)
      {
        sprintf(nc_postfix, "%05d", instance_xtra->file_number);
        WritePFNC(file_prefix, nc_postfix, t, evap_trans,
                  public_xtra->numVarTimeVariant, "evaptrans", 3,
                  false, public_xtra->numVarIni);
        any_file_dumped = 1;
      }


      if (public_xtra->print_evaptrans_sum
          || public_xtra->write_silo_evaptrans_sum
          || public_xtra->write_netcdf_evaptrans_sum)
      {
        if (public_xtra->write_netcdf_evaptrans_sum)
        {
          sprintf(nc_postfix, "%05d", instance_xtra->file_number);
          WritePFNC(file_prefix, nc_postfix, t, evap_trans_sum,
                    public_xtra->numVarTimeVariant, "evaptrans_sum",
                    3, false, public_xtra->numVarIni);
          any_file_dumped = 1;
        }

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
          WriteSilo(file_prefix, file_type, file_postfix,
                    evap_trans_sum, t, instance_xtra->file_number,
                    "EvapTransSum");
          any_file_dumped = 1;
        }


        if (public_xtra->write_silopmpio_evaptrans_sum)
        {
          sprintf(file_postfix, "%05d", instance_xtra->file_number);
          sprintf(file_type, "evaptranssum");
          WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                         evap_trans_sum, t,
                         instance_xtra->file_number, "EvapTransSum");
          any_file_dumped = 1;
        }

        /* reset sum after output */
        PFVConstInit(0.0, evap_trans_sum);
      }

      if (public_xtra->print_overland_sum
          || public_xtra->write_silo_overland_sum
          || public_xtra->write_netcdf_overland_sum)
      {
        if (public_xtra->write_netcdf_overland_sum)
        {
          sprintf(nc_postfix, "%05d", instance_xtra->file_number);
          WritePFNC(file_prefix, nc_postfix, t, overland_sum,
                    public_xtra->numVarTimeVariant, "overland_sum",
                    2, false, public_xtra->numVarIni);
          any_file_dumped = 1;
        }

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
          WriteSilo(file_prefix, file_type, file_postfix,
                    overland_sum, t, instance_xtra->file_number,
                    "OverlandSum");
          any_file_dumped = 1;
        }

        if (public_xtra->write_silopmpio_overland_sum)
        {
          sprintf(file_postfix, "%05d", instance_xtra->file_number);
          sprintf(file_type, "overlandsum");
          WriteSiloPMPIO(file_prefix, file_type, file_postfix,
                         overland_sum, t, instance_xtra->file_number,
                         "OverlandSum");
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

        // Update with new timesteps
        /* No initial call to add the field
         * MetadataAddDynamicField(
         * js_outputs, file_prefix, t, instance_xtra->file_number,
         * "overland bc flux", "m", "cell", "subsurface", 0, NULL);
         */
      }

      if (public_xtra->write_netcdf_overland_bc_flux)
      {
        sprintf(nc_postfix, "%05d", instance_xtra->file_number);
        WritePFNC(file_prefix, nc_postfix, t,
                  instance_xtra->ovrl_bc_flx,
                  public_xtra->numVarTimeVariant, "overland_bc_flux",
                  2, false, public_xtra->numVarIni);
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

      if (public_xtra->write_silopmpio_overland_bc_flux)
      {
        sprintf(file_postfix, "%05d", instance_xtra->file_number);
        sprintf(file_type, "overland_bc_flux");
        WriteSiloPMPIO(file_prefix, file_type, file_postfix,
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

        // TODO: Add metadata here? print_lsm_sink seems to be superseded by other flags.
      }
    }                           // End of if (dump_files)

    /***************************************************************/
    /*             Print CLM output files at this time             */
    /***************************************************************/

    /***************************************************************/
    /*             Compute the l2 error                            */
    /***************************************************************/

    PFModuleInvokeType(L2ErrorNormInvoke, l2_error_norm,
                       (t, instance_xtra->pressure, problem_data,
                        &err_norm));
    if ((!amps_Rank(amps_CommWorld)) && (err_norm >= 0.0))
    {
      amps_Printf("l2-error in pressure: %20.8e\n", err_norm);
      amps_Printf("tcl: set pressure_l2_error(%d) %20.8e\n",
                  instance_xtra->iteration_number, err_norm);
      fflush(NULL);
    }

    /*******************************************************************/
    /*                   Print the Well Data                           */
    /*******************************************************************/

    if (print_wells && dump_files)
    {
      WriteWells(file_prefix,
                 problem,
                 ProblemDataWellData(problem_data),
                 t, WELLDATA_DONTWRITEHEADER);
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
TeardownRichardsCoupled(PFModule * this_module)
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

  if (instance_xtra->evap_trans_sum)
  {
    FreeVector(instance_xtra->evap_trans_sum);
  }

  if (instance_xtra->overland_sum)
  {
    FreeVector(instance_xtra->overland_sum);
  }

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

    log_file = OpenLogFile("SolverRichardsCoupled");

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
 * SolverRichardsCoupledInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule *
SolverRichardsCoupledInitInstanceXtra()
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
  parameter_sz = PFModuleSizeOfTempData(instance_xtra->problem_saturation)
                 + PFModuleSizeOfTempData(instance_xtra->phase_rel_perm);

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

  temp_data_placeholder +=
    pfmax(PFModuleSizeOfTempData(instance_xtra->retardation),
          PFModuleSizeOfTempData(instance_xtra->advect_concen));
  /* set temporary vector data used for advection */

  temp_data += temp_data_size;

  PFModuleInstanceXtra(this_module) = instance_xtra;

  return this_module;
}

/*--------------------------------------------------------------------------
 * SolverRichardsCoupledFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void
SolverRichardsCoupledFreeInstanceXtra()
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

    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * SolverRichardsCoupledNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule *
SolverRichardsCoupledNewPublicXtra(char *name)
{
  PFModule *this_module = ThisPFModule;
  PublicXtra *public_xtra;

  char key[IDB_MAX_KEY_LEN];

  char *switch_name;
  int switch_value;
  NameArray switch_na;
  NameArray nonlin_switch_na;
  NameArray lsm_switch_na;

  switch_na = NA_NewNameArray("False True");

  public_xtra = ctalloc(PublicXtra, 1);

  (public_xtra->permeability_face) = PFModuleNewModule(PermeabilityFace, ());
  (public_xtra->advect_concen) = PFModuleNewModule(Godunov, ());
  (public_xtra->set_problem_data) = PFModuleNewModule(SetProblemData, ());
  (public_xtra->problem) = NewProblem(RichardsSolve);

  nonlin_switch_na = NA_NewNameArray("KINSol");
  sprintf(key, "%s.NonlinearSolver", name);
  switch_name = GetStringDefault(key, "KINSol");
  switch_value = NA_NameToIndex(nonlin_switch_na, switch_name);
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
      InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
                 key);
    }
  }
  NA_FreeNameArray(nonlin_switch_na);

  lsm_switch_na = NA_NewNameArray("none CLM");
  sprintf(key, "%s.LSM", name);
  switch_name = GetStringDefault(key, "none");
  switch_value = NA_NameToIndex(lsm_switch_na, switch_name);
  switch (switch_value)
  {
    case 0:
    {
      public_xtra->lsm = 0;
      break;
    }

    case 1:
    {
      InputError
        ("Error: <%s> used for key <%s> but this version of Parflow is compiled without CLM\n",
        switch_name, key);
      break;
    }

    default:
    {
      InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
                 key);
    }
  }
  NA_FreeNameArray(lsm_switch_na);

  /* IMF: Following are only used /w CLM */

  //CPS
  /* @RMM added switch for terrain-following grid */
  /* RMM set terrain grid (default=False) */
  sprintf(key, "%s.TerrainFollowingGrid", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
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
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_subsurf_data = switch_value;

  sprintf(key, "%s.PrintSlopes", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_slopes = switch_value;

  sprintf(key, "%s.PrintMannings", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_mannings = switch_value;

  sprintf(key, "%s.PrintSpecificStorage", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_specific_storage = switch_value;

  sprintf(key, "%s.PrintTop", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_top = switch_value;

  sprintf(key, "%s.PrintPressure", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_press = switch_value;

  sprintf(key, "%s.PrintVelocities", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_velocities = switch_value;

  sprintf(key, "%s.PrintSaturation", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_satur = switch_value;

  sprintf(key, "%s.PrintConcentration", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_concen = switch_value;

  sprintf(key, "%s.PrintDZMultiplier", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_dzmult = switch_value;

  sprintf(key, "%s.PrintMask", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_mask = switch_value;

  sprintf(key, "%s.PrintEvapTrans", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_evaptrans = switch_value;

  sprintf(key, "%s.PrintEvapTransSum", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_evaptrans_sum = switch_value;

  sprintf(key, "%s.PrintOverlandSum", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_overland_sum = switch_value;

  sprintf(key, "%s.PrintOverlandBCFlux", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_overland_bc_flux = switch_value;

  sprintf(key, "%s.PrintWells", name);
  switch_name = GetStringDefault(key, "True");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_wells = switch_value;

  // SGS TODO
  // Need to add this to the user manual, this is new for LSM stuff that was added.
  sprintf(key, "%s.PrintLSMSink", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->print_lsm_sink = switch_value;

  if (public_xtra->print_lsm_sink)
  {
    InputError("Error: setting %s to %s but do not have CLM\n",
               switch_name, key);
  }

  /* Silo file writing control */
  sprintf(key, "%s.WriteSiloSubsurfData", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silo_subsurf_data = switch_value;

  sprintf(key, "%s.WriteSiloPressure", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silo_press = switch_value;

  sprintf(key, "%s.WriteSiloVelocities", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silo_velocities = switch_value;

  sprintf(key, "%s.WriteSiloSaturation", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silo_satur = switch_value;

  sprintf(key, "%s.WriteSiloEvapTrans", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silo_evaptrans = switch_value;

  sprintf(key, "%s.WriteSiloEvapTransSum", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silo_evaptrans_sum = switch_value;

  sprintf(key, "%s.WriteSiloOverlandSum", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silo_overland_sum = switch_value;

  sprintf(key, "%s.WriteSiloOverlandBCFlux", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silo_overland_bc_flux = switch_value;

  sprintf(key, "%s.WriteSiloDZMultiplier", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
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
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  if (switch_value == 1)
  {
    public_xtra->numVarTimeVariant++;
    public_xtra->numVarIni++;
  }
  public_xtra->write_netcdf_press = switch_value;


  sprintf(key, "NetCDF.WriteSaturation");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  if (switch_value == 1)
  {
    public_xtra->numVarTimeVariant++;
    public_xtra->numVarIni++;
  }
  public_xtra->write_netcdf_satur = switch_value;

  sprintf(key, "NetCDF.WriteEvapTrans");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  if (switch_value == 1)
  {
    public_xtra->numVarTimeVariant++;
  }
  public_xtra->write_netcdf_evaptrans = switch_value;

  sprintf(key, "NetCDF.WriteEvapTransSum");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  if (switch_value == 1)
  {
    public_xtra->numVarTimeVariant++;
  }
  public_xtra->write_netcdf_evaptrans_sum = switch_value;

  sprintf(key, "NetCDF.WriteOverlandSum");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  if (switch_value == 1)
  {
    public_xtra->numVarTimeVariant++;
  }
  public_xtra->write_netcdf_overland_sum = switch_value;

  sprintf(key, "NetCDF.WriteOverlandBCFlux");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  if (switch_value == 1)
  {
    public_xtra->numVarTimeVariant++;
  }
  public_xtra->write_netcdf_overland_bc_flux = switch_value;

  sprintf(key, "NetCDF.WriteMannings");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  if (switch_value == 1)
  {
    public_xtra->numVarIni++;
  }
  public_xtra->write_netcdf_mannings = switch_value;

  sprintf(key, "NetCDF.WriteSubsurface");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  if (switch_value == 1)
  {
    /*Increamenting by 5 for x, y, z permiability, porosity and specific storage */
    public_xtra->numVarIni = public_xtra->numVarIni + 5;
  }
  public_xtra->write_netcdf_subsurface = switch_value;

  sprintf(key, "NetCDF.WriteSlopes");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  if (switch_value == 1)
  {
    /*Increamenting by 2 for x, y slopes */
    public_xtra->numVarIni = public_xtra->numVarIni + 2;
  }
  public_xtra->write_netcdf_slopes = switch_value;

  sprintf(key, "NetCDF.WriteDZMultiplier");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  if (switch_value == 1)
  {
    public_xtra->numVarIni++;
  }
  public_xtra->write_netcdf_dzmult = switch_value;

  sprintf(key, "NetCDF.WriteMask");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
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
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
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

  if (public_xtra->write_silo_overland_bc_flux)
  {
    InputError("Error: setting %s to %s but do not have CLM\n",
               switch_name, key);
  }

  sprintf(key, "%s.WriteSiloConcentration", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silo_concen = switch_value;

  sprintf(key, "%s.WriteSiloMask", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silo_mask = switch_value;


  sprintf(key, "%s.WriteSiloSlopes", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silo_slopes = switch_value;

  sprintf(key, "%s.WriteSiloMannings", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silo_mannings = switch_value;

  sprintf(key, "%s.WriteSiloSpecificStorage", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silo_specific_storage = switch_value;

  sprintf(key, "%s.WriteSiloTop", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
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

  /* @RMM -- added control block for silo pmpio
   * writing.  Previous silo is backward compat/included and true/false
   * switches work the same as SILO and PFB.  We can change defaults eventually
   * to write silopmpio and to write a single file in the future */
  /* Silo PMPIO file writing control */
  sprintf(key, "%s.WriteSiloPMPIOSubsurfData", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_subsurf_data = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOPressure", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_press = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOVelocities", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_velocities = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOSaturation", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_satur = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOEvapTrans", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_evaptrans = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOEvapTransSum", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_evaptrans_sum = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOOverlandSum", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_overland_sum = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOOverlandBCFlux", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_overland_bc_flux = switch_value;

  sprintf(key, "%s.WriteSiloPMPIODZMultiplier", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_dzmult = switch_value;

  if (public_xtra->write_silopmpio_overland_bc_flux)
  {
    InputError("Error: setting %s to %s but do not have CLM\n",
               switch_name, key);
  }

  sprintf(key, "%s.WriteSiloPMPIOConcentration", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_concen = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOMask", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_mask = switch_value;


  sprintf(key, "%s.WriteSiloPMPIOSlopes", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_slopes = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOMannings", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_mannings = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOSpecificStorage", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_specific_storage = switch_value;

  sprintf(key, "%s.WriteSiloPMPIOTop", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->write_silopmpio_top = switch_value;

  //@RMM spinup key
  sprintf(key, "%s.Spinup", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->spinup = switch_value;


  /* @RMM read evap trans as SS file before advance richards
   * for P-E spinup type runs                                  */

  sprintf(key, "%s.EvapTransFile", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->evap_trans_file = switch_value;


  sprintf(key, "%s.EvapTransFileTransient", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
  public_xtra->evap_trans_file_transient = switch_value;

  /* Nick's addition */
  sprintf(key, "%s.EvapTrans.FileLooping", name);
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  if (switch_value < 0)
  {
    InputError("Error: invalid print switch value <%s> for key <%s>\n",
               switch_name, key);
  }
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
 * SolverRichardsCoupledFreePublicXtra
 *--------------------------------------------------------------------------*/

void
SolverRichardsCoupledFreePublicXtra()
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
 * SolverRichardsCoupledSizeOfTempData
 *--------------------------------------------------------------------------*/

int
SolverRichardsCoupledSizeOfTempData()
{
  /* SGS temp data */

  return 0;
}

/*--------------------------------------------------------------------------
 * SolverRichardsCoupled
 *--------------------------------------------------------------------------*/
void
SolverRichardsCoupled()
{
  PFModule *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra *instance_xtra =
    (InstanceXtra*)PFModuleInstanceXtra(this_module);

  Problem *problem = (public_xtra->problem);

  double start_time = ProblemStartTime(problem);
  double stop_time = ProblemStopTime(problem);

  Grid *grid = (instance_xtra->grid);

  Vector *pressure_out;
  Vector *porosity_out;
  Vector *saturation_out;

  char filename[2048];

  VectorUpdateCommHandle *handle;

  /*
   * sk: Vector that contains the sink terms from the land surface model
   */
  Vector *evap_trans;

  SetupRichardsCoupled(this_module);

  /*sk Initialize LSM terms */
  evap_trans = NewVectorType(grid, 1, 1, vector_cell_centered);
  InitVectorAll(evap_trans, 0.0);

  if (public_xtra->evap_trans_file)
  {
    sprintf(filename, "%s", public_xtra->evap_trans_filename);
    //printf("%s %s \n",filename, public_xtra -> evap_trans_filename);
    ReadPFBinary(filename, evap_trans);

    handle = InitVectorUpdate(evap_trans, VectorUpdateAll);
    FinalizeVectorUpdate(handle);
  }

  AdvanceRichardsCoupled(this_module,
                  start_time,
                  stop_time,
                  NULL,
                  evap_trans, &pressure_out, &porosity_out, &saturation_out);

  /*
   * Record amount of memory in use.
   */
  recordMemoryInfo();

  TeardownRichardsCoupled(this_module);

  FreeVector(evap_trans);
}

/*
 * Getter/Setter methods
 */

ProblemData *
GetProblemDataRichardsCoupled(PFModule * this_module)
{
  InstanceXtra *instance_xtra =
    (InstanceXtra*)PFModuleInstanceXtra(this_module);

  return(instance_xtra->problem_data);
}

Problem *
GetProblemRichardsCoupled(PFModule * this_module)
{
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  return(public_xtra->problem);
}

PFModule *
GetICPhasePressureRichardsCoupled(PFModule * this_module)
{
  InstanceXtra *instance_xtra =
    (InstanceXtra*)PFModuleInstanceXtra(this_module);

  return(instance_xtra->ic_phase_pressure);
}
