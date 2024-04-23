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
  int reset_surface_pressure;   /* surface pressure flag set to True and pressures are reset per threshold keys */
  int threshold_pressure;       /* surface pressure threshold pressure */
  int reset_pressure;           /* surface pressure reset pressure */
  int evap_trans_file;          /* read evap_trans as a SS file before advance richards */
  int evap_trans_file_transient;        /* read evap_trans as a transient file before advance richards timestep */
  char *evap_trans_filename;    /* File name for evap trans */
  int evap_trans_file_looping;  /* Loop over the flux files if we run out */
  int surface_predictor;  /* key to turn on surface predictor feature RMM */
  double surface_predictor_pressure;  /* surface predictor pressure value RMM */
  int surface_predictor_print;  /* key to turn on surface predictor printing RMM */


#ifdef HAVE_CLM                 /* VARIABLES FOR CLM ONLY */
  char *clm_file_dir;           /* directory location for CLM files */
  int clm_dump_interval;        /* time interval, integer, for CLM output */
  int clm_1d_out;               /* boolean 0-1, integer, for CLM 1-d output */
  int clm_forc_veg;             /* boolean 0-1, integer, for CLM vegetation forcing option */
  /*BH*/ int clm_bin_out_dir;   /* boolean 0-1, integer, for sep dirs for each clm binary output */
  // int                clm_dump_files;     /* boolean 0-1, integer, for write CLM output from PF */

  int clm_nz;                   /* Number of CLM soil layers (layers in root zone) */
  int clm_SoiLayer;             /* NBE: Layer number for LAI seasonal variations */
  int clm_istep_start;          /* CLM time counter for met forcing (line in 1D file; name extension of 2D/3D files) */
  int clm_fstep_start;          /* CLM time counter for inside met forcing files -- used for time keeping w/in 3D met files */
  int clm_metforce;             /* CLM met forcing  -- 1=uniform (default), 2=distributed, 3=distributed w/ multiple timesteps */
  int clm_metnt;                /* CLM met forcing  -- if 3D, length of time axis in each file */
  int clm_metsub;               /* Flag for met vars in subdirs of clm_metpath or all in clm_metpath */
  char *clm_metfile;            /* File name for 1D forcing *or* base name for 2D forcing */
  char *clm_metpath;            /* Path to CLM met forcing file(s) */
  double *sw1d, *lw1d, *prcp1d, /* 1D forcing variables */
    *tas1d, *u1d, *v1d, *patm1d, *qatm1d, *lai1d, *sai1d, *z0m1d, *displa1d;    /* BH: added lai, sai, z0m, displa */

  int clm_beta_function;        /* CLM evap function for var sat 0=none, 1=linear, 2=cos */
  double clm_res_sat;           /* CLM residual saturation in soil sat units [-] */
  int clm_veg_function;         /* CLM veg function for water stress 0=none, 1=press, 2=sat */
  double clm_veg_wilting;       /* CLM veg function wilting point in meters or soil moisture */
  double clm_veg_fieldc;        /* CLM veg function field capacity in meters or soil moisture */

  int clm_irr_type;             /* CLM irrigation type flag -- 0=none, 1=Spray, 2=Drip, 3=Instant */
  int clm_irr_cycle;            /* CLM irrigation cycle flag -- 0=Constant, 1=Deficit */
  double clm_irr_rate;          /* CLM irrigation application rate [mm/s] */
  double clm_irr_start;         /* CLM irrigation schedule -- start time of constant cycle [GMT] */
  double clm_irr_stop;          /* CLM irrigation schedule -- stop time of constant cyle [GMT] */
  double clm_irr_threshold;     /* CLM irrigation schedule -- soil moisture threshold for deficit cycle */
  int clm_irr_thresholdtype;    /* Deficit-based saturation criteria (top, bottom, column avg) */

  int clm_reuse_count;          /* NBE: Number of times to use each CLM input */
  int clm_write_logs;           /* NBE: Write the processor logs for CLM or not */
  int clm_last_rst;             /* NBE: Only write/overwrite one rst file or write a lot of them */
  int clm_daily_rst;            /* NBE: Write daily RST files or hourly */
#endif

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

  Vector *evap_trans;           /* sk: Vector that contains the sink terms from the land surface model */
  Vector *evap_trans_sum;       /* running sum of evaporation and transpiration */
  Vector *overland_sum;
  Vector *ovrl_bc_flx;          /* vector containing outflow at the boundary */
  Vector *dz_mult;              /* vector containing dz multplier values for all cells */
  Vector *x_velocity;           /* vector containing x-velocity face values */
  Vector *y_velocity;           /* vector containing y-velocity face values */
  Vector *z_velocity;           /* vector containing z-velocity face values */
#ifdef HAVE_CLM
  /* RM: vars for pf printing of clm output */
  Vector *eflx_lh_tot;          /* total LH flux from canopy height to atmosphere [W/m^2] */
  Vector *eflx_lwrad_out;       /* outgoing LW radiation from ground+canopy [W/m^2] */
  Vector *eflx_sh_tot;          /* total SH flux from canopy height to atmosphere [W/m^2] */
  Vector *eflx_soil_grnd;       /* ground heat flux [W/m^2] */
  Vector *qflx_evap_tot;        /* total ET flux from canopy height to atmosphere [mm/s] */
  Vector *qflx_evap_grnd;       /* evap flux from ground (first soil layer) [mm/s] (defined equal to qflx_evap_soi) */
  Vector *qflx_evap_soi;        /* evap flux from ground [mm/s] */
  Vector *qflx_evap_veg;        /* evap+trans from leaves [mm/s] */
  Vector *qflx_tran_veg;        /* trans from veg [mm/s] */
  Vector *qflx_infl;            /* infiltration [mm/s] */
  Vector *swe_out;              /* snow water equivalent [mm] */
  Vector *t_grnd;               /* CLM soil surface temperature [K] */
  Vector *tsoil;                /* CLM soil temp, all 10 layers [K] */
  Grid *gridTs;                 /* New grid fro tsoi (nx*ny*10) */

  /* IMF: vars for printing clm irrigation output */
  Vector *irr_flag;             /* Flag for irrigating/pumping under deficit-based irrigation scheme */
  Vector *qflx_qirr;            /* Irrigation applied at surface -- spray or drip */
  Vector *qflx_qirr_inst;       /* Irrigation applied by inflating soil moisture -- "instant" */

  /* IMF: vars for distributed met focing */
  Grid *metgrid;                /* new grid for 2D or 3D met forcing vars (nx*ny*clm_metnt; clm_metnt defaults to 1) */
  Vector *sw_forc;              /* shortwave radiation forcing [W/m^2] */
  Vector *lw_forc;              /* longwave radiation forcing [W/m^2] */
  Vector *prcp_forc;            /* precipitation [mm/s] */
  Vector *tas_forc;             /* air temp [K] @ ref height (hgt set in drv_clmin.dat, currently 2m) */
  Vector *u_forc;               /* east-west wind [m/s] @ ref height (hgt set in drv_clmin.dat, currently 10m) */
  Vector *v_forc;               /* south-north wind [m/s] @ ref height (hgt set in drv_clmin.dat, currently 10m) */
  Vector *patm_forc;            /* surface air pressure [Pa] */
  Vector *qatm_forc;            /* surface air humidity [kg/kg] @ ref height (hgt set in drv_clmin.dat, currently 2m) */
  Vector *lai_forc;             /* LAI                              BH */
  Vector *sai_forc;             /* SAI                                                  BH */
  Vector *z0m_forc;             /* Aerodynamic roughness length [m] BH */
  Vector *displa_forc;          /* Displacement height [m]                  BH */
  Vector *veg_map_forc;         /* Vegetation map [classes 1-18]    BH */

  Grid *snglclm;                /* NBE: New grid for single file CLM ouptut */
  Vector *clm_out_grid;         /* NBE - Holds multi-layer, single file output of CLM */
#endif

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
  double clm_dump_index;
} InstanceXtra;

static const char* dswr_filenames[] = { "DSWR" };
static const char* dlwr_filenames[] = { "DLWR" };
static const char* apcp_filenames[] = { "APCP" };
static const char* temp_filenames[] = { "Temp" };
static const char* wind_filenames[] = { "UGRD", "VGRD" };
static const char* prss_filenames[] = { "Press" };
static const char* spfh_filenames[] = { "SPFH" };

static const char* vlai_filenames[] = { "LAI" };
static const char* vsai_filenames[] = { "SAI" };
static const char* vz0m_filenames[] = { "Z0M" };
static const char* vdsp_filenames[] = { "DISPLA" };

typedef struct _CLMForcingField {
  const char* field_name; // Name for human presentation.
  const char* field_units; // Units (if available).
  const char** component_names; // Filenames assigned to each component.
  int num_components; // Number of components.
  int vegetative; // Is this a vegetation forcing function?
} CLMForcingField;

static const CLMForcingField clmForcingFields[] = {
  { "downward shortwave radiation", NULL, dswr_filenames, sizeof(dswr_filenames) / sizeof(dswr_filenames[0]), 0 },
  { "downward longwave radiation", NULL, dlwr_filenames, sizeof(dlwr_filenames) / sizeof(dlwr_filenames[0]), 0 },
  { "precipitation", NULL, apcp_filenames, sizeof(apcp_filenames) / sizeof(apcp_filenames[0]), 0 },
  { "temperature", NULL, temp_filenames, sizeof(temp_filenames) / sizeof(temp_filenames[0]), 0 },
  { "wind velocity", NULL, wind_filenames, sizeof(wind_filenames) / sizeof(wind_filenames[0]), 0 },
  { "atmospheric pressure", NULL, prss_filenames, sizeof(prss_filenames) / sizeof(prss_filenames[0]), 0 },
  { "specific humidity", NULL, spfh_filenames, sizeof(spfh_filenames) / sizeof(spfh_filenames[0]), 0 },

  // vegetative forcing functions (optionally enabled):
  { "leaf area index", NULL, vlai_filenames, sizeof(vlai_filenames) / sizeof(vlai_filenames[0]), 1 },
  { "stem area index", NULL, vsai_filenames, sizeof(vsai_filenames) / sizeof(vsai_filenames[0]), 1 },
  { "aerodynamic roughness length", NULL, vz0m_filenames, sizeof(vz0m_filenames) / sizeof(vz0m_filenames[0]), 1 },
  { "displacement height", NULL, vdsp_filenames, sizeof(vdsp_filenames) / sizeof(vdsp_filenames[0]), 1 },
};
int numForcingFields = sizeof(clmForcingFields) / sizeof(clmForcingFields[0]);

void
SetupRichards(PFModule * this_module)
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

#ifdef HAVE_CLM
  /* IMF: for CLM met forcings (local to SetupRichards) */
  char filename[128];
  int n, nc, c;                 /*Added c BH */
  int ch;
  double sw, lw, prcp, tas, u, v, patm, qatm, lai, sai, z0m, displa;    // forcing vars added vegetation BH
  FILE *metf_temp;              // temp file for forcings
  amps_Invoice invoice;         // for distributing 1D met forcings
  amps_File metf1d;             // for distributing 1D met forcings
  Grid *metgrid = (instance_xtra->metgrid);     // grid for 2D and 3D met forcings
  Grid *gridTs = (instance_xtra->gridTs);       // grid for writing T-soil or instant irrig flux as Silo

  Grid *snglclm = (instance_xtra->snglclm);     // NBE: grid for single file CLM outputs
#endif

  t = start_time;
  dt = 0.0e0;

  NewMetadata(this_module);
  MetadataAddParflowDomainInfo(js_domains, this_module, grid);

#ifdef HAVE_CLM
  /* Add metadata for forcings */
  {
    int ff;
    for (ff = 0; ff < numForcingFields; ++ff)
    {
      if (
          !clmForcingFields[ff].vegetative ||
          (public_xtra->clm_metforce == 3 && public_xtra->clm_forc_veg == 1))
      {

        MetadataAddForcingField(
                                js_inputs,
                                clmForcingFields[ff].field_name,
                                clmForcingFields[ff].field_units,
                                "cell", "surface",
                                public_xtra->clm_metforce,
                                public_xtra->clm_metsub,
                                public_xtra->clm_metpath,
                                public_xtra->clm_metfile,
                                public_xtra->clm_istep_start,
                                public_xtra->clm_fstep_start,
                                public_xtra->clm_metnt,
                                clmForcingFields[ff].num_components,
                                clmForcingFields[ff].component_names
                                );
      }
    }
  }
#endif

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

  if(public_xtra->print_top || public_xtra->write_silo_top)
  {
    ComputePatchTop(problem, problem_data);
  }

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
    strcpy(file_postfix, "top_zindex");
    WritePFBinary(file_prefix, file_postfix, ProblemDataIndexOfDomainTop(problem_data));
    strcpy(file_postfix, "top_patch");
    WritePFBinary(file_prefix, file_postfix, ProblemDataPatchIndexOfDomainTop(problem_data));
  }

  if (public_xtra->write_silo_top)
  {
    strcpy(file_postfix, "");
    strcpy(file_type, "top_zindex");
    WriteSilo(file_prefix, file_type, file_postfix, ProblemDataIndexOfDomainTop(problem_data),
              t, 0, "TopZIndex");
    strcpy(file_type, "top_patch");
    WriteSilo(file_prefix, file_type, file_postfix, ProblemDataPatchIndexOfDomainTop(problem_data),
              t, 0, "TopPatch");
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
  instance_xtra->clm_dump_index = 1.0;

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

    /*sk Initialize LSM terms */
    instance_xtra->evap_trans = NewVectorType(grid, 1, 1, vector_cell_centered);
    InitVectorAll(instance_xtra->evap_trans, 0.0);

    if (public_xtra->evap_trans_file)
    {
      ReadPFBinary(public_xtra->evap_trans_filename, instance_xtra->evap_trans);

      handle = InitVectorUpdate(instance_xtra->evap_trans, VectorUpdateAll);
      FinalizeVectorUpdate(handle);
    }

    /* IMF: the following are only used w/ CLM */
#ifdef HAVE_CLM
    /* NBE: CLM single file output */
    if (public_xtra->single_clm_file)
    {
      instance_xtra->clm_out_grid =
        NewVectorType(snglclm, 1, 1, vector_met);
      InitVectorAll(instance_xtra->clm_out_grid, 0.0);
    }

    /*IMF Initialize variables for printing CLM output */
    instance_xtra->eflx_lh_tot =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->eflx_lh_tot, 0.0);

    instance_xtra->eflx_lwrad_out =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->eflx_lwrad_out, 0.0);

    instance_xtra->eflx_sh_tot =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->eflx_sh_tot, 0.0);

    instance_xtra->eflx_soil_grnd =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->eflx_soil_grnd, 0.0);

    instance_xtra->qflx_evap_tot =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->qflx_evap_tot, 0.0);

    instance_xtra->qflx_evap_grnd =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->qflx_evap_grnd, 0.0);

    instance_xtra->qflx_evap_soi =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->qflx_evap_soi, 0.0);

    instance_xtra->qflx_evap_veg =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->qflx_evap_veg, 0.0);

    instance_xtra->qflx_tran_veg =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->qflx_tran_veg, 0.0);

    instance_xtra->qflx_infl =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->qflx_infl, 0.0);

    instance_xtra->swe_out =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->swe_out, 0.0);

    instance_xtra->t_grnd =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->t_grnd, 0.0);

    instance_xtra->tsoil = NewVectorType(gridTs, 1, 1, vector_clm_topsoil);
    InitVectorAll(instance_xtra->tsoil, 0.0);

    /*IMF Initialize variables for CLM irrigation output */
    instance_xtra->irr_flag =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->irr_flag, 0.0);

    instance_xtra->qflx_qirr =
      NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
    InitVectorAll(instance_xtra->qflx_qirr, 0.0);

    instance_xtra->qflx_qirr_inst =
      NewVectorType(gridTs, 1, 1, vector_clm_topsoil);
    InitVectorAll(instance_xtra->qflx_qirr_inst, 0.0);

    /*IMF Initialize variables for CLM forcing fields
     * SW rad, LW rad, precip, T(air), U, V, P(air), q(air) */
    instance_xtra->sw_forc = NewVectorType(metgrid, 1, 1, vector_met);
    InitVectorAll(instance_xtra->sw_forc, 100.0);

    instance_xtra->lw_forc = NewVectorType(metgrid, 1, 1, vector_met);
    InitVectorAll(instance_xtra->lw_forc, 100.0);

    instance_xtra->prcp_forc = NewVectorType(metgrid, 1, 1, vector_met);
    InitVectorAll(instance_xtra->prcp_forc, 100.0);

    instance_xtra->tas_forc = NewVectorType(metgrid, 1, 1, vector_met);
    InitVectorAll(instance_xtra->tas_forc, 100.0);

    instance_xtra->u_forc = NewVectorType(metgrid, 1, 1, vector_met);
    InitVectorAll(instance_xtra->u_forc, 100.0);

    instance_xtra->v_forc = NewVectorType(metgrid, 1, 1, vector_met);
    InitVectorAll(instance_xtra->v_forc, 100.0);

    instance_xtra->patm_forc = NewVectorType(metgrid, 1, 1, vector_met);
    InitVectorAll(instance_xtra->patm_forc, 100.0);

    instance_xtra->qatm_forc = NewVectorType(metgrid, 1, 1, vector_met);
    InitVectorAll(instance_xtra->qatm_forc, 100.0);

    /* BH: added vegetatin vectors (LAI, SAI, z0m, DISPLA) and vegetation map) */
    instance_xtra->lai_forc = NewVectorType(metgrid, 1, 1, vector_met);
    InitVectorAll(instance_xtra->lai_forc, 100.0);

    instance_xtra->sai_forc = NewVectorType(metgrid, 1, 1, vector_met);
    InitVectorAll(instance_xtra->sai_forc, 100.0);

    instance_xtra->z0m_forc = NewVectorType(metgrid, 1, 1, vector_met);
    InitVectorAll(instance_xtra->z0m_forc, 100.0);

    instance_xtra->displa_forc = NewVectorType(metgrid, 1, 1, vector_met);
    InitVectorAll(instance_xtra->displa_forc, 100.0);

    instance_xtra->veg_map_forc = NewVectorType(metgrid, 1, 1, vector_met);
    InitVectorAll(instance_xtra->veg_map_forc, 100.0);
    /* BH: end add */

    /*IMF If 1D met forcing, read forcing vars to arrays */
    if (public_xtra->clm_metforce == 1)
    {
      // SGS Fixme This should not be here should be in init xtra.
      // Set filename for 1D forcing file
      sprintf(filename, "%s/%s", public_xtra->clm_metpath,
              public_xtra->clm_metfile);

      // Open file, count number of lines
      if ((metf_temp = fopen(filename, "r")) == NULL)
      {
        printf("Error: can't open file %s \n", filename);
        exit(1);
      }
      else
      {
        nc = 0;
        while ((ch = fgetc(metf_temp)) != EOF)
          if (ch == 10)
            nc++;
        fclose(metf_temp);
      }
      // Read 1D met file to arrays of length nc
      (public_xtra->sw1d) = ctalloc(double, nc);
      (public_xtra->lw1d) = ctalloc(double, nc);
      (public_xtra->prcp1d) = ctalloc(double, nc);
      (public_xtra->tas1d) = ctalloc(double, nc);
      (public_xtra->u1d) = ctalloc(double, nc);
      (public_xtra->v1d) = ctalloc(double, nc);
      (public_xtra->patm1d) = ctalloc(double, nc);
      (public_xtra->qatm1d) = ctalloc(double, nc);
      if ((metf1d = amps_SFopen(filename, "r")) == NULL)
      {
        amps_Printf("Error: can't open file %s \n", filename);
        exit(1);
      }
      // SGS this should be done as an array not individual elements
      invoice =
        amps_NewInvoice("%d%d%d%d%d%d%d%d", &sw, &lw, &prcp, &tas, &u,
                        &v, &patm, &qatm);
      for (n = 0; n < nc; n++)
      {
        amps_SFBCast(amps_CommWorld, metf1d, invoice);
        (public_xtra->sw1d)[n] = sw;
        (public_xtra->lw1d)[n] = lw;
        (public_xtra->prcp1d)[n] = prcp;
        (public_xtra->tas1d)[n] = tas;
        (public_xtra->u1d)[n] = u;
        (public_xtra->v1d)[n] = v;
        (public_xtra->patm1d)[n] = patm;
        (public_xtra->qatm1d)[n] = qatm;
      }
      amps_FreeInvoice(invoice);
      amps_SFclose(metf1d);

      /* BH: added the option to force vegetation or not: here LAI, SAI, Z0M, Displa and pfb vegetation maps are read */
      (public_xtra->lai1d) = ctalloc(double, nc * 18);
      (public_xtra->sai1d) = ctalloc(double, nc * 18);
      (public_xtra->z0m1d) = ctalloc(double, nc * 18);
      (public_xtra->displa1d) = ctalloc(double, nc * 18);
      if (public_xtra->clm_forc_veg == 1)
      {
        /*Reading file LAI */ /*BH*/
        sprintf(filename, "%s/%s", public_xtra->clm_metpath,
                "lai.dat");

        // Open file, count number of lines
        if ((metf_temp = fopen(filename, "r")) == NULL)
        {
          printf("Error: can't open file %s \n", filename);
          exit(1);
        }
        /*assume nc remains the same BH */
        // Read 1D met file to arrays of length nc
        if ((metf1d = amps_SFopen(filename, "r")) == NULL)
        {
          amps_Printf("Error: can't open file %s \n", filename);
          exit(1);
        }
        // SGS this should be done as an array not individual elements
        invoice = amps_NewInvoice("%d", &lai);
        for (n = 0; n < nc; n++)
        {
          for (c = 0; c < 18; c++)
          {
            amps_SFBCast(amps_CommWorld, metf1d, invoice);
            (public_xtra->lai1d)[18 * n + c] = lai;
          }
        }
        amps_FreeInvoice(invoice);
        amps_SFclose(metf1d);

        /*Reading file SAI */ /*BH*/
        sprintf(filename, "%s/%s", public_xtra->clm_metpath,
                "sai.dat");

        // Open file, count number of lines
        if ((metf_temp = fopen(filename, "r")) == NULL)
        {
          printf("Error: can't open file %s \n", filename);
          exit(1);
        }
        /*assume nc remains the same BH */
        // Read 1D met file to arrays of length nc
        if ((metf1d = amps_SFopen(filename, "r")) == NULL)
        {
          amps_Printf("Error: can't open file %s \n", filename);
          exit(1);
        }
        // SGS this should be done as an array not individual elements
        invoice = amps_NewInvoice("%d", &sai);
        for (n = 0; n < nc; n++)
        {
          for (c = 0; c < 18; c++)
          {
            amps_SFBCast(amps_CommWorld, metf1d, invoice);
            (public_xtra->sai1d)[18 * n + c] = sai;
          }
        }
        amps_FreeInvoice(invoice);
        amps_SFclose(metf1d);

        /*Reading file z0m */ /*BH*/
        /*sprintf(filename, "%s/%s", public_xtra -> clm_metpath, public_xtra -> clm_metfile); */
        sprintf(filename, "%s/%s", public_xtra->clm_metpath,
                "z0m.dat");

        // Open file, count number of lines
        if ((metf_temp = fopen(filename, "r")) == NULL)
        {
          printf("Error: can't open file %s \n", filename);
          exit(1);
        }
        /*assume nc remains the same BH */
        // Read 1D met file to arrays of length nc
        if ((metf1d = amps_SFopen(filename, "r")) == NULL)
        {
          amps_Printf("Error: can't open file %s \n", filename);
          exit(1);
        }
        // SGS this should be done as an array not individual elements
        invoice = amps_NewInvoice("%d", &z0m);
        for (n = 0; n < nc; n++)
        {
          for (c = 0; c < 18; c++)
          {
            amps_SFBCast(amps_CommWorld, metf1d, invoice);
            (public_xtra->z0m1d)[18 * n + c] = z0m;
          }
        }
        amps_FreeInvoice(invoice);
        amps_SFclose(metf1d);

        /*Reading file displa */ /*BH*/
        sprintf(filename, "%s/%s", public_xtra->clm_metpath,
                "displa.dat");

        // Open file, count number of lines
        if ((metf_temp = fopen(filename, "r")) == NULL)
        {
          printf("Error: can't open file %s \n", filename);
          exit(1);
        }
        /*assume nc remains the same BH */
        // Read 1D met file to arrays of length nc
        if ((metf1d = amps_SFopen(filename, "r")) == NULL)
        {
          amps_Printf("Error: can't open file %s \n", filename);
          exit(1);
        }
        // SGS this should be done as an array not individual elements
        invoice = amps_NewInvoice("%d", &displa);
        for (n = 0; n < nc; n++)
        {
          for (c = 0; c < 18; c++)
          {
            amps_SFBCast(amps_CommWorld, metf1d, invoice);
            (public_xtra->displa1d)[18 * n + c] = displa;
          }
        }
        amps_FreeInvoice(invoice);
        amps_SFclose(metf1d);

        /*Reading file vegetation map *//* BH */

        sprintf(filename, "%s/%s.pfb", public_xtra->clm_metpath,
                "veg_map");
        ReadPFBinary(filename, instance_xtra->veg_map_forc);
      }
      /* BH: end of reading LAI/SAI/Z0M/DISPLA/vegetation map */
    }
#endif

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
AdvanceRichards(PFModule * this_module, double start_time,      /* Starting time */
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

  if (evap_trans == NULL)
  {
    evap_trans = instance_xtra->evap_trans;
  }

#ifdef HAVE_OAS3
  Grid *grid = (instance_xtra->grid);
  Subgrid *subgrid;
  Subvector *p_sub, *s_sub, *et_sub, *m_sub, *po_sub, *dz_sub;
  double *pp, *sp, *et, *ms, *po_dat, *dz_dat;
  double sw_lat = .0;
  double sw_lon = .0;
#endif

#ifdef HAVE_CLM
  Grid *grid = (instance_xtra->grid);
  Subgrid *subgrid;
  Subvector *p_sub, *s_sub, *et_sub, *m_sub, *po_sub, *dz_sub;
  double *pp, *sp, *et, *ms, *po_dat, *dz_dat;

  /* IMF: For CLM met forcing (local to AdvanceRichards) */
  int istep;                    // IMF: counter for clm output times

  /* NBE added for clm reuse of inputs */
  int clm_next = 1;             //NBE: Counter for reuse loop
  int clm_skip = public_xtra->clm_reuse_count;  // NBE:defaults to 1
  int clm_write_logs = public_xtra->clm_write_logs;     // NBE: defaults to 1, disables log file writing if 0
  int clm_last_rst = public_xtra->clm_last_rst; // Reuse of the RST file
  int clm_daily_rst = public_xtra->clm_daily_rst;       // Daily or hourly RST files, defaults to daily

  int fstep = INT_MIN;
  int fflag, fstart, fstop;     // IMF: index w/in 3D forcing array corresponding to istep
  int n, c;                     // IMF: index vars for looping over subgrid data BH: added c
  int ind_veg;                  /*BH: temporary variable to store vegetation index */
  int Stepcount = 0;            /* Added for transient EvapTrans file management - NBE */
  int Loopcount = 0;            /* Added for transient EvapTrans file management - NBE */
  double sw=NAN, lw=NAN, prcp=NAN, tas=NAN, u=NAN, v=NAN, patm=NAN, qatm=NAN;   // IMF: 1D forcing vars (local to AdvanceRichards)
  double lai[18], sai[18], z0m[18], displa[18]; /*BH: array with lai/sai/z0m/displa values for each veg class */
  double *sw_data = NULL;
  double *lw_data = NULL;
  double *prcp_data = NULL;     // IMF: 2D forcing vars (SubvectorData) (local to AdvanceRichards)
  double *tas_data = NULL;
  double *u_data = NULL;
  double *v_data = NULL;
  double *patm_data = NULL;
  double *qatm_data = NULL;
  double *lai_data = NULL;
  /*BH*/ double *sai_data = NULL;
  /*BH*/ double *z0m_data = NULL;
  /*BH*/ double *displa_data = NULL;
  /*BH*/ double *veg_map_data = NULL;
  /*BH*/                        /*will fail if veg_map_data is declared as int */
  char filename[2048];          // IMF: 1D input file name *or* 2D/3D input file base name
  Subvector *sw_forc_sub, *lw_forc_sub, *prcp_forc_sub, *tas_forc_sub, *u_forc_sub, *v_forc_sub, *patm_forc_sub, *qatm_forc_sub, *lai_forc_sub, *sai_forc_sub, *z0m_forc_sub, *displa_forc_sub, *veg_map_forc_sub;      /*BH: added LAI/SAI/Z0M/DISPLA/vegmap */

  /* Slopes */
  Subvector *slope_x_sub, *slope_y_sub;
  double *slope_x_data, *slope_y_data;

  /* IMF: For writing CLM output */
  Subvector *eflx_lh_tot_sub, *eflx_lwrad_out_sub, *eflx_sh_tot_sub,
    *eflx_soil_grnd_sub, *qflx_evap_tot_sub, *qflx_evap_grnd_sub,
    *qflx_evap_soi_sub, *qflx_evap_veg_sub, *qflx_tran_veg_sub,
    *qflx_infl_sub, *swe_out_sub, *t_grnd_sub, *tsoil_sub, *irr_flag_sub,
    *qflx_qirr_sub, *qflx_qirr_inst_sub;

  double *eflx_lh, *eflx_lwrad, *eflx_sh, *eflx_grnd, *qflx_tot, *qflx_grnd,
    *qflx_soi, *qflx_eveg, *qflx_tveg, *qflx_in, *swe, *t_g, *t_soi, *iflag,
    *qirr, *qirr_inst;
  int clm_file_dir_length;

  double print_cdt;
  int clm_dump_files = 0;
  int rank = amps_Rank(amps_CommWorld);
#endif

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
#ifdef HAVE_OAS3
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
#endif // end to HAVE_OAS3 CALL

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

#ifdef HAVE_CLM
  istep = public_xtra->clm_istep_start; // IMF: initialize time counter for CLM
  fflag = 0;                    // IMF: flag tripped when first met file is read
  fstart = 0;                   // init to something, only used with 3D met forcing
  fstop = 0;                    // init to something, only used with 3D met forcing
#endif

  do                            /* while take_more_time_steps */
  {
    if (t == ct)
    {
      ct += cdt;

      //CPS oasis exchange
#ifdef HAVE_OAS3
      ForSubgridI(is, GridSubgrids(grid))
      {
        int ix, iy, nx, ny, nz, nx_f, ny_f;

        subgrid = GridSubgrid(grid, is);


        p_sub = VectorSubvector(instance_xtra->pressure, is);
        s_sub = VectorSubvector(instance_xtra->saturation, is);
        et_sub = VectorSubvector(evap_trans, is);
        m_sub = VectorSubvector(instance_xtra->mask, is);
        po_sub = VectorSubvector(porosity, is);
        dz_sub = VectorSubvector(instance_xtra->dz_mult, is);

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
        po_dat = SubvectorData(po_sub);
        dz_dat = SubvectorData(dz_sub);
        //CPS       amps_Printf("Calling oasis send/receive for time  %3.1f \n", t);
        CALL_send_fld2_clm(pp, sp, ms, ix, iy, nx, ny, nz, nx_f, ny_f,
                           t,po_dat,dz_dat);
        amps_Sync(amps_CommWorld);
        CALL_receive_fld2_clm(et, ms, ix, iy, nx, ny, nz, nx_f, ny_f, t);
      }
      amps_Sync(amps_CommWorld);
      handle = InitVectorUpdate(evap_trans, VectorUpdateAll);
      FinalizeVectorUpdate(handle);
#endif // end to HAVE_OAS3 CALL

      // IMF: Added to include CLM dumps in file_number updating.
      //      Init to zero outside of ifdef HAVE_CLM
      clm_file_dumped = 0;

      /* IMF: The following are only used w/ CLM */
#ifdef HAVE_CLM
      BeginTiming(CLMTimingIndex);

      /* sk: call to the land surface model/subroutine */
      /* sk: For the couple with CLM */
      int p = GlobalsP;
      int q = GlobalsQ;
      int r = GlobalsR;
      /* @RMM get grid from global (assuming this is comp grid) to pass to CLM */
      int gnx = BackgroundNX(GlobalsBackground);
      int gny = BackgroundNY(GlobalsBackground);
      // printf("global nx, ny: %d %d \n", gnx, gny);
      int is;

      // NBE: setting up a way to reuse CLM inputs for multiple time steps
      if (clm_next == 1)
      {
        /* IMF: If 1D met forcing */
        if (public_xtra->clm_metforce == 1)
        {
          // Read forcing values for correct timestep
          sw = (public_xtra->sw1d)[istep - 1];
          lw = (public_xtra->lw1d)[istep - 1];
          prcp = (public_xtra->prcp1d)[istep - 1];
          tas = (public_xtra->tas1d)[istep - 1];
          u = (public_xtra->u1d)[istep - 1];
          v = (public_xtra->v1d)[istep - 1];
          patm = (public_xtra->patm1d)[istep - 1];
          qatm = (public_xtra->qatm1d)[istep - 1];

          /*BH: populating vegetation vectors */
          for (c = 0; c < 18; c++)
          {
            lai[c] = (public_xtra->lai1d)[(istep - 1) * 18 + c];
            /*printf("LAI by class: class %d: value %f\n",c,lai[c]); */
            sai[c] = (public_xtra->sai1d)[(istep - 1) * 18 + c];
            z0m[c] = (public_xtra->z0m1d)[(istep - 1) * 18 + c];
            displa[c] =
              (public_xtra->displa1d)[(istep - 1) * 18 + c];
          }

          /*BH: end populating vegetation vectors */
        }                       //end if (clm_metforce==1)
        else
        {
          // Initialize unused variables to something
          sw = 0.0;
          lw = 0.0;
          prcp = 0.0;
          tas = 0.0;
          u = 0.0;
          v = 0.0;
          patm = 0.0;
          qatm = 0.0;
        }

        /* IMF: If 2D met forcing...read input files @ each timestep... */
        if (public_xtra->clm_metforce == 2)
        {
          // Subdirectories for each variable?
          if (public_xtra->clm_metsub)
          {
            sprintf(filename, "%s/%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath, "DSWR",
                    public_xtra->clm_metfile, "DSWR", istep);
            ReadPFBinary(filename, instance_xtra->sw_forc);
            sprintf(filename, "%s/%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath, "DLWR",
                    public_xtra->clm_metfile, "DLWR", istep);
            ReadPFBinary(filename, instance_xtra->lw_forc);
            sprintf(filename, "%s/%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath, "APCP",
                    public_xtra->clm_metfile, "APCP", istep);
            ReadPFBinary(filename, instance_xtra->prcp_forc);
            sprintf(filename, "%s/%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath, "Temp",
                    public_xtra->clm_metfile, "Temp", istep);
            ReadPFBinary(filename, instance_xtra->tas_forc);
            sprintf(filename, "%s/%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath, "UGRD",
                    public_xtra->clm_metfile, "UGRD", istep);
            ReadPFBinary(filename, instance_xtra->u_forc);
            sprintf(filename, "%s/%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath, "VGRD",
                    public_xtra->clm_metfile, "VGRD", istep);
            ReadPFBinary(filename, instance_xtra->v_forc);
            sprintf(filename, "%s/%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath, "Press",
                    public_xtra->clm_metfile, "Press", istep);
            ReadPFBinary(filename, instance_xtra->patm_forc);
            sprintf(filename, "%s/%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath, "SPFH",
                    public_xtra->clm_metfile, "SPFH", istep);
            ReadPFBinary(filename, instance_xtra->qatm_forc);
          }
          else
          {
            sprintf(filename, "%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath,
                    public_xtra->clm_metfile, "DSWR", istep);
            ReadPFBinary(filename, instance_xtra->sw_forc);
            sprintf(filename, "%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath,
                    public_xtra->clm_metfile, "DLWR", istep);
            ReadPFBinary(filename, instance_xtra->lw_forc);
            sprintf(filename, "%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath,
                    public_xtra->clm_metfile, "APCP", istep);
            ReadPFBinary(filename, instance_xtra->prcp_forc);
            sprintf(filename, "%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath,
                    public_xtra->clm_metfile, "Temp", istep);
            ReadPFBinary(filename, instance_xtra->tas_forc);
            sprintf(filename, "%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath,
                    public_xtra->clm_metfile, "UGRD", istep);
            ReadPFBinary(filename, instance_xtra->u_forc);
            sprintf(filename, "%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath,
                    public_xtra->clm_metfile, "VGRD", istep);
            ReadPFBinary(filename, instance_xtra->v_forc);
            sprintf(filename, "%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath,
                    public_xtra->clm_metfile, "Press", istep);
            ReadPFBinary(filename, instance_xtra->patm_forc);
            sprintf(filename, "%s/%s.%s.%06d.pfb",
                    public_xtra->clm_metpath,
                    public_xtra->clm_metfile, "SPFH", istep);
            ReadPFBinary(filename, instance_xtra->qatm_forc);
          }                     //end if/else (clm_metsub==True)
        }                       //end if (clm_metforce==2)

        /* IMF: If 3D met forcing... */
        if (public_xtra->clm_metforce == 3)
        {
          // Calculate z-index in forcing vars corresponding to istep
          fstep = ((istep - 1) % public_xtra->clm_metnt);               // index w/in met vars corresponding to istep

          // Read input files... *IF* istep is a multiple of clm_metnt
          //                     *OR* file hasn't been read yet (fflag==0)
          if (fstep == 0 || fflag == 0)
          {
            //Figure out which file to read (i.e., calculate correct file time-stamps)
            if (fflag == 0)
            {
              fflag = 1;
              fstart = (public_xtra->clm_istep_start) - fstep;                  // first time value in 3D met file names
              fstop = fstart - 1 + public_xtra->clm_metnt;              // second time value in 3D met file names
            }
            else
            {
              fstart = istep;                   // forst time value in 3D met file names
              fstop = fstart - 1 + public_xtra->clm_metnt;              // second value in 3D met file names
            }                   // end if fflag==0

            // Subdirectories for each variable?
            if (public_xtra->clm_metsub)
            {
              sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath, "DSWR",
                      public_xtra->clm_metfile, "DSWR", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->sw_forc);

              sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath, "DLWR",
                      public_xtra->clm_metfile, "DLWR", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->lw_forc);

              sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath, "APCP",
                      public_xtra->clm_metfile, "APCP", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->prcp_forc);

              sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath, "Temp",
                      public_xtra->clm_metfile, "Temp", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->tas_forc);

              sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath, "UGRD",
                      public_xtra->clm_metfile, "UGRD", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->u_forc);

              sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath, "VGRD",
                      public_xtra->clm_metfile, "VGRD", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->v_forc);

              sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath, "Press",
                      public_xtra->clm_metfile, "Press", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->patm_forc);

              sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath, "SPFH",
                      public_xtra->clm_metfile, "SPFH", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->qatm_forc);

              /*BH: added the option to force vegetation or not */
              if (public_xtra->clm_forc_veg == 1)
              {
                sprintf(filename,
                        "%s/%s/%s.%s.%06d_to_%06d.pfb",
                        public_xtra->clm_metpath, "LAI",
                        public_xtra->clm_metfile, "LAI",
                        fstart, fstop);
                ReadPFBinary(filename,
                             instance_xtra->lai_forc);

                sprintf(filename,
                        "%s/%s/%s.%s.%06d_to_%06d.pfb",
                        public_xtra->clm_metpath, "SAI",
                        public_xtra->clm_metfile, "SAI",
                        fstart, fstop);
                ReadPFBinary(filename,
                             instance_xtra->sai_forc);

                sprintf(filename,
                        "%s/%s/%s.%s.%06d_to_%06d.pfb",
                        public_xtra->clm_metpath, "Z0M",
                        public_xtra->clm_metfile, "Z0M",
                        fstart, fstop);
                ReadPFBinary(filename,
                             instance_xtra->z0m_forc);

                sprintf(filename,
                        "%s/%s/%s.%s.%06d_to_%06d.pfb",
                        public_xtra->clm_metpath, "DISPLA",
                        public_xtra->clm_metfile, "DISPLA",
                        fstart, fstop);
                ReadPFBinary(filename,
                             instance_xtra->displa_forc);
              }
              /*BH: end added the option to force vegetation or not */
            }
            else
            {
              sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath,
                      public_xtra->clm_metfile, "DSWR", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->sw_forc);

              sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath,
                      public_xtra->clm_metfile, "DLWR", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->lw_forc);

              sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath,
                      public_xtra->clm_metfile, "APCP", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->prcp_forc);

              sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath,
                      public_xtra->clm_metfile, "Temp", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->tas_forc);

              sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath,
                      public_xtra->clm_metfile, "UGRD", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->u_forc);

              sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath,
                      public_xtra->clm_metfile, "VGRD", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->v_forc);

              sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath,
                      public_xtra->clm_metfile, "Press", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->patm_forc);

              sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb",
                      public_xtra->clm_metpath,
                      public_xtra->clm_metfile, "SPFH", fstart,
                      fstop);
              ReadPFBinary(filename, instance_xtra->qatm_forc);

              /*BH: added the option to force vegetation or not */
              if (public_xtra->clm_forc_veg == 1)
              {
                sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb",
                        public_xtra->clm_metpath,
                        public_xtra->clm_metfile, "LAI",
                        fstart, fstop);
                ReadPFBinary(filename,
                             instance_xtra->lai_forc);

                sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb",
                        public_xtra->clm_metpath,
                        public_xtra->clm_metfile, "SAI",
                        fstart, fstop);
                ReadPFBinary(filename,
                             instance_xtra->sai_forc);

                sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb",
                        public_xtra->clm_metpath,
                        public_xtra->clm_metfile, "Z0M",
                        fstart, fstop);
                ReadPFBinary(filename,
                             instance_xtra->z0m_forc);

                sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb",
                        public_xtra->clm_metpath,
                        public_xtra->clm_metfile, "DISPLA",
                        fstart, fstop);
                ReadPFBinary(filename,
                             instance_xtra->displa_forc);
              }
              /*BH: end added the option to force vegetation or not */
            }                   // end if/else clm_metsub==False
          }                     //end if (fstep==0)
        }                       //end if (clm_metforce==3)

        /* KKu Added NetCDF based forcing option. Treated similar to 2D binary files where
         * at every time step forcing data is read. */
        if (public_xtra->clm_metforce == 4)
        {
          /* KKu Since NetCDF indices start at 0, istep-1 is supplied to read
           * for the given time step*/
          sprintf(filename, "%s", public_xtra->clm_metfile);
          ReadPFNC(filename, instance_xtra->sw_forc, "DSWR",
                   istep - 1, 2);
          ReadPFNC(filename, instance_xtra->lw_forc, "DLWR",
                   istep - 1, 2);
          ReadPFNC(filename, instance_xtra->prcp_forc, "APCP",
                   istep - 1, 2);
          ReadPFNC(filename, instance_xtra->tas_forc, "Temp",
                   istep - 1, 2);
          ReadPFNC(filename, instance_xtra->u_forc, "UGRD",
                   istep - 1, 2);
          ReadPFNC(filename, instance_xtra->v_forc, "VGRD",
                   istep - 1, 2);
          ReadPFNC(filename, instance_xtra->patm_forc, "Press",
                   istep - 1, 2);
          ReadPFNC(filename, instance_xtra->qatm_forc, "SPFH",
                   istep - 1, 2);
        }
      }                         /* NBE - End of clm_reuse_count block */

      // Update metadata to indicate forcing time-data present.
      if (public_xtra->clm_metforce >= 2 && public_xtra->clm_metforce <= 3)
      {
        int ff;
        for (ff = 0; ff < numForcingFields; ++ff)
        {
          if (
              !clmForcingFields[ff].vegetative ||
              (public_xtra->clm_metforce == 3 && public_xtra->clm_forc_veg == 1))
          {
            MetadataUpdateForcingField(js_inputs, clmForcingFields[ff].field_name, istep);
          }
        }
      }



      ForSubgridI(is, GridSubgrids(grid))
      {
        double dx, dy, dz;
        int nx, ny, nz, nx_f, ny_f, nz_f, nz_rz, ip, ix, iy, iz;
        int soi_z;
        int x, y, z;

        subgrid = GridSubgrid(grid, is);
        p_sub = VectorSubvector(instance_xtra->pressure, is);
        s_sub = VectorSubvector(instance_xtra->saturation, is);
        et_sub = VectorSubvector(evap_trans, is);
        m_sub = VectorSubvector(instance_xtra->mask, is);
        po_sub = VectorSubvector(porosity, is);
        dz_sub = VectorSubvector(instance_xtra->dz_mult, is);

        /* IMF: Subvectors -- CLM surface fluxes, SWE, t_grnd */
        eflx_lh_tot_sub =
          VectorSubvector(instance_xtra->eflx_lh_tot, is);
        eflx_lwrad_out_sub =
          VectorSubvector(instance_xtra->eflx_lwrad_out, is);
        eflx_sh_tot_sub =
          VectorSubvector(instance_xtra->eflx_sh_tot, is);
        eflx_soil_grnd_sub =
          VectorSubvector(instance_xtra->eflx_soil_grnd, is);
        qflx_evap_tot_sub =
          VectorSubvector(instance_xtra->qflx_evap_tot, is);
        qflx_evap_grnd_sub =
          VectorSubvector(instance_xtra->qflx_evap_grnd, is);
        qflx_evap_soi_sub =
          VectorSubvector(instance_xtra->qflx_evap_soi, is);
        qflx_evap_veg_sub =
          VectorSubvector(instance_xtra->qflx_evap_veg, is);
        qflx_tran_veg_sub =
          VectorSubvector(instance_xtra->qflx_tran_veg, is);
        qflx_infl_sub = VectorSubvector(instance_xtra->qflx_infl, is);
        swe_out_sub = VectorSubvector(instance_xtra->swe_out, is);
        t_grnd_sub = VectorSubvector(instance_xtra->t_grnd, is);
        tsoil_sub = VectorSubvector(instance_xtra->tsoil, is);
        irr_flag_sub = VectorSubvector(instance_xtra->irr_flag, is);
        qflx_qirr_sub = VectorSubvector(instance_xtra->qflx_qirr, is);
        qflx_qirr_inst_sub =
          VectorSubvector(instance_xtra->qflx_qirr_inst, is);

        /* IMF: Subvectors -- CLM met forcings */
        sw_forc_sub = VectorSubvector(instance_xtra->sw_forc, is);
        lw_forc_sub = VectorSubvector(instance_xtra->lw_forc, is);
        prcp_forc_sub = VectorSubvector(instance_xtra->prcp_forc, is);
        tas_forc_sub = VectorSubvector(instance_xtra->tas_forc, is);
        u_forc_sub = VectorSubvector(instance_xtra->u_forc, is);
        v_forc_sub = VectorSubvector(instance_xtra->v_forc, is);
        patm_forc_sub = VectorSubvector(instance_xtra->patm_forc, is);
        qatm_forc_sub = VectorSubvector(instance_xtra->qatm_forc, is);
        /*BH: added LAI/SAI/Z0M/DISPLA/VEGMAP for vegetation forcing */
        lai_forc_sub = VectorSubvector(instance_xtra->lai_forc, is);
        sai_forc_sub = VectorSubvector(instance_xtra->sai_forc, is);
        z0m_forc_sub = VectorSubvector(instance_xtra->z0m_forc, is);
        displa_forc_sub =
          VectorSubvector(instance_xtra->displa_forc, is);
        veg_map_forc_sub =
          VectorSubvector(instance_xtra->veg_map_forc, is);

        /* Slope */
        slope_x_sub = VectorSubvector(ProblemDataTSlopeX(problem_data), is);
        slope_y_sub = VectorSubvector(ProblemDataTSlopeY(problem_data), is);
        slope_x_data = SubvectorData(slope_x_sub);
        slope_y_data = SubvectorData(slope_y_sub);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        dx = SubgridDX(subgrid);
        dy = SubgridDY(subgrid);
        dz = SubgridDZ(subgrid);

        nx_f = SubvectorNX(et_sub);
        ny_f = SubvectorNY(et_sub);
        nz_f = SubvectorNZ(et_sub);
        nz_rz = public_xtra->clm_nz;
        soi_z = public_xtra->clm_SoiLayer;

        sp = SubvectorData(s_sub);
        pp = SubvectorData(p_sub);
        et = SubvectorData(et_sub);
        ms = SubvectorData(m_sub);
        po_dat = SubvectorData(po_sub);
        dz_dat = SubvectorData(dz_sub);

        /* IMF: Subvector Data -- CLM surface fluxes, SWE, t_grnd */
        eflx_lh = SubvectorData(eflx_lh_tot_sub);
        eflx_lwrad = SubvectorData(eflx_lwrad_out_sub);
        eflx_sh = SubvectorData(eflx_sh_tot_sub);
        eflx_grnd = SubvectorData(eflx_soil_grnd_sub);
        qflx_tot = SubvectorData(qflx_evap_tot_sub);
        qflx_grnd = SubvectorData(qflx_evap_grnd_sub);
        qflx_soi = SubvectorData(qflx_evap_soi_sub);
        qflx_eveg = SubvectorData(qflx_evap_veg_sub);
        qflx_tveg = SubvectorData(qflx_tran_veg_sub);
        qflx_in = SubvectorData(qflx_infl_sub);
        swe = SubvectorData(swe_out_sub);
        t_g = SubvectorData(t_grnd_sub);
        t_soi = SubvectorData(tsoil_sub);
        iflag = SubvectorData(irr_flag_sub);
        qirr = SubvectorData(qflx_qirr_sub);
        qirr_inst = SubvectorData(qflx_qirr_inst_sub);

        /* IMF: Subvector Data -- CLM met forcings */
        // 1D Case...
        if (public_xtra->clm_metforce == 1)
        {
          // Grab SubvectorData's (still have init value)
          sw_data = SubvectorData(sw_forc_sub);
          lw_data = SubvectorData(lw_forc_sub);
          prcp_data = SubvectorData(prcp_forc_sub);
          tas_data = SubvectorData(tas_forc_sub);
          u_data = SubvectorData(u_forc_sub);
          v_data = SubvectorData(v_forc_sub);
          patm_data = SubvectorData(patm_forc_sub);
          qatm_data = SubvectorData(qatm_forc_sub);

          /* BH: added LAI/SAI/Z0M/DISPLA/VEGMAP for vegetation forcing */
          lai_data = SubvectorData(lai_forc_sub);
          sai_data = SubvectorData(sai_forc_sub);
          z0m_data = SubvectorData(z0m_forc_sub);
          displa_data = SubvectorData(displa_forc_sub);
          veg_map_data = SubvectorData(veg_map_forc_sub);

          // Fill SubvectorData's w/ uniform forcinv values
          for (n = 0; n < ((nx + 2) * (ny + 2) * 3); n++)
          {
            sw_data[n] = sw;
            lw_data[n] = lw;
            prcp_data[n] = prcp;
            tas_data[n] = tas;
            u_data[n] = u;
            v_data[n] = v;
            patm_data[n] = patm;
            qatm_data[n] = qatm;

            /* BH: added LAI/SAI/Z0M/DISPLA/VEGMAP for vegetation forcing */
            ind_veg = veg_map_data[n];
            /*printf("current index:%d \n",ind_veg); */
            lai_data[n] = lai[ind_veg - 1];
            /*printf("lai of current index:%f \n",lai[ind_veg-1]); */
            sai_data[n] = sai[ind_veg - 1];
            z0m_data[n] = z0m[ind_veg - 1];
            displa_data[n] = displa[ind_veg - 1];
          }
        }
        // 2D Case...
        if (public_xtra->clm_metforce == 2)
        {
          // Just need to grab SubvectorData's
          sw_data = SubvectorData(sw_forc_sub);
          lw_data = SubvectorData(lw_forc_sub);
          prcp_data = SubvectorData(prcp_forc_sub);
          tas_data = SubvectorData(tas_forc_sub);
          u_data = SubvectorData(u_forc_sub);
          v_data = SubvectorData(v_forc_sub);
          patm_data = SubvectorData(patm_forc_sub);
          qatm_data = SubvectorData(qatm_forc_sub);
        }
        // 3D Case...
        if (public_xtra->clm_metforce == 3)
        {
          // Determine bounds of correct time slice
          x = SubvectorIX(sw_forc_sub);
          y = SubvectorIY(sw_forc_sub);
          z = fstep - 1;
          // Extract SubvectorElt
          // (Array size is correct -- includes ghost nodes
          //  OK because ghost values not used by CLM)
          sw_data = SubvectorElt(sw_forc_sub, x, y, z);
          lw_data = SubvectorElt(lw_forc_sub, x, y, z);
          prcp_data = SubvectorElt(prcp_forc_sub, x, y, z);
          tas_data = SubvectorElt(tas_forc_sub, x, y, z);
          u_data = SubvectorElt(u_forc_sub, x, y, z);
          v_data = SubvectorElt(v_forc_sub, x, y, z);
          patm_data = SubvectorElt(patm_forc_sub, x, y, z);
          qatm_data = SubvectorElt(qatm_forc_sub, x, y, z);

          /* BH: added LAI/SAI/Z0M/DISPLA/VEGMAP for vegetation forcing */
          lai_data = SubvectorElt(lai_forc_sub, x, y, z);
          sai_data = SubvectorElt(sai_forc_sub, x, y, z);
          z0m_data = SubvectorElt(z0m_forc_sub, x, y, z);
          displa_data = SubvectorElt(displa_forc_sub, x, y, z);
        }
        /* KKu NetCDF case similar to 2D Case */
        if (public_xtra->clm_metforce == 4)
        {
          // Just need to grab SubvectorData's
          sw_data = SubvectorData(sw_forc_sub);
          lw_data = SubvectorData(lw_forc_sub);
          prcp_data = SubvectorData(prcp_forc_sub);
          tas_data = SubvectorData(tas_forc_sub);
          u_data = SubvectorData(u_forc_sub);
          v_data = SubvectorData(v_forc_sub);
          patm_data = SubvectorData(patm_forc_sub);
          qatm_data = SubvectorData(qatm_forc_sub);
        }

        ip = SubvectorEltIndex(p_sub, ix, iy, iz);
        switch (public_xtra->lsm)
        {
          case 0:
          {
            // No LSM
            break;
          }

          case 1:
          {
            /*BH: added vegetation forcings and associated option (clm_forc_veg) */
            clm_file_dir_length = strlen(public_xtra->clm_file_dir);
            CALL_CLM_LSM(pp, sp, et, ms, po_dat, dz_dat, istep, cdt, t,
                         start_time, dx, dy, dz, ix, iy, nx, ny, nz,
                         nx_f, ny_f, nz_f, nz_rz, ip, p, q, r, gnx,
                         gny, rank, sw_data, lw_data, prcp_data,
                         tas_data, u_data, v_data, patm_data,
                         qatm_data, lai_data, sai_data, z0m_data, displa_data,
                         slope_x_data, slope_y_data,
                         eflx_lh, eflx_lwrad, eflx_sh,
                         eflx_grnd, qflx_tot, qflx_grnd, qflx_soi,
                         qflx_eveg, qflx_tveg, qflx_in, swe, t_g,
                         t_soi, public_xtra->clm_dump_interval,
                         public_xtra->clm_1d_out,
                         public_xtra->clm_forc_veg,
                         public_xtra->clm_file_dir,
                         clm_file_dir_length,
                         public_xtra->clm_bin_out_dir,
                         public_xtra->write_CLM_binary,
                         public_xtra->slope_accounting_CLM,
                         public_xtra->clm_beta_function,
                         public_xtra->clm_veg_function,
                         public_xtra->clm_veg_wilting,
                         public_xtra->clm_veg_fieldc,
                         public_xtra->clm_res_sat,
                         public_xtra->clm_irr_type,
                         public_xtra->clm_irr_cycle,
                         public_xtra->clm_irr_rate,
                         public_xtra->clm_irr_start,
                         public_xtra->clm_irr_stop,
                         public_xtra->clm_irr_threshold, qirr,
                         qirr_inst, iflag,
                         public_xtra->clm_irr_thresholdtype, soi_z,
                         clm_next, clm_write_logs, clm_last_rst,
                         clm_daily_rst,
                         public_xtra->clm_nz,
                         public_xtra->clm_nz);

            break;
          }

          default:
          {
            amps_Printf("Calling unknown LSM model");
          }
        }                       /* switch on LSM */
      }


      handle = InitVectorUpdate(evap_trans, VectorUpdateAll);
      FinalizeVectorUpdate(handle);




      //#endif   //End of call to CLM

      /******************************************/
      /*    read transient evap trans flux file */
      /******************************************/
      if (public_xtra->nc_evap_trans_file_transient)
      {
        strcpy(filename, public_xtra->nc_evap_trans_filename);
        /*KKu: evaptrans is the name of the variable expected in NetCDF file */
        /*Here looping similar to pfb is not implemented. All steps are assumed to be
         * present in the single NetCDF file*/
        ReadPFNC(filename, evap_trans, "evaptrans", istep - 1, 3);
        handle = InitVectorUpdate(evap_trans, VectorUpdateAll);
        FinalizeVectorUpdate(handle);
      }
      else if (public_xtra->evap_trans_file_transient)
      {
        sprintf(filename, "%s.%05d.pfb",
                public_xtra->evap_trans_filename, (istep - 1));
        printf("%d %s %s \n", istep, filename,
               public_xtra->evap_trans_filename);

        /* Added flag to give the option to loop back over the flux files
         * This means a file doesn't have to exist for each time step - NBE */
        if (public_xtra->evap_trans_file_looping)
        {
          if (access(filename, 0) != -1)
          {
            // file exists
            Stepcount += 1;
          }
          else
          {
            if (Loopcount > Stepcount)
            {
              Loopcount = 0;
            }
            sprintf(filename, "%s.%05d.pfb",
                    public_xtra->evap_trans_filename, Loopcount);
            //printf("Using flux file %s \n",filename);
            Loopcount += 1;
          }
        }                       // NBE
        printf("%d %s %s \n", istep, filename,
               public_xtra->evap_trans_filename);

        ReadPFBinary(filename, evap_trans);

        //printf("Checking time step logging, steps = %i\n",Stepcount);

        handle = InitVectorUpdate(evap_trans, VectorUpdateAll);
        FinalizeVectorUpdate(handle);
      }


      /* NBE counter for reusing CLM input files */
      clm_next += 1;
      if (clm_next > clm_skip)
      {
        istep = istep + 1;
        clm_next = 1;
      }                         // NBE

      //istep  = istep + 1;

      EndTiming(CLMTimingIndex);


      /* =============================================================
       *  NBE: It looks like the time step isn't really scaling the CLM
       *  inputs, but the looping flag is working as intended as
       *  of 2014-04-06.
       *
       *  It is using the different time step counter BUT then it
       *  isn't scaling the inputs properly.
       *  ============================================================= */
#endif
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

#ifdef HAVE_OAS3
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
#endif

#ifdef HAVE_CLM
      /*
       * Force timestep to LSM model if we are trying to advance beyond
       * LSM timesteping.
       */
      switch (public_xtra->lsm)
      {
        case 0:
        {
          // No LSM
          break;
        }

        case 1:
        {
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
                  ("Time increment is too small; CLM wants a small timestep\n");
              }
            }
          }
          break;
        }

        default:
        {
          amps_Printf("Calling unknown LSM model");
        }
      }

      //#endif

      /* RMM added fix to adjust evap_trans for time step */
      if (public_xtra->evap_trans_file_transient)
      {
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
                ("Time increment is too small; CLM wants a small timestep\n");
            }
          }
        }
        //  break;
      }
#endif
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

#ifdef HAVE_CLM
      // Print CLM output?
      // (parallel setup to PF, but without resetting dt == print_dt for very small dt)
      clm_dump_files = 0;
      if (public_xtra->clm_dump_interval > 0)
      {
        print_cdt =
          ProblemStartTime(problem) +
          instance_xtra->clm_dump_index *
          public_xtra->clm_dump_interval - t;
        if ((dt + TIME_EPSILON) > print_cdt)
        {
          clm_dump_files = 1;
        }
      }
      else if (public_xtra->clm_dump_interval < 0)
      {
        if ((instance_xtra->iteration_number %
             (-(int)public_xtra->clm_dump_interval)) == 0)
        {
          clm_dump_files = 1;
        }
      }
      else
      {
        clm_dump_files = 0;
      }
#endif

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

      /*  experiment with a predictor to adjust land surface pressures to be >0 if rainfall*/
      if (public_xtra->surface_predictor == 1)
      {
        GrGeomSolid *gr_domain = ProblemDataGrDomain(problem_data);

        int i, j, k, r, is;
        int ix, iy, iz;
        int nx, ny, nz;
        int ip;
        double dx, dy, dz;
        double vol, vol_max, flux_in, press_pred, flux_darcy;

        Subvector *p_sub, *s_sub, *et_sub, *po_sub, *dz_sub, *vz_sub, *vx_sub, *vy_sub;
        double *pp, *sp, *et, *po_dat, *dz_dat, *vz, *vx, *vy;

        Subgrid *subgrid;
        Grid *grid = VectorGrid(evap_trans_sum);

        ForSubgridI(is, GridSubgrids(grid))
        {
          subgrid = GridSubgrid(grid, is);
          p_sub = VectorSubvector(instance_xtra->pressure, is);
          et_sub = VectorSubvector(evap_trans, is);
          dz_sub = VectorSubvector(instance_xtra->dz_mult, is);
          s_sub = VectorSubvector(instance_xtra->saturation, is);
          po_sub = VectorSubvector(porosity, is);
          vx_sub = VectorSubvector(instance_xtra->x_velocity, is);
          vy_sub = VectorSubvector(instance_xtra->y_velocity, is);
          vz_sub = VectorSubvector(instance_xtra->z_velocity, is);

          r = SubgridRX(subgrid);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          dx = SubgridDX(subgrid);
          dy = SubgridDY(subgrid);
          dz = SubgridDZ(subgrid);

          pp = SubvectorData(p_sub);
          et = SubvectorData(et_sub);
          dz_dat = SubvectorData(dz_sub);
          po_dat = SubvectorData(po_sub);
          sp = SubvectorData(s_sub);

	  vx = SubvectorData(vx_sub);
	  vy = SubvectorData(vy_sub);
	  vz = SubvectorData(vz_sub);

          
          GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
          {
            ip = SubvectorEltIndex(p_sub, i, j, k);
	    int vxi = SubvectorEltIndex(vx_sub, i + 1, j, k);
	    int vyi = SubvectorEltIndex(vy_sub, i, j + 1, k);
	    int vxi_im1 = SubvectorEltIndex(vx_sub, i , j, k);
	    int vyi_jm1 = SubvectorEltIndex(vy_sub, i, j , k);
	    int vzi_km1 = SubvectorEltIndex(vz_sub, i, j, k );
	    
	    int vxi_p1 = SubvectorEltIndex(vx_sub, i + 1, j, k+1);
	    int vyi_p1 = SubvectorEltIndex(vy_sub, i, j + 1, k+1);
	    int vxi_im1_p1 = SubvectorEltIndex(vx_sub, i , j, k+1);
	    int vyi_jm1_p1 = SubvectorEltIndex(vy_sub, i, j , k+1);
	    
            if (k == (nz - 1))
            {
              vol = dx*dy*dz*dz_dat[ip]*po_dat[ip]*sp[ip];
              flux_in = dx*dy*dz*dz_dat[ip]*et[ip]*dt;
              vol_max = dx*dy*dz*dz_dat[ip]*po_dat[ip];
              
	      flux_darcy = vz[vzi_km1]*dx*dy*dt+(-vx[vxi]+vx[vxi_im1])*dy*dz*dz_dat[ip]*dt+(-vy[vyi]+vy[vyi_jm1])*dx*dz*dz_dat[ip]*dt;
	      press_pred = ((flux_in+flux_darcy)-(vol_max - vol))/(dx*dy*po_dat[ip]);
              if ((flux_in+flux_darcy) > (vol_max - vol))
              {
                if (pp[ip] < 0.0)
		{
                  if (public_xtra->surface_predictor_pressure>0.0)
		  {
		    press_pred = public_xtra->surface_predictor_pressure;
		  }
		  
		  if (public_xtra->surface_predictor_print == 1)
		  {
		    amps_Printf("SP: Cell vol: %3.6e vol_max: %3.6e flux_in: %3.6e  Flux Darcy: %3.6e Cell Pressure: %3.6e Pred Pressure: %3.6e I: %d J: %d  Time: %12.4e  \n",vol, vol_max,flux_in, flux_darcy,pp[ip],press_pred,i,j,t);
		    amps_Printf("SP: vx_r: %3.6e vx_l: %3.6e vy_r: %3.6e vy_l: %3.6e vz_l: %3.6e  I: %d J: %d k: %d \n",vx[vxi], vx[vxi_im1], vy[vyi], vy[vyi_jm1],vz[vzi_km1],i,j,k);
		    amps_Printf("SP: vx_r: %3.6e vx_l: %3.6e vy_r: %3.6e vy_l: %3.6e    k+1 \n",vx[vxi_p1], vx[vxi_im1_p1], vy[vyi_p1], vy[vyi_jm1_p1]);
		  }
                  pp[ip] = press_pred;
                }
              }
            }
          }
        );
        }
      }

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
          if (k == (nz - 1))
          {
            if (pp_sp[ip] > 0.0)
            {
              printf(" pressure-> 0 %d %d %d %10.3f \n", i, j, k,
                     pp_sp[ip]); pp_sp[ip] = 0.0;
            }
          }
        });
      }
    }


   /***************************************************************
    *          modify land surface pressures                      *
    ***************************************************************/
   
    if (public_xtra->reset_surface_pressure == 1)
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
          if (k == (nz - 1))
          {
            if (pp_sp[ip] > public_xtra->threshold_pressure)
            {
              amps_Printf(" time: %10.3f  pressure reset: %d %d %d %10.3f \n",t, i, j, k,
                     pp_sp[ip]); pp_sp[ip] = public_xtra->reset_pressure;
            }
          }
        }
                     );
      }
    /* update pressure,  not sure if we need to do this but we might if pressures are reset along processor edges RMM */
    handle = InitVectorUpdate(instance_xtra->pressure, VectorUpdateAll);
    FinalizeVectorUpdate(handle);
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

#ifdef HAVE_CLM
    int k;

    /* Dump the fluxes, infiltration, etc. at this time-step */
    clm_file_dumped = 0;
    if (clm_dump_files)
    {
      instance_xtra->clm_dump_index++;


      if (public_xtra->write_silo_CLM)
      {
        //          /* IMF Write Met to Silo (for testing) */
        //          sprintf(file_postfix, "precip.%05d", instance_xtra -> file_number );
        //          WriteSilo( file_prefix, file_postfix, instance_xtra -> prcp_forc,
        //                     t, instance_xtra -> file_number, "Precipitation");
        //          clm_file_dumped = 1;
        //          sprintf(file_postfix, "air_temp.%05d", instance_xtra -> file_number );
        //          WriteSilo( file_prefix, file_postfix, instance_xtra -> tas_forc,
        //                     t, instance_xtra -> file_number, "AirTemperature");
        //          clm_file_dumped = 1;

        sprintf(file_postfix, "%05d", instance_xtra->file_number);
        sprintf(file_type, "eflx_lh_tot");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->eflx_lh_tot, t,
                  instance_xtra->file_number, "LatentHeat");
        clm_file_dumped = 1;

        // @RMM remove a number of output fields to limit files
        sprintf(file_type, "eflx_lwrad_out");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->eflx_lwrad_out, t,
                  instance_xtra->file_number, "LongWave");
        clm_file_dumped = 1;

        sprintf(file_type, "eflx_sh_tot");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->eflx_sh_tot, t,
                  instance_xtra->file_number, "SensibleHeat");
        clm_file_dumped = 1;

        sprintf(file_type, "eflx_soil_grnd");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->eflx_soil_grnd, t,
                  instance_xtra->file_number, "GroundHeat");
        clm_file_dumped = 1;

        sprintf(file_type, "qflx_evap_tot");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->qflx_evap_tot, t,
                  instance_xtra->file_number, "EvaporationTotal");
        clm_file_dumped = 1;

        sprintf(file_type, "qflx_evap_grnd");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->qflx_evap_grnd, t,
                  instance_xtra->file_number,
                  "EvaporationGroundNoSublimation");
        clm_file_dumped = 1;

        sprintf(file_type, "qflx_evap_soi");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->qflx_evap_soi, t,
                  instance_xtra->file_number, "EvaporationGround");
        clm_file_dumped = 1;

        sprintf(file_type, "qflx_evap_veg");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->qflx_evap_veg, t,
                  instance_xtra->file_number, "EvaporationCanopy");
        clm_file_dumped = 1;

        sprintf(file_type, "qflx_tran_veg");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->qflx_tran_veg, t,
                  instance_xtra->file_number, "Transpiration");
        clm_file_dumped = 1;

        sprintf(file_type, "qflx_infl");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->qflx_infl, t,
                  instance_xtra->file_number, "Infiltration");
        clm_file_dumped = 1;

        sprintf(file_type, "swe_out");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->swe_out, t,
                  instance_xtra->file_number, "SWE");
        clm_file_dumped = 1;

        sprintf(file_type, "t_grnd");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->t_grnd, t, instance_xtra->file_number,
                  "TemperatureGround");
        clm_file_dumped = 1;

        sprintf(file_type, "t_soil");
        WriteSilo(file_prefix, file_type, file_postfix,
                  instance_xtra->tsoil, t, instance_xtra->file_number,
                  "TemperatureSoil");
        clm_file_dumped = 1;

        // IMF: irrigation applied to surface -- spray or drip
        if (public_xtra->clm_irr_type == 1
            || public_xtra->clm_irr_type == 2)
        {
          sprintf(file_type, "qflx_qirr");
          WriteSilo(file_prefix, file_type, file_postfix,
                    instance_xtra->qflx_qirr, t,
                    instance_xtra->file_number, "IrrigationSurface");
          clm_file_dumped = 1;
        }

        // IMF: irrigation applied directly as soil moisture flux -- "instant"
        if (public_xtra->clm_irr_type == 3)
        {
          sprintf(file_postfix, "qflx_qirr_inst");
          WriteSilo(file_prefix, file_type, file_postfix,
                    instance_xtra->qflx_qirr_inst, t,
                    instance_xtra->file_number, "IrrigationInstant");
          clm_file_dumped = 1;
        }
      }                         // end of if (write_silo_CLM)

      if (public_xtra->write_netcdf_clm)
      {
        sprintf(nc_postfix, "%05d", instance_xtra->file_number);
        WriteCLMNC(file_prefix, nc_postfix, t,
                   instance_xtra->eflx_lh_tot,
                   public_xtra->numCLMVarTimeVariant, "time", 1);
        WriteCLMNC(file_prefix, nc_postfix, t,
                   instance_xtra->eflx_lh_tot,
                   public_xtra->numCLMVarTimeVariant, "eflx_lh_tot",
                   2);
        WriteCLMNC(file_prefix, nc_postfix, t,
                   instance_xtra->eflx_lwrad_out,
                   public_xtra->numCLMVarTimeVariant, "eflx_lwrad_out",
                   2);
        WriteCLMNC(file_prefix, nc_postfix, t,
                   instance_xtra->eflx_sh_tot,
                   public_xtra->numCLMVarTimeVariant, "eflx_sh_tot",
                   2);
        WriteCLMNC(file_prefix, nc_postfix, t,
                   instance_xtra->eflx_soil_grnd,
                   public_xtra->numCLMVarTimeVariant, "eflx_soil_grnd",
                   2);
        WriteCLMNC(file_prefix, nc_postfix, t,
                   instance_xtra->qflx_evap_tot,
                   public_xtra->numCLMVarTimeVariant, "qflx_evap_tot",
                   2);
        WriteCLMNC(file_prefix, nc_postfix, t,
                   instance_xtra->qflx_evap_grnd,
                   public_xtra->numCLMVarTimeVariant, "qflx_evap_grnd",
                   2);
        WriteCLMNC(file_prefix, nc_postfix, t,
                   instance_xtra->qflx_evap_soi,
                   public_xtra->numCLMVarTimeVariant, "qflx_evap_soi",
                   2);
        WriteCLMNC(file_prefix, nc_postfix, t,
                   instance_xtra->qflx_evap_veg,
                   public_xtra->numCLMVarTimeVariant, "qflx_evap_veg",
                   2);
        WriteCLMNC(file_prefix, nc_postfix, t,
                   instance_xtra->qflx_tran_veg,
                   public_xtra->numCLMVarTimeVariant, "qflx_tran_veg",
                   2);
        WriteCLMNC(file_prefix, nc_postfix, t,
                   instance_xtra->qflx_infl,
                   public_xtra->numCLMVarTimeVariant, "qflx_infl", 2);
        WriteCLMNC(file_prefix, nc_postfix, t, instance_xtra->swe_out,
                   public_xtra->numCLMVarTimeVariant, "swe_out", 2);
        WriteCLMNC(file_prefix, nc_postfix, t, instance_xtra->t_grnd,
                   public_xtra->numCLMVarTimeVariant, "t_grnd", 2);
        WriteCLMNC(file_prefix, nc_postfix, t, instance_xtra->tsoil,
                   public_xtra->numCLMVarTimeVariant, "t_soil", 3);
        if (public_xtra->clm_irr_type == 1
            || public_xtra->clm_irr_type == 2)
        {
          WriteCLMNC(file_prefix, nc_postfix, t,
                     instance_xtra->qflx_qirr,
                     public_xtra->numCLMVarTimeVariant, "qflx_qirr",
                     2);
        }
        if (public_xtra->clm_irr_type == 3)
        {
          WriteCLMNC(file_prefix, nc_postfix, t,
                     instance_xtra->qflx_qirr_inst,
                     public_xtra->numCLMVarTimeVariant,
                     "qflx_qirr_inst", 3);
        }
        clm_file_dumped = 1;
      }                         // end of if (write_netcdf_clm)

      if (public_xtra->print_CLM)
      {
        if (public_xtra->single_clm_file)       //NBE
        {
          // NBE: CLM single file output
          PFVLayerCopy(0, 0, instance_xtra->clm_out_grid,
                       instance_xtra->eflx_lh_tot);
          PFVLayerCopy(1, 0, instance_xtra->clm_out_grid,
                       instance_xtra->eflx_lwrad_out);
          PFVLayerCopy(2, 0, instance_xtra->clm_out_grid,
                       instance_xtra->eflx_sh_tot);
          PFVLayerCopy(3, 0, instance_xtra->clm_out_grid,
                       instance_xtra->eflx_soil_grnd);
          PFVLayerCopy(4, 0, instance_xtra->clm_out_grid,
                       instance_xtra->qflx_evap_tot);
          PFVLayerCopy(5, 0, instance_xtra->clm_out_grid,
                       instance_xtra->qflx_evap_grnd);
          PFVLayerCopy(6, 0, instance_xtra->clm_out_grid,
                       instance_xtra->qflx_evap_soi);
          PFVLayerCopy(7, 0, instance_xtra->clm_out_grid,
                       instance_xtra->qflx_evap_veg);
          PFVLayerCopy(8, 0, instance_xtra->clm_out_grid,
                       instance_xtra->qflx_tran_veg);
          PFVLayerCopy(9, 0, instance_xtra->clm_out_grid,
                       instance_xtra->qflx_infl);
          PFVLayerCopy(10, 0, instance_xtra->clm_out_grid,
                       instance_xtra->swe_out);
          PFVLayerCopy(11, 0, instance_xtra->clm_out_grid,
                       instance_xtra->t_grnd);

          if (public_xtra->clm_irr_type == 1
              || public_xtra->clm_irr_type == 2)
          {
            PFVLayerCopy(12, 0, instance_xtra->clm_out_grid,
                         instance_xtra->qflx_qirr);
          }
          if (public_xtra->clm_irr_type == 3)
          {
            PFVLayerCopy(12, 0, instance_xtra->clm_out_grid,
                         instance_xtra->qflx_qirr_inst);
          }

          for (k = 0; k < public_xtra->clm_nz; k++)
          {
            //Write out the bottom layer in the lowest index position, build upward
            PFVLayerCopy(13 + k, k, instance_xtra->clm_out_grid,
                         instance_xtra->tsoil);
          }
          /* NBE: added .C instead of writing a different write function with
           * a different extension since PFB is hard-wired */
          sprintf(file_postfix, "clm_output.%05d.C",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->clm_out_grid);
          clm_file_dumped = 1;
          // Update with new timesteps
          /* No initial call to add the field and no support for .C.pfb files in vtkParFlowMetaReader yet.
           * MetadataAddDynamicField(
           * js_outputs, file_prefix, t, instance_xtra->file_number,
           * "CLM", "m", "cell", "subsurface", 0, NULL);
           */
          // End of CLM Single file output
        }
        else
        {
          // Otherwise do the old output
          sprintf(file_postfix, "eflx_lh_tot.%05d",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->eflx_lh_tot);
          clm_file_dumped = 1;

          sprintf(file_postfix, "eflx_lwrad_out.%05d",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->eflx_lwrad_out);
          clm_file_dumped = 1;

          sprintf(file_postfix, "eflx_sh_tot.%05d",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->eflx_sh_tot);
          clm_file_dumped = 1;

          sprintf(file_postfix, "eflx_soil_grnd.%05d",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->eflx_soil_grnd);
          clm_file_dumped = 1;

          sprintf(file_postfix, "qflx_evap_tot.%05d",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->qflx_evap_tot);
          clm_file_dumped = 1;

          sprintf(file_postfix, "qflx_evap_grnd.%05d",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->qflx_evap_grnd);
          clm_file_dumped = 1;

          sprintf(file_postfix, "qflx_evap_soi.%05d",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->qflx_evap_soi);
          clm_file_dumped = 1;

          sprintf(file_postfix, "qflx_evap_veg.%05d",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->qflx_evap_veg);
          clm_file_dumped = 1;

          sprintf(file_postfix, "qflx_tran_veg.%05d",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->qflx_tran_veg);
          clm_file_dumped = 1;

          sprintf(file_postfix, "qflx_infl.%05d",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->qflx_infl);
          clm_file_dumped = 1;

          sprintf(file_postfix, "swe_out.%05d",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->swe_out);
          clm_file_dumped = 1;

          sprintf(file_postfix, "t_grnd.%05d",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->t_grnd);
          clm_file_dumped = 1;

          sprintf(file_postfix, "t_soil.%05d",
                  instance_xtra->file_number);
          WritePFBinary(file_prefix, file_postfix,
                        instance_xtra->tsoil);
          clm_file_dumped = 1;

          // IMF: irrigation applied to surface -- spray or drip
          if (public_xtra->clm_irr_type == 1
              || public_xtra->clm_irr_type == 2)
          {
            sprintf(file_postfix, "qflx_qirr.%05d",
                    instance_xtra->file_number);
            WritePFBinary(file_prefix, file_postfix,
                          instance_xtra->qflx_qirr);
            clm_file_dumped = 1;
          }

          // IMF: irrigation applied directly as soil moisture flux -- "instant"
          if (public_xtra->clm_irr_type == 3)
          {
            sprintf(file_postfix, "qflx_qirr_inst.%05d",
                    instance_xtra->file_number);
            WritePFBinary(file_prefix, file_postfix,
                          instance_xtra->qflx_qirr_inst);
            clm_file_dumped = 1;
          }
        }                       // end of multi-file output - NBE
      }                         // end of if (print_CLM)
    }                           // end of if (clm_dump_files)
#endif

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
