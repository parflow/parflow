/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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
  **********************************************************************EHEADER*/

/*****************************************************************************
 *
 * Top level 
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include "parflow.h"

#include <float.h>

#define EPSILON 0.00000000001

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{ 
   PFModule          *permeability_face;
   PFModule          *phase_velocity_face;
   PFModule          *advect_concen;
   PFModule          *set_problem_data;
   PFModule          *nonlin_solver;

   Problem           *problem;

   int                advect_order;
   double             CFL;
   double             drop_tol;
   int                max_iterations;
   int                max_convergence_failures;   /* maximum number of convergence failures that are allowed */
   int                lsm;                        /* land surface model */

   int                print_subsurf_data;         /* print permeability/porosity? */
   int                print_press;                /* print pressures? */
   int                print_velocities;           /* print velocities? */
   int                print_satur;                /* print saturations? */
   int                print_concen;               /* print concentrations? */
   int                print_wells;                /* print well data? */
   int                write_silo_subsurf_data;    /* write permeability/porosity? */
   int                write_silo_press;           /* write pressures? */
   int                write_silo_velocities;      /* write velocities? */
   int                write_silo_satur;           /* write saturations? */
   int                write_silo_concen;          /* write concentrations? */
   int                write_silo_mask;            /* write mask? */
   int                write_silo_evaptrans;       /* write evaptrans? */
   int                write_silo_evaptrans_sum;   /* write evaptrans sum? */
   int                write_silo_slopes;          /* write slopes? */
   int                write_silo_mannings;        /* write mannings? */
   int                write_silo_specific_storage;/* write specific storage? */
   int                write_silo_overland_sum;    /* write sum of overland outflow? */

#ifdef HAVE_CLM                           /* VARIABLES FOR CLM ONLY */
   char              *clm_file_dir;       /* directory location for CLM files */
   int                clm_dump_interval;  /* time interval, integer, for CLM output */
   int                clm_1d_out;         /* boolean 0-1, integer, for CLM 1-d output */
   int                clm_bin_out_dir;    /* boolean 0-1, integer, for sep dirs for each clm binary output */
   int                clm_dump_files;     /* boolean 0-1, integer, for write CLM output from PF */

   int                clm_istep_start;    /* CLM time counter for met forcing (line in 1D file; name extension of 2D/3D files) */
   int                clm_fstep_start;    /* CLM time counter for inside met forcing files -- used for time keeping w/in 3D met files */
   int                clm_metforce;       /* CLM met forcing  -- 1=uniform (default), 2=distributed, 3=distributed w/ multiple timesteps */
   int                clm_metnt;          /* CLM met forcing  -- if 3D, length of time axis in each file */
   int                clm_metsub;         /* Flag for met vars in subdirs of clm_metpath or all in clm_metpath */
   char              *clm_metfile;        /* File name for 1D forcing *or* base name for 2D forcing */
   char              *clm_metpath;        /* Path to CLM met forcing file(s) */
   double            *sw1d,*lw1d,*prcp1d, /* 1D forcing variables */
                     *tas1d,*u1d,*v1d,*patm1d,*qatm1d;

   int                clm_beta_function;  /* CLM evap function for var sat 0=none, 1=linear, 2=cos */
   double             clm_res_sat;        /* CLM residual saturation in soil sat units [-] */
   int                clm_veg_function;   /* CLM veg function for water stress 0=none, 1=press, 2=sat */
   double             clm_veg_wilting;    /* CLM veg function wilting point in meters or soil moisture */
   double             clm_veg_fieldc;     /* CLM veg function field capacity in meters or soil moisture */

   int                clm_irr_type;       /* CLM irrigation type flag -- 0=none, 1=Spray, 2=Drip, 3=Instant */
   int                clm_irr_cycle;      /* CLM irrigation cycle flag -- 0=Constant, 1=Deficit */
   double             clm_irr_rate;       /* CLM irrigation application rate [mm/s] */
   double             clm_irr_start;      /* CLM irrigation schedule -- start time of constant cycle [GMT] */
   double             clm_irr_stop;       /* CLM irrigation schedule -- stop time of constant cyle [GMT] */
   double             clm_irr_threshold;  /* CLM irrigation schedule -- soil moisture threshold for deficit cycle */
   int                clm_irr_thresholdtype;  /* Decicit-based saturation criteria (top, bottom, column avg) */
#endif

   int                print_lsm_sink;     /* print LSM sink term? */
   int                write_silo_CLM;     /* write CLM output as silo? */
   int                write_CLM_binary;   /* write binary output (**default**)? */

} PublicXtra; 

typedef struct
{ 
   PFModule          *permeability_face;
   PFModule          *phase_velocity_face;
   PFModule          *advect_concen;
   PFModule          *set_problem_data;

   PFModule          *retardation;
   PFModule          *phase_rel_perm;
   PFModule          *ic_phase_pressure;
   PFModule          *ic_phase_concen;
   PFModule          *problem_saturation;
   PFModule          *phase_density;
   PFModule          *select_time_step;
   PFModule          *l2_error_norm;
   PFModule          *nonlin_solver;

   Grid              *grid;
   Grid              *grid2d;
   Grid              *x_grid;
   Grid              *y_grid;
   Grid              *z_grid;

   ProblemData       *problem_data;

   double            *temp_data;

   /*****************************************************************************
    * Local variables that need to be kept around 
    *****************************************************************************/
   Vector      *pressure;
   Vector      *saturation;
   Vector      *density;
   Vector      *old_density;
   Vector      *old_saturation;
   Vector      *old_pressure;
   Vector      *mask;

   Vector      *evap_trans_sum;       /* running sum of evaporation and transpiration */
   Vector      *overland_sum;         
   Vector      *ovrl_bc_flx;          /* vector containing outflow at the boundary */

#ifdef HAVE_CLM
   /* RM: vars for pf printing of clm output */
   Vector      *eflx_lh_tot;          /* total LH flux from canopy height to atmosphere [W/m^2] */
   Vector      *eflx_lwrad_out;       /* outgoing LW radiation from ground+canopy [W/m^2] */
   Vector      *eflx_sh_tot;          /* total SH flux from canopy height to atmosphere [W/m^2] */
   Vector      *eflx_soil_grnd;       /* ground heat flux [W/m^2] */
   Vector      *qflx_evap_tot;        /* total ET flux from canopy height to atmosphere [mm/s] */
   Vector      *qflx_evap_grnd;       /* evap flux from ground (first soil layer) [mm/s] (defined equal to qflx_evap_soi) */
   Vector      *qflx_evap_soi;        /* evap flux from ground [mm/s] */
   Vector      *qflx_evap_veg;        /* evap+trans from leaves [mm/s] */
   Vector      *qflx_tran_veg;        /* trans from veg [mm/s] */
   Vector      *qflx_infl;            /* infiltration [mm/s] */
   Vector      *swe_out;              /* snow water equivalent [mm] */
   Vector      *t_grnd;               /* CLM soil surface temperature [K] */
   Vector      *tsoil;                /* CLM soil temp, all 10 layers [K] */
   Grid        *gridTs;               /* New grid fro tsoi (nx*ny*10) */

   /* IMF: vars for printing clm irrigation output */
   Vector      *irr_flag;             /* Flag for irrigating/pumping under deficit-based irrigation scheme */
   Vector      *qflx_qirr;            /* Irrigation applied at surface -- spray or drip */
   Vector      *qflx_qirr_inst;       /* Irrigation applied by inflating soil moisture -- "instant" */

   /* IMF: vars for distributed met focing */
   Grid        *metgrid;              /* new grid for 2D or 3D met forcing vars (nx*ny*clm_metnt; clm_metnt defaults to 1) */
   Vector      *sw_forc;              /* shortwave radiation forcing [W/m^2] */  
   Vector      *lw_forc;              /* longwave radiation forcing [W/m^2] */
   Vector      *prcp_forc;            /* precipitation [mm/s] */
   Vector      *tas_forc;             /* air temp [K] @ ref height (hgt set in drv_clmin.dat, currently 2m) */
   Vector      *u_forc;               /* east-west wind [m/s] @ ref height (hgt set in drv_clmin.dat, currently 10m) */
   Vector      *v_forc;               /* south-north wind [m/s] @ ref height (hgt set in drv_clmin.dat, currently 10m)*/
   Vector      *patm_forc;            /* surface air pressure [Pa] */
   Vector      *qatm_forc;            /* surface air humidity [kg/kg] @ ref height (hgt set in drv_clmin.dat, currently 2m) */ 
#endif

   double      *time_log;
   double      *dt_log;
   double      *outflow_log;
   int         *seq_log;
   int         *dumped_log;
   char        *recomp_log; 
   char        *dt_info_log;

   int          file_number;
   int          number_logged;
   int          iteration_number;
   double       dump_index;

} InstanceXtra; 

void SetupRichards(PFModule *this_module) {
   PublicXtra   *public_xtra         = (PublicXtra *)PFModulePublicXtra(this_module);
   InstanceXtra *instance_xtra       = (InstanceXtra *)PFModuleInstanceXtra(this_module);
   Problem      *problem             = (public_xtra -> problem);
   PFModule     *ic_phase_pressure   = (instance_xtra -> ic_phase_pressure);
   PFModule     *phase_density       = (instance_xtra -> phase_density);
   PFModule     *problem_saturation  = (instance_xtra -> problem_saturation);

   int           print_subsurf_data  = (public_xtra -> print_subsurf_data);
   int           print_press         = (public_xtra -> print_press);
   int           print_satur         = (public_xtra -> print_satur);
   int           print_wells         = (public_xtra -> print_wells);

   ProblemData  *problem_data        = (instance_xtra -> problem_data);
   PFModule     *set_problem_data    = (instance_xtra -> set_problem_data);
   Grid         *grid                = (instance_xtra -> grid);
   Grid         *grid2d              = (instance_xtra -> grid2d);

   double        start_time          = ProblemStartTime(problem);
   double        stop_time           = ProblemStopTime(problem);
   int           start_count         = ProblemStartCount(problem);

   char          file_prefix[64], file_postfix[64];

   int           take_more_time_steps;

   double        t;
   double        dt = 0.0;
   double        gravity = ProblemGravity(problem);

   double        dtmp;

   CommHandle   *handle;

   int           any_file_dumped;

#ifdef HAVE_CLM
   /* IMF: for CLM met forcings (local to SetupRichards)*/
   char          filename[128];         
   int           n,nc;
   int           ch;
   double        sw,lw,prcp,tas,u,v,patm,qatm;         // forcing vars
   FILE         *metf_temp;                            // temp file for forcings   
   amps_Invoice  invoice;                              // for distributing 1D met forcings
   amps_File     metf1d;                               // for distributing 1D met forcings
   Grid         *metgrid = (instance_xtra -> metgrid); // grid for 2D and 3D met forcings
   Grid         *gridTs = (instance_xtra -> gridTs);   // grid for writing T-soil or instant irrig flux as Silo
#endif

   t = start_time;
   dt = 0.0e0;

   IfLogging(1)
   {
      int max_iterations            = (public_xtra -> max_iterations);
      instance_xtra -> seq_log      = talloc(int,    max_iterations + 1);
      instance_xtra -> time_log     = talloc(double, max_iterations + 1);
      instance_xtra -> dt_log       = talloc(double, max_iterations + 1);
      instance_xtra -> dt_info_log  = talloc(char,   max_iterations + 1);
      instance_xtra -> dumped_log   = talloc(int,    max_iterations + 1);
      instance_xtra -> recomp_log   = talloc(char,   max_iterations + 1);
      instance_xtra -> outflow_log  = talloc(double, max_iterations + 1);
      instance_xtra -> number_logged = 0;
   }

   sprintf(file_prefix, GlobalsOutFileName);

   /* Do turning bands (and other stuff maybe) */
   PFModuleInvokeType(SetProblemDataInvoke, set_problem_data, (problem_data));
   ComputeTop(problem, problem_data);

   /* Write subsurface data */
   if ( print_subsurf_data )
   {
      sprintf(file_postfix, "perm_x");
      WritePFBinary(file_prefix, file_postfix, ProblemDataPermeabilityX(problem_data));

      sprintf(file_postfix, "perm_y");
      WritePFBinary(file_prefix, file_postfix, ProblemDataPermeabilityY(problem_data));

      sprintf(file_postfix, "perm_z");
      WritePFBinary(file_prefix, file_postfix, ProblemDataPermeabilityZ(problem_data));

      sprintf(file_postfix, "porosity");
      WritePFBinary(file_prefix, file_postfix, ProblemDataPorosity(problem_data));
   }

   if ( public_xtra -> write_silo_subsurf_data )
   {
      sprintf(file_postfix, "perm_x");
      WriteSilo(file_prefix, file_postfix, ProblemDataPermeabilityX(problem_data),
                t, 0, "PermeabilityX");

      sprintf(file_postfix, "perm_y");
      WriteSilo(file_prefix, file_postfix, ProblemDataPermeabilityY(problem_data),
                t, 0, "PermeabilityY");

      sprintf(file_postfix, "perm_z");
      WriteSilo(file_prefix, file_postfix, ProblemDataPermeabilityZ(problem_data),
                t, 0, "PermeabilityZ");

      sprintf(file_postfix, "porosity");
      WriteSilo(file_prefix, file_postfix, ProblemDataPorosity(problem_data),
	        t, 0, "Porosity");
   }

   if ( public_xtra -> write_silo_slopes )
   {
      sprintf(file_postfix, "slope_x");
      WriteSilo(file_prefix, file_postfix, ProblemDataTSlopeX(problem_data),
                t, 0, "SlopeX");

      sprintf(file_postfix, "slope_y");
      WriteSilo(file_prefix, file_postfix, ProblemDataTSlopeY(problem_data),
                t, 0, "SlopeY");
   }

   if ( public_xtra -> write_silo_mannings )
   {
      sprintf(file_postfix, "mannings");
      WriteSilo(file_prefix, file_postfix, ProblemDataMannings(problem_data),
                t, 0, "Mannings");
   }

   if ( public_xtra -> write_silo_specific_storage )
   {
      sprintf(file_postfix, "specific_storage");
      WriteSilo(file_prefix, file_postfix, ProblemDataSpecificStorage(problem_data),
                t, 0, "SpecificStorage");
   }

   if(!amps_Rank(amps_CommWorld))
   {
      PrintWellData(ProblemDataWellData(problem_data), 
		    (WELLDATA_PRINTPHYSICAL | WELLDATA_PRINTVALUES));
   }

   /* Check to see if pressure solves are requested */
   /* start_count < 0 implies that subsurface data ONLY is requested */
   /*    Thus, we do not want to allocate memory or initialize storage for */
   /*    other variables.  */
   if ( start_count < 0 )
   {
      take_more_time_steps = 0;
   }
   else  
   {
      take_more_time_steps = 1;
   }

   instance_xtra -> iteration_number = instance_xtra -> file_number = start_count;
   instance_xtra -> dump_index = 1.0;

   if ( ( (t >= stop_time) || (instance_xtra -> iteration_number > public_xtra -> max_iterations) ) 
	&& ( take_more_time_steps == 1) )
   {
      take_more_time_steps = 0;

      print_press           = 0;
      print_satur           = 0;
      print_wells           = 0;
   }
         
   if (take_more_time_steps)
   {
      /*-------------------------------------------------------------------
       * Allocate and set up initial values
       *-------------------------------------------------------------------*/

      instance_xtra -> pressure = NewVector( grid, 1, 1 );
      InitVectorAll(instance_xtra -> pressure, -FLT_MAX);
      // InitVectorAll(instance_xtra -> pressure, 0.0);      

      instance_xtra -> saturation = NewVector( grid, 1, 1 );
      InitVectorAll(instance_xtra -> saturation, -FLT_MAX);
      // InitVectorAll(instance_xtra -> saturation, 0.0);

      instance_xtra -> density = NewVector( grid, 1, 1 );
      InitVectorAll(instance_xtra -> density, 0.0);

      instance_xtra -> old_pressure = NewVector( grid, 1, 1 );
      InitVectorAll(instance_xtra -> old_pressure, 0.0);

      instance_xtra -> old_saturation = NewVector( grid, 1, 1 );
      InitVectorAll(instance_xtra -> old_saturation, 0.0);

      instance_xtra -> old_density = NewVector( grid, 1, 1 );
      InitVectorAll(instance_xtra -> old_density, 0.0);

      /*sk Initialize Overland flow boundary fluxes*/
      instance_xtra -> ovrl_bc_flx = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> ovrl_bc_flx, 0.0);

      if(public_xtra -> write_silo_overland_sum) 
      {
			instance_xtra -> overland_sum = NewVector( grid2d, 1, 1 );
			InitVectorAll(instance_xtra -> overland_sum, 0.0);
      }

      /*IMF these need to be outside of ifdef or won't run w/o CLM */
      /*sk Initialize LSM mask */
      instance_xtra -> mask = NewVector( grid, 1, 1 );
      InitVectorAll(instance_xtra -> mask, 0.0);

      instance_xtra -> evap_trans_sum = NewVector( grid, 1, 0 );
      InitVectorAll(instance_xtra -> evap_trans_sum, 0.0);

/* IMF: the following are only used w/ CLM */
#ifdef HAVE_CLM 

      /* SGS FIXME should only init these if we are actually running with CLM */
      /*sk Initialize LSM mask */
      //instance_xtra -> mask = NewVector( grid, 1, 1 );
      //InitVectorAll(instance_xtra -> mask, 0.0);
      //
      //instance_xtra -> evap_trans_sum = NewVector( grid, 1, 0 );
      //InitVectorAll(instance_xtra -> evap_trans_sum, 0.0);

      /*IMF Initialize variables for printing CLM output*/
      instance_xtra -> eflx_lh_tot = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> eflx_lh_tot, 0.0);
	   
      instance_xtra -> eflx_lwrad_out = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> eflx_lwrad_out, 0.0);		
	
      instance_xtra -> eflx_sh_tot = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> eflx_sh_tot, 0.0);
	
      instance_xtra -> eflx_soil_grnd = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> eflx_soil_grnd, 0.0);
		
      instance_xtra -> qflx_evap_tot = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> qflx_evap_tot, 0.0);
	
      instance_xtra -> qflx_evap_grnd = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> qflx_evap_grnd, 0.0);

      instance_xtra -> qflx_evap_soi = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> qflx_evap_soi, 0.0);
		
      instance_xtra -> qflx_evap_veg = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> qflx_evap_veg, 0.0);		
		
      instance_xtra -> qflx_tran_veg = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> qflx_tran_veg, 0.0);
		
      instance_xtra -> qflx_infl = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> qflx_infl, 0.0);
		
      instance_xtra -> swe_out = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> swe_out, 0.0);
		
      instance_xtra -> t_grnd = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> t_grnd, 0.0);

      instance_xtra -> tsoil = NewVector( gridTs, 1, 1);
      InitVectorAll(instance_xtra -> tsoil, 0.0);

      /*IMF Initialize variables for CLM irrigation output */
      instance_xtra -> irr_flag  = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> irr_flag, 0.0);
   
      instance_xtra -> qflx_qirr = NewVector( grid2d, 1, 1 );
      InitVectorAll(instance_xtra -> qflx_qirr, 0.0);

      instance_xtra -> qflx_qirr_inst = NewVector( gridTs, 1, 1);
      InitVectorAll(instance_xtra -> qflx_qirr_inst, 0.0);

      /*IMF Initialize variables for CLM forcing fields
            SW rad, LW rad, precip, T(air), U, V, P(air), q(air) */
      instance_xtra -> sw_forc = NewVector( metgrid, 1, 1 );
      InitVectorAll(instance_xtra -> sw_forc, 100.0);

      instance_xtra -> lw_forc = NewVector( metgrid, 1, 1 );
      InitVectorAll(instance_xtra -> lw_forc, 100.0);

      instance_xtra -> prcp_forc = NewVector( metgrid, 1, 1 );
      InitVectorAll(instance_xtra -> prcp_forc, 100.0);

      instance_xtra -> tas_forc = NewVector( metgrid, 1, 1 );
      InitVectorAll(instance_xtra -> tas_forc, 100.0);

      instance_xtra -> u_forc = NewVector( metgrid, 1, 1 );
      InitVectorAll(instance_xtra -> u_forc, 100.0);

      instance_xtra -> v_forc = NewVector( metgrid, 1, 1 );
      InitVectorAll(instance_xtra -> v_forc, 100.0);

      instance_xtra -> patm_forc = NewVector( metgrid, 1, 1 );
      InitVectorAll(instance_xtra -> patm_forc, 100.0);

      instance_xtra -> qatm_forc = NewVector( metgrid, 1, 1 );
      InitVectorAll(instance_xtra -> qatm_forc, 100.0); 

      /*IMF If 1D met forcing, read forcing vars to arrays */
      if (public_xtra -> clm_metforce == 1)
      {
         // Set filename for 1D forcing file
         sprintf(filename, "%s/%s", public_xtra -> clm_metpath, public_xtra -> clm_metfile);

         // Open file, count number of lines
         if ( (metf_temp = fopen(filename,"r")) == NULL )
         {
            printf("Error: can't open file %s \n", filename);
            exit(1);
         }
         else
         {
            nc   = 0;
            while( (ch=fgetc(metf_temp)) != EOF ) if (ch==10) nc++;
            fclose( metf_temp );
         }
         // Read 1D met file to arrays of length nc
         (public_xtra -> sw1d)   = ctalloc(double,nc);
         (public_xtra -> lw1d)   = ctalloc(double,nc);
         (public_xtra -> prcp1d) = ctalloc(double,nc);
         (public_xtra -> tas1d)  = ctalloc(double,nc);
         (public_xtra -> u1d)    = ctalloc(double,nc);
         (public_xtra -> v1d)    = ctalloc(double,nc);
         (public_xtra -> patm1d) = ctalloc(double,nc);
         (public_xtra -> qatm1d) = ctalloc(double,nc);
         if ((metf1d = amps_SFopen( filename, "r")) == NULL )
         {
            amps_Printf( "Error: can't open file %s \n", filename);
            exit(1);
         }
         invoice = amps_NewInvoice( "%d%d%d%d%d%d%d%d", &sw,&lw,&prcp,&tas,&u,&v,&patm,&qatm );
         for (n=0; n<nc; n++)
         {
            amps_SFBCast( amps_CommWorld, metf1d, invoice);
            (public_xtra -> sw1d)[n]   = sw;
            (public_xtra -> lw1d)[n]   = lw;
            (public_xtra -> prcp1d)[n] = prcp;
            (public_xtra -> tas1d)[n]  = tas;
            (public_xtra -> u1d)[n]    = u;
            (public_xtra -> v1d)[n]    = v;
            (public_xtra -> patm1d)[n] = patm;
            (public_xtra -> qatm1d)[n] = qatm;
         }
         amps_FreeInvoice( invoice );
         amps_SFclose( metf1d );
      } 

#endif

      /* Set initial pressures and pass around ghost data to start */
      PFModuleInvokeType(ICPhasePressureInvoke, 
			 ic_phase_pressure, 
			 (instance_xtra -> pressure, instance_xtra -> mask, problem_data, problem));

      handle = InitVectorUpdate(instance_xtra -> pressure, VectorUpdateAll);
      FinalizeVectorUpdate(handle); 

      /* Set initial densities and pass around ghost data to start */
      PFModuleInvokeType(PhaseDensityInvoke,  
			 phase_density, 
			 (0, instance_xtra -> pressure, instance_xtra -> density, &dtmp, &dtmp, CALCFCN));

      handle = InitVectorUpdate(instance_xtra -> density, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      /* Set initial saturations */
      PFModuleInvokeType(SaturationInvoke, problem_saturation, 
		    (instance_xtra -> saturation, instance_xtra -> pressure, instance_xtra -> density, gravity, problem_data, 
		      CALCFCN));

      handle = InitVectorUpdate(instance_xtra -> pressure, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      /*-----------------------------------------------------------------
       * Allocate phase velocities 
       *-----------------------------------------------------------------*/
      /*
	phase_x_velocity = ctalloc(Vector *, ProblemNumPhases(problem) );
	for(phase = 0; phase < ProblemNumPhases(problem); phase++)
	{
	phase_x_velocity[phase] = NewVector( x_grid, 1, 1 );
	InitVectorAll(phase_x_velocity[phase], 0.0);
	}

	phase_y_velocity = ctalloc(Vector *, ProblemNumPhases(problem) );
	for(phase = 0; phase < ProblemNumPhases(problem); phase++)
	{
	phase_y_velocity[phase] = NewVector( y_grid, 1, 1 );
	InitVectorAll(phase_y_velocity[phase], 0.0);
	}

	phase_z_velocity = ctalloc(Vector *, ProblemNumPhases(problem) );
	for(phase = 0; phase < ProblemNumPhases(problem); phase++)
	{
	phase_z_velocity[phase] = NewVector( z_grid, 1, 2 );
	InitVectorAll(phase_z_velocity[phase], 0.0);
	}
      */

      /*****************************************************************/
      /*          Print out any of the requested initial data          */
      /*****************************************************************/

      any_file_dumped = 0;

      /*-------------------------------------------------------------------
       * Print out the initial well data?
       *-------------------------------------------------------------------*/

      if ( print_wells )
      {
	 WriteWells(file_prefix,
		    problem,
		    ProblemDataWellData(problem_data),
		    t, 
		    WELLDATA_WRITEHEADER);
      }

      /*-----------------------------------------------------------------
       * Print out the initial pressures?
       *-----------------------------------------------------------------*/

      if ( print_press )
      {
	 sprintf(file_postfix, "press.%05d", instance_xtra -> file_number );
	 WritePFBinary(file_prefix, file_postfix, instance_xtra -> pressure );
	 any_file_dumped = 1;
      }

      if ( public_xtra -> write_silo_press )
      {
	 sprintf(file_postfix, "press.%05d", instance_xtra -> file_number );
	 WriteSilo(file_prefix, file_postfix, instance_xtra -> pressure,
                   t, instance_xtra -> file_number, "Pressure");
	 any_file_dumped = 1;
      }

      /*-----------------------------------------------------------------
       * Print out the initial saturations?
       *-----------------------------------------------------------------*/

      if ( print_satur )
      {
	 sprintf(file_postfix, "satur.%05d", instance_xtra -> file_number );
	 WritePFBinary(file_prefix, file_postfix, instance_xtra -> saturation );
	 any_file_dumped = 1;
      }

      if ( public_xtra -> write_silo_satur )
      {
	 sprintf(file_postfix, "satur.%05d", instance_xtra -> file_number );
	 WriteSilo(file_prefix, file_postfix, instance_xtra -> saturation, 
                   t, instance_xtra -> file_number, "Saturation");
	 any_file_dumped = 1;
      }

      /*-----------------------------------------------------------------
       * Print out mask?
       *-----------------------------------------------------------------*/
 
      if ( print_satur )
      {
	 sprintf(file_postfix, "mask");
	 WritePFBinary(file_prefix, file_postfix, instance_xtra -> mask );
	 any_file_dumped = 1;
      }


      if ( public_xtra -> write_silo_mask )
      {
	 sprintf(file_postfix, "mask");
	 WriteSilo(file_prefix, file_postfix, instance_xtra -> mask, 
                   t, instance_xtra -> file_number, "Mask");
	 any_file_dumped = 1;
      }

      /*-----------------------------------------------------------------
       * Log this step
       *-----------------------------------------------------------------*/

      IfLogging(1)
      {
	 double        outflow = 0.0;

	 /*
	  * SGS Better error handing should be added 
	  */

	 if(instance_xtra -> number_logged > public_xtra -> max_iterations + 1) {
	    printf("Error: max_iterations reached, can't log anymore data\n");
	    exit(1);
	 }

	 instance_xtra -> seq_log[instance_xtra -> number_logged]       = instance_xtra -> iteration_number;
	 instance_xtra -> time_log[instance_xtra -> number_logged]      = t;
	 instance_xtra -> dt_log[instance_xtra -> number_logged]        = dt;
	 instance_xtra -> dt_info_log[instance_xtra -> number_logged]   = 'i';
	 instance_xtra -> outflow_log[instance_xtra -> number_logged]   = outflow;
	 if ( any_file_dumped )
	 {
	    instance_xtra -> dumped_log[instance_xtra -> number_logged] = instance_xtra -> file_number;
	 } else
	 {
	    instance_xtra -> dumped_log[instance_xtra -> number_logged] = -1;

	 }
	 instance_xtra -> recomp_log[instance_xtra -> number_logged]   = 'n';
	 instance_xtra -> number_logged++;
      }

      if (any_file_dumped) 
      {
	 instance_xtra -> file_number++;
      }

   } /* End if take_more_time_steps */
}

void AdvanceRichards(PFModule *this_module, 
		     double start_time,      /* Starting time */
		     double stop_time,       /* Stopping time */
		     PFModule *time_step_control, /* Use this module to control timestep if supplied */
		     Vector *evap_trans,     /* Flux from land surface model */ 
		     Vector **pressure_out,  /* Output vars */
		     Vector **porosity_out,
		     Vector **saturation_out
   ) 
{

   PublicXtra   *public_xtra        = (PublicXtra *)PFModulePublicXtra(this_module);
   InstanceXtra *instance_xtra      = (InstanceXtra *)PFModuleInstanceXtra(this_module);
   Problem      *problem            = (public_xtra -> problem);

   int           max_iterations      = (public_xtra -> max_iterations);
   int           print_satur         = (public_xtra -> print_satur);
   int           print_wells         = (public_xtra -> print_wells);

   PFModule     *problem_saturation  = (instance_xtra -> problem_saturation);
   PFModule     *phase_density       = (instance_xtra -> phase_density);
   PFModule     *select_time_step    = (instance_xtra -> select_time_step);
   PFModule     *l2_error_norm       = (instance_xtra -> l2_error_norm);
   PFModule     *nonlin_solver       = (instance_xtra -> nonlin_solver);

   ProblemData  *problem_data        = (instance_xtra -> problem_data);

   int           start_count         = ProblemStartCount(problem);
   double        dump_interval       = ProblemDumpInterval(problem);

   Vector       *porosity            = ProblemDataPorosity(problem_data);
   Vector       *evap_trans_sum      = instance_xtra -> evap_trans_sum;
   Vector       *overland_sum        = instance_xtra -> overland_sum;     /* sk: Vector of outflow at the boundary*/

/* IMF: The following are only used w/ CLM */
#ifdef HAVE_CLM
   Grid         *grid                = (instance_xtra -> grid);
   Subgrid      *subgrid;
   Subvector    *p_sub, *s_sub, *et_sub, *m_sub, *po_sub;
   double       *pp,*sp, *et, *ms, *po_dat;     

   /* IMF: For CLM met forcing (local to AdvanceRichards) */
   int           istep;                                           // IMF: counter for clm output times
   int           fstep,fflag,fstart,fstop;                        // IMF: index w/in 3D forcing array corresponding to istep
   int           n;                                               // IMF: index vars for looping over subgrid data
   double        sw,lw,prcp,tas,u,v,patm,qatm;                    // IMF: 1D forcing vars (local to AdvanceRichards) 
   double       *sw_data,*lw_data,*prcp_data,                     // IMF: 2D forcing vars (SubvectorData) (local to AdvanceRichards)
                *tas_data,*u_data,*v_data,*patm_data,*qatm_data;  
   char          filename[128];                                   // IMF: 1D input file name *or* 2D/3D input file base name
   Subvector    *sw_forc_sub, *lw_forc_sub, *prcp_forc_sub, *tas_forc_sub, 
                *u_forc_sub, *v_forc_sub, *patm_forc_sub, *qatm_forc_sub;

   /* IMF: For writing CLM output */ 
   Subvector    *eflx_lh_tot_sub, *eflx_lwrad_out_sub, *eflx_sh_tot_sub, *eflx_soil_grnd_sub,
                *qflx_evap_tot_sub, *qflx_evap_grnd_sub, *qflx_evap_soi_sub, *qflx_evap_veg_sub, 
                *qflx_tran_veg_sub, *qflx_infl_sub, *swe_out_sub, *t_grnd_sub, *tsoil_sub, 
                *irr_flag_sub, *qflx_qirr_sub, *qflx_qirr_inst_sub;
   double       *eflx_lh, *eflx_lwrad, *eflx_sh, *eflx_grnd, *qflx_tot, *qflx_grnd, *qflx_soi, 
                *qflx_eveg, *qflx_tveg, *qflx_in, *swe, *t_g, *t_soi, *iflag, *qirr, *qirr_inst;
   int           clm_file_dir_length;
#endif

   int           rank;
   int           any_file_dumped;
   int           dump_files;
   int           retval;
   int           converged;
   int           take_more_time_steps;
   int           conv_failures;
   int           max_failures         = public_xtra -> max_convergence_failures;

   double        t;
   double        dt = 0.0;
   double        ct = 0.0;
   double        cdt = 0.0;
   double        print_dt;
   double        dtmp, err_norm;
   double        gravity = ProblemGravity(problem);

   double        outflow = 0.0;      //sk Outflow due to overland flow

   CommHandle   *handle;

   char          dt_info;
   char          file_prefix[64], file_postfix[64];

   sprintf(file_prefix, GlobalsOutFileName);

   /***********************************************************************/
   /*                                                                     */
   /*                Begin the main computational section                 */
   /*                                                                     */
   /***********************************************************************/
 
   // Initialize ct in either case
   ct = start_time;
   t  = start_time;
   if(time_step_control) {
      PFModuleInvokeType(SelectTimeStepInvoke, time_step_control, (&cdt, &dt_info, t, problem,
								  problem_data) );
   } else {

      PFModuleInvokeType(SelectTimeStepInvoke, select_time_step, (&cdt, &dt_info, t, problem,
								  problem_data) );
   }
   dt = cdt;

   rank = amps_Rank(amps_CommWorld);

   /* 
      Check to see if pressure solves are requested 
      start_count < 0 implies that subsurface data ONLY is requested 
      Thus, we do not want to allocate memory or initialize storage for 
      other variables.
   */

   if ( start_count < 0 )
   {
      take_more_time_steps = 0;
   }
   else  
   {
      take_more_time_steps = 1;
   }



#ifdef HAVE_CLM
   istep  = public_xtra -> clm_istep_start;     // IMF: initialize time counter for CLM
   fflag  = 0;                                  // IMF: flag tripped when first met file is read
   fstart = 0;                                  // init to something, only used with 3D met forcing
   fstop  = 0;                                  // init to something, only used with 3D met forcing
#endif

   do  /* while take_more_time_steps */
   {
      if (t == ct)
      { 

	 ct += cdt;

/* IMF: The following are only used w/ CLM */
#ifdef HAVE_CLM      

         BeginTiming(CLMTimingIndex);

	 // SGS FIXME this should not be here, should not be reading input at this point
	 // Should get these values from somewhere else.
	 /* sk: call to the land surface model/subroutine*/
	 /* sk: For the couple with CLM*/
	 int p = GetInt("Process.Topology.P");
	 int q = GetInt("Process.Topology.Q");
	 int r = GetInt("Process.Topology.R");
	 int is;

         /* IMF: If 1D met forcing */ 
         if (public_xtra -> clm_metforce == 1)
         {         
            // Read forcing values for correct timestep
            sw   = (public_xtra -> sw1d)[istep-1];
            lw   = (public_xtra -> lw1d)[istep-1];
            prcp = (public_xtra -> prcp1d)[istep-1];
            tas  = (public_xtra -> tas1d)[istep-1];
            u    = (public_xtra -> u1d)[istep-1];
            v    = (public_xtra -> v1d)[istep-1];
            patm = (public_xtra -> patm1d)[istep-1];
            qatm = (public_xtra -> qatm1d)[istep-1];
         } //end if (clm_metforce==1)
	 else
	 {
	    // Initialize unused variables to something 
            sw   = 0.0;
            lw   = 0.0;
            prcp = 0.0;
            tas  = 0.0;
            u    = 0.0;
            v    = 0.0;
            patm = 0.0;
            qatm = 0.0;
	 }

         /* IMF: If 2D met forcing...read input files @ each timestep... */
         if ( public_xtra -> clm_metforce == 2 )
         {
            // Subdirectories for each variable?
            if ( public_xtra -> clm_metsub )
            {
               sprintf(filename, "%s/%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, "DSWR",  public_xtra -> clm_metfile, "DSWR",  istep);
               ReadPFBinary( filename, instance_xtra -> sw_forc );  
               sprintf(filename, "%s/%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, "DLWR",  public_xtra -> clm_metfile, "DLWR",  istep);
               ReadPFBinary( filename, instance_xtra -> lw_forc );
               sprintf(filename, "%s/%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, "APCP",  public_xtra -> clm_metfile, "APCP",  istep);
               ReadPFBinary( filename, instance_xtra -> prcp_forc );
               sprintf(filename, "%s/%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, "Temp",  public_xtra -> clm_metfile, "Temp",  istep);
               ReadPFBinary( filename, instance_xtra -> tas_forc );
               sprintf(filename, "%s/%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, "UGRD",  public_xtra -> clm_metfile, "UGRD",  istep);
               ReadPFBinary( filename, instance_xtra -> u_forc );
               sprintf(filename, "%s/%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, "VGRD",  public_xtra -> clm_metfile, "VGRD",  istep);
               ReadPFBinary( filename, instance_xtra -> v_forc );
               sprintf(filename, "%s/%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, "Press", public_xtra -> clm_metfile, "Press", istep);
               ReadPFBinary( filename, instance_xtra -> patm_forc );
               sprintf(filename, "%s/%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, "SPFH",  public_xtra -> clm_metfile, "SPFH",  istep);
               ReadPFBinary( filename, instance_xtra -> qatm_forc );
            }
            else
            {
               sprintf(filename, "%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, public_xtra -> clm_metfile, "DSWR", istep);
               ReadPFBinary( filename, instance_xtra -> sw_forc );
               sprintf(filename, "%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, public_xtra -> clm_metfile, "DLWR", istep);
               ReadPFBinary( filename, instance_xtra -> lw_forc );
               sprintf(filename, "%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, public_xtra -> clm_metfile, "APCP", istep);
               ReadPFBinary( filename, instance_xtra -> prcp_forc );
               sprintf(filename, "%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, public_xtra -> clm_metfile, "Temp", istep);
               ReadPFBinary( filename, instance_xtra -> tas_forc );
               sprintf(filename, "%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, public_xtra -> clm_metfile, "UGRD", istep);
               ReadPFBinary( filename, instance_xtra -> u_forc );
               sprintf(filename, "%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, public_xtra -> clm_metfile, "VGRD", istep);
               ReadPFBinary( filename, instance_xtra -> v_forc );
               sprintf(filename, "%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, public_xtra -> clm_metfile, "Press", istep);
               ReadPFBinary( filename, instance_xtra -> patm_forc );
               sprintf(filename, "%s/%s.%s.%06d.pfb", public_xtra -> clm_metpath, public_xtra -> clm_metfile, "SPFH", istep);
               ReadPFBinary( filename, instance_xtra -> qatm_forc );
            }  //end if/else (clm_metsub==True)
         }  //end if (clm_metforce==2)         

         /* IMF: If 3D met forcing... */
         if ( public_xtra -> clm_metforce == 3 )
         {
            // Calculate z-index in forcing vars corresponding to istep
            fstep  = ( (istep-1) % public_xtra -> clm_metnt );         // index w/in met vars corresponding to istep

            // Read input files... *IF* istep is a multiple of clm_metnt
            //                     *OR* file hasn't been read yet (fflag==0)
            if ( fstep==0 || fflag==0 )
            {
             
               //Figure out which file to read (i.e., calculate correct file time-stamps)
               if ( fflag==0 )
               {
                  fflag   = 1;
                  fstart  = (public_xtra -> clm_istep_start) - fstep;  // first time value in 3D met file names
                  fstop   = fstart - 1 + public_xtra -> clm_metnt;     // second time value in 3D met file names
               }
               else
               {
                  fstart  = istep;                                     // forst time value in 3D met file names
                  fstop   = fstart - 1 + public_xtra -> clm_metnt;     // second value in 3D met file names
               }  // end if fflag==0

               // Subdirectories for each variable?
               if ( public_xtra -> clm_metsub )
               {
                  sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, "DSWR", \
                          public_xtra -> clm_metfile, "DSWR", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> sw_forc );
                  sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, "DLWR", \
                          public_xtra -> clm_metfile, "DLWR", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> lw_forc );
                  sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, "APCP", \
                          public_xtra -> clm_metfile, "APCP", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> prcp_forc );
                  sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, "Temp", \
                          public_xtra -> clm_metfile, "Temp", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> tas_forc );
                  sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, "UGRD", \
                          public_xtra -> clm_metfile, "UGRD", fstart, fstop);
                  ReadPFBinary( filename, instance_xtra -> u_forc );
                  sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, "VGRD", \
                          public_xtra -> clm_metfile, "VGRD", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> v_forc );
                  sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, "Press", \
                          public_xtra -> clm_metfile, "Press", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> patm_forc );
                  sprintf(filename, "%s/%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, "SPFH", \
                          public_xtra -> clm_metfile, "SPFH", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> qatm_forc );
               }  
               else
               {
                  sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, \
                          public_xtra -> clm_metfile, "DSWR", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> sw_forc );
                  sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, \
                          public_xtra -> clm_metfile, "DLWR", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> lw_forc );
                  sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, \ 
                          public_xtra -> clm_metfile, "APCP", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> prcp_forc );
                  sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, \
                          public_xtra -> clm_metfile, "Temp", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> tas_forc );
                  sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, \
                          public_xtra -> clm_metfile, "UGRD", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> u_forc );
                  sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, \
                          public_xtra -> clm_metfile, "VGRD", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> v_forc );
                  sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, \
                          public_xtra -> clm_metfile, "Press", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> patm_forc );
                  sprintf(filename, "%s/%s.%s.%06d_to_%06d.pfb", public_xtra -> clm_metpath, \
                          public_xtra -> clm_metfile, "SPFH", fstart, fstop );
                  ReadPFBinary( filename, instance_xtra -> qatm_forc );
               }  // end if/else clm_metsub==False
             }  //end if (fstep==0)
         }  //end if (clm_metforce==3)

	 ForSubgridI(is, GridSubgrids(grid))
	 {
	    double        dx,dy,dz;
	    int           nx,ny,nz,nx_f,ny_f,nz_f,ip,ix,iy,iz; 
            int           x,y,z;

	    subgrid = GridSubgrid(grid, is);
	    p_sub = VectorSubvector(instance_xtra -> pressure, is);
	    s_sub  = VectorSubvector(instance_xtra -> saturation, is);
	    et_sub = VectorSubvector(evap_trans, is);
	    m_sub = VectorSubvector(instance_xtra -> mask, is);
	    po_sub = VectorSubvector(porosity, is);

            /* IMF: Subvectors -- CLM surface fluxes, SWE, t_grnd*/
	    eflx_lh_tot_sub    = VectorSubvector(instance_xtra -> eflx_lh_tot,is);
	    eflx_lwrad_out_sub = VectorSubvector(instance_xtra -> eflx_lwrad_out,is);
	    eflx_sh_tot_sub    = VectorSubvector(instance_xtra -> eflx_sh_tot,is);
	    eflx_soil_grnd_sub = VectorSubvector(instance_xtra -> eflx_soil_grnd,is);
	    qflx_evap_tot_sub  = VectorSubvector(instance_xtra -> qflx_evap_tot,is);
	    qflx_evap_grnd_sub = VectorSubvector(instance_xtra -> qflx_evap_grnd,is);
	    qflx_evap_soi_sub  = VectorSubvector(instance_xtra -> qflx_evap_soi,is);
	    qflx_evap_veg_sub  = VectorSubvector(instance_xtra -> qflx_evap_veg,is);
	    qflx_tran_veg_sub  = VectorSubvector(instance_xtra -> qflx_tran_veg,is);
	    qflx_infl_sub      = VectorSubvector(instance_xtra -> qflx_infl,is);
	    swe_out_sub        = VectorSubvector(instance_xtra -> swe_out,is);
	    t_grnd_sub         = VectorSubvector(instance_xtra -> t_grnd,is);
            tsoil_sub          = VectorSubvector(instance_xtra -> tsoil,is);
            irr_flag_sub       = VectorSubvector(instance_xtra -> irr_flag,is);
            qflx_qirr_sub      = VectorSubvector(instance_xtra -> qflx_qirr,is);
            qflx_qirr_inst_sub = VectorSubvector(instance_xtra -> qflx_qirr_inst,is);

            /* IMF: Subvectors -- CLM met forcings */	 
            sw_forc_sub        = VectorSubvector(instance_xtra -> sw_forc,is);
            lw_forc_sub        = VectorSubvector(instance_xtra -> lw_forc,is);
            prcp_forc_sub      = VectorSubvector(instance_xtra -> prcp_forc,is);
            tas_forc_sub       = VectorSubvector(instance_xtra -> tas_forc,is);
            u_forc_sub         = VectorSubvector(instance_xtra -> u_forc,is);
            v_forc_sub         = VectorSubvector(instance_xtra -> v_forc,is);
            patm_forc_sub      = VectorSubvector(instance_xtra -> patm_forc,is);
            qatm_forc_sub      = VectorSubvector(instance_xtra -> qatm_forc,is);
 
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
	    
	    sp = SubvectorData(s_sub);
	    pp = SubvectorData(p_sub);
	    et = SubvectorData(et_sub);
	    ms = SubvectorData(m_sub);
	    po_dat = SubvectorData(po_sub);

            /* IMF: Subvector Data -- CLM surface fluxes, SWE, t_grnd*/
	    eflx_lh            = SubvectorData(eflx_lh_tot_sub);
	    eflx_lwrad         = SubvectorData(eflx_lwrad_out_sub);
	    eflx_sh            = SubvectorData(eflx_sh_tot_sub);
	    eflx_grnd          = SubvectorData(eflx_soil_grnd_sub);
	    qflx_tot           = SubvectorData(qflx_evap_tot_sub);
	    qflx_grnd          = SubvectorData(qflx_evap_grnd_sub);
	    qflx_soi           = SubvectorData(qflx_evap_soi_sub);
	    qflx_eveg          = SubvectorData(qflx_evap_veg_sub);
	    qflx_tveg          = SubvectorData(qflx_tran_veg_sub);
	    qflx_in            = SubvectorData(qflx_infl_sub);
	    swe                = SubvectorData(swe_out_sub);
	    t_g                = SubvectorData(t_grnd_sub);
            t_soi              = SubvectorData(tsoil_sub);
            iflag              = SubvectorData(irr_flag_sub); 
            qirr               = SubvectorData(qflx_qirr_sub);
            qirr_inst          = SubvectorData(qflx_qirr_inst_sub);
 
            /* IMF: Subvector Data -- CLM met forcings */
            // 1D Case...
            if (public_xtra -> clm_metforce == 1)
            {
               // Grab SubvectorData's (still have init value)
               sw_data         = SubvectorData(sw_forc_sub);
               lw_data         = SubvectorData(lw_forc_sub);
               prcp_data       = SubvectorData(prcp_forc_sub);
               tas_data        = SubvectorData(tas_forc_sub);
               u_data          = SubvectorData(u_forc_sub);
               v_data          = SubvectorData(v_forc_sub);
               patm_data       = SubvectorData(patm_forc_sub);
               qatm_data       = SubvectorData(qatm_forc_sub);
               // Fill SubvectorData's w/ uniform forcinv values  
               for (n=0; n<((nx+2)*(ny+2)*3); n++)
               {
                  sw_data[n]   = sw;
                  lw_data[n]   = lw;
                  prcp_data[n] = prcp;
                  tas_data[n]  = tas;
                  u_data[n]    = u;
                  v_data[n]    = v;
                  patm_data[n] = patm;
                  qatm_data[n] = qatm;
               }
            }
            // 2D Case...
            if (public_xtra -> clm_metforce == 2)
            {
               // Just need to grab SubvectorData's
               sw_data         = SubvectorData(sw_forc_sub);
               lw_data         = SubvectorData(lw_forc_sub);
               prcp_data       = SubvectorData(prcp_forc_sub);
               tas_data        = SubvectorData(tas_forc_sub);
               u_data          = SubvectorData(u_forc_sub);
               v_data          = SubvectorData(v_forc_sub);
               patm_data       = SubvectorData(patm_forc_sub);
               qatm_data       = SubvectorData(qatm_forc_sub);
            }
            // 3D Case...
            if (public_xtra -> clm_metforce == 3)
            {
               // Determine bounds of correct time slice
               x               = SubvectorIX(sw_forc_sub);
               y               = SubvectorIY(sw_forc_sub);
               z               = fstep - 1;
               // Extract SubvectorElt 
               // (Array size is correct -- includes ghost nodes
               //  OK because ghost values not used by CLM)
               sw_data         = SubvectorElt(sw_forc_sub,x,y,z);
               lw_data         = SubvectorElt(lw_forc_sub,x,y,z);
               prcp_data       = SubvectorElt(prcp_forc_sub,x,y,z);
               tas_data        = SubvectorElt(tas_forc_sub,x,y,z);
               u_data          = SubvectorElt(u_forc_sub,x,y,z);
               v_data          = SubvectorElt(v_forc_sub,x,y,z);
               patm_data       = SubvectorElt(patm_forc_sub,x,y,z);
               qatm_data       = SubvectorElt(qatm_forc_sub,x,y,z);
            }
             
	    ip = SubvectorEltIndex(p_sub, ix, iy, iz);
	    switch (public_xtra -> lsm)
	    {
	       case 0:
	       {
		  // No LSM
		  break;
	       }
	       case 1:
	       {

		  clm_file_dir_length=strlen(public_xtra -> clm_file_dir);
		  CALL_CLM_LSM(pp,sp,et,ms,po_dat,istep,cdt,t,start_time, 
			       dx,dy,dz,ix,iy,nx,ny,nz,nx_f,ny_f,nz_f,ip,p,q,r,rank,
                               sw_data,lw_data,prcp_data,tas_data,u_data,v_data,patm_data,qatm_data,
                               eflx_lh,eflx_lwrad,eflx_sh,eflx_grnd,qflx_tot,qflx_grnd,
			       qflx_soi,qflx_eveg,qflx_tveg,qflx_in,swe,t_g,t_soi,
                               public_xtra -> clm_dump_interval, 
                               public_xtra -> clm_1d_out, 
                               public_xtra -> clm_file_dir, 
                               clm_file_dir_length, 
                               public_xtra -> clm_bin_out_dir, 
                               public_xtra -> write_CLM_binary,
                               public_xtra -> clm_beta_function,
                               public_xtra -> clm_veg_function,
                               public_xtra -> clm_veg_wilting,
                               public_xtra -> clm_veg_fieldc,
                               public_xtra -> clm_res_sat, 
                               public_xtra -> clm_irr_type, 
                               public_xtra -> clm_irr_cycle,
                               public_xtra -> clm_irr_rate, 
                               public_xtra -> clm_irr_start,
                               public_xtra -> clm_irr_stop, 
                               public_xtra -> clm_irr_threshold,
                               qirr, qirr_inst, iflag, 
                               public_xtra -> clm_irr_thresholdtype);

                  /* IMF Write Met to Silo (for testing) 
                  sprintf(file_postfix, "precip.%05d", instance_xtra -> file_number );
                  WriteSilo( file_prefix, file_postfix, instance_xtra -> prcp_forc,
                            t, instance_xtra -> file_number, "Precipitation");
                  sprintf(file_postfix, "air_temp.%05d", instance_xtra -> file_number );
                  WriteSilo( file_prefix, file_postfix, instance_xtra -> tas_forc,
                            t, instance_xtra -> file_number, "AirTemperature");
                  */

                  /* IMF Write CLM? */
                  if ( (instance_xtra -> iteration_number % (-(int)public_xtra -> clm_dump_interval)) == 0 )
		     {
		        public_xtra -> clm_dump_files = 1;
		     }
		  else 
		     {
		        public_xtra -> clm_dump_files = 0;
                     } 

                  /* IMF Write as silo? */
                  if ( public_xtra -> clm_dump_files && public_xtra -> write_silo_CLM )
                     {
                       sprintf(file_postfix, "eflx_lh_tot.%05d", instance_xtra -> file_number );
                       WriteSilo( file_prefix, file_postfix, instance_xtra -> eflx_lh_tot, 
                                 t, instance_xtra -> file_number, "LatentHeat");
			   
                       sprintf(file_postfix, "eflx_lwrad_out.%05d", instance_xtra -> file_number );
                       WriteSilo(file_prefix, file_postfix, instance_xtra -> eflx_lwrad_out, 
                                 t, instance_xtra -> file_number, "LongWave");
			   
                       sprintf(file_postfix, "eflx_sh_tot.%05d", instance_xtra -> file_number );
                       WriteSilo(file_prefix, file_postfix, instance_xtra -> eflx_sh_tot, 
                                 t, instance_xtra -> file_number, "SensibleHeat");
			   
                       sprintf(file_postfix, "eflx_soil_grnd.%05d", instance_xtra -> file_number );
                       WriteSilo(file_prefix, file_postfix, instance_xtra -> eflx_soil_grnd, 
                                 t, instance_xtra -> file_number, "GroundHeat");
			   
                       sprintf(file_postfix, "qflx_evap_tot.%05d", instance_xtra -> file_number );
                       WriteSilo(file_prefix, file_postfix, instance_xtra -> qflx_evap_tot, 
                                 t, instance_xtra -> file_number, "EvaporationTotal");
			   
                       sprintf(file_postfix, "qflx_evap_grnd.%05d", instance_xtra -> file_number );
                       WriteSilo(file_prefix, file_postfix, instance_xtra -> qflx_evap_grnd, 
                                 t, instance_xtra -> file_number, "EvaporationGroundNoSublimation");
			   
                       sprintf(file_postfix, "qflx_evap_soi.%05d", instance_xtra -> file_number );
                       WriteSilo(file_prefix, file_postfix, instance_xtra -> qflx_evap_soi, 
                                 t, instance_xtra -> file_number, "EvaporationGround");
			   
                       sprintf(file_postfix, "qflx_evap_veg.%05d", instance_xtra -> file_number );
                       WriteSilo(file_prefix, file_postfix, instance_xtra -> qflx_evap_veg, 
                                 t, instance_xtra -> file_number, "EvaporationCanopy");
			   
                       sprintf(file_postfix, "qflx_tran_veg.%05d", instance_xtra -> file_number );
                       WriteSilo(file_prefix, file_postfix, instance_xtra -> qflx_tran_veg, 
                                 t, instance_xtra -> file_number, "Transpiration");
			   
                       sprintf(file_postfix, "qflx_infl.%05d", instance_xtra -> file_number );
                       WriteSilo(file_prefix, file_postfix, instance_xtra -> qflx_infl, 
                                 t, instance_xtra -> file_number, "Infiltration");
			   
                       sprintf(file_postfix, "swe_out.%05d", instance_xtra -> file_number );
                       WriteSilo(file_prefix, file_postfix, instance_xtra -> swe_out, 
                                 t, instance_xtra -> file_number, "SWE");
			   
                       sprintf(file_postfix, "t_grnd.%05d", instance_xtra -> file_number );
                       WriteSilo(file_prefix, file_postfix, instance_xtra -> t_grnd, 
                                 t, instance_xtra -> file_number, "TemperatureGround");

                       sprintf(file_postfix, "t_soil.%05d", instance_xtra -> file_number );
                       WriteSilo(file_prefix, file_postfix, instance_xtra -> tsoil,
                                 t, instance_xtra -> file_number, "TemperatureSoil");

                       // IMF: irrigation flag -- 1.0 when irrigated, 0.0 when not irrigated
//                       if ( public_xtra -> clm_irr_type == 1 || public_xtra -> clm_irr_type == 2 )
//                          {
//                            sprintf(file_postfix, "qflx_qirr.%05d" ,instance_xtra -> file_number );
//                            WriteSilo(file_prefix, file_postfix, instance_xtra -> qflx_qirr,
//                            t, instance_xtra -> file_number, "IrrigationSurface");
//                          }

                       // IMF: irrigation applied to surface -- spray or drip
                       if ( public_xtra -> clm_irr_type == 1 || public_xtra -> clm_irr_type == 2 )
                          {
                            sprintf(file_postfix, "qflx_qirr.%05d" ,instance_xtra -> file_number );
                            WriteSilo(file_prefix, file_postfix, instance_xtra -> qflx_qirr, 
                            t, instance_xtra -> file_number, "IrrigationSurface");
                          }
                       
                       // IMF: irrigation applied directly as soil moisture flux -- "instant"
                       if ( public_xtra -> clm_irr_type == 3 )
                          {
                            sprintf(file_postfix, "qflx_qirr_inst.%05d" ,instance_xtra -> file_number );
                            WriteSilo(file_prefix, file_postfix, instance_xtra -> qflx_qirr_inst,
                            t, instance_xtra -> file_number, "IrrigationInstant");
                          }

                     } // end of write silo 
		  break;		  
	       }
	       default:
	       {
		  amps_Printf("Calling unknown LSM model");
	       }
            
	    } /* switch on LSM */
	    
	 }
	 handle = InitVectorUpdate(evap_trans, VectorUpdateAll);
	 FinalizeVectorUpdate(handle); 

         istep  = istep + 1;
	 EndTiming(CLMTimingIndex);

#endif   //End of call to CLM

      } //Endif to check whether an entire dt is complete

      converged = 1;
      conv_failures = 0;

      do  /* while not converged */
      {

	 /*
	   Record amount of memory in use.
	 */

	 recordMemoryInfo();

	 /*******************************************************************/
	 /*                  Compute time step                              */
	 /*******************************************************************/
	 if (converged)
	 {
	    if(time_step_control) {
	       PFModuleInvokeType(SelectTimeStepInvoke, time_step_control, (&dt, &dt_info, t, problem,
									    problem_data) );
	    } else {
	       PFModuleInvokeType(SelectTimeStepInvoke, select_time_step, (&dt, &dt_info, t, problem,
									   problem_data) );
	    }

	    PFVCopy(instance_xtra -> density,    instance_xtra -> old_density);
	    PFVCopy(instance_xtra -> saturation, instance_xtra -> old_saturation);
	    PFVCopy(instance_xtra -> pressure,   instance_xtra -> old_pressure);
	 }
	 else  /* Not converged, so decrease time step */
	 {
	    t = t - dt;
	    dt = 0.5 * dt;
	    PFVCopy(instance_xtra -> old_density,    instance_xtra -> density);
	    PFVCopy(instance_xtra -> old_saturation, instance_xtra -> saturation);
	    PFVCopy(instance_xtra -> old_pressure,   instance_xtra -> pressure);
	 }

#ifdef HAVE_CLM
	 /*
	  * Force timestep to LSM model if we are trying to advance beyond 
	  * LSM timesteping.
	  */
	 switch (public_xtra -> lsm)
	 {
	    case 0:
	    {
	       // No LSM
	       break;
	    }
	    case 1:
	    {
	       printf("SGS t=%f, dt=%f, ct=%f, cdt=%f\n", t, dt, ct, cdt);
	       // Note ct is time we want to advance to at this point
	       if ( t + dt > ct) {
		  dt = ct - t;
	       }
	       break;		  
	    }
	    default:
	    {
	       amps_Printf("Calling unknown LSM model");
	    }
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

	 dump_files = 0;
	 if ( dump_interval > 0 )
	 {
	    print_dt = ProblemStartTime(problem) +  instance_xtra -> dump_index*dump_interval - t;

	    if ( (dt + EPSILON) > print_dt )
	    {
	       /*
		* if the difference is small don't try to compute
		* at print_dt, just use dt.  This will
		* output slightly off in time but avoids
		* extremely small dt values.
		*/
	       if( fabs(dt - print_dt) > EPSILON) {
		  dt = print_dt;
	       }
	       dt_info = 'p';

	       dump_files = 1;
	    }
	 }
	 else if (dump_interval < 0)
	 {
	    if ( (instance_xtra -> iteration_number % (-(int)dump_interval)) == 0 )
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
	 if ( (t + dt) >= stop_time )
	 {   
	    dt = stop_time - t;
	    dt_info = 'f';
	 }
         
	 t += dt;

	 /*******************************************************************/
	 /*          Solve the nonlinear system for this time step          */
	 /*******************************************************************/
	  
	 retval = PFModuleInvokeType(NonlinSolverInvoke, nonlin_solver, 
				     (instance_xtra -> pressure, 
				      instance_xtra -> density, 
				      instance_xtra -> old_density, 
				      instance_xtra -> saturation, 
				      instance_xtra -> old_saturation, 
				      t, dt, 
				      problem_data, instance_xtra -> old_pressure, 
				      &outflow, evap_trans, 
				      instance_xtra -> ovrl_bc_flx));

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
	    if(!amps_Rank(amps_CommWorld))
	    { 
	       amps_Printf("Error: Time step failed for time %12.4e.\n", t);
	       amps_Printf("Shutting down.\n");
	    }
	 }

      }  /* Ends do for convergence of time step loop */
      while ( (!converged) && (conv_failures < max_failures) );

      instance_xtra -> iteration_number++;
     
      /* Calculate densities and saturations for the new pressure. */
      PFModuleInvokeType(PhaseDensityInvoke,  phase_density, 
		     (0, instance_xtra -> pressure, instance_xtra -> density, 
		      &dtmp, &dtmp, CALCFCN));
      handle = InitVectorUpdate(instance_xtra -> density, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      PFModuleInvokeType(SaturationInvoke, problem_saturation, 
                     (instance_xtra -> saturation, instance_xtra -> pressure, 
		      instance_xtra -> density, gravity, problem_data,
		      CALCFCN));

      any_file_dumped = 0;

      /***************************************************************
       * Compute running sum of evap trans for water balance 
       **************************************************************/
      if(public_xtra -> write_silo_evaptrans_sum) {
	 EvapTransSum(problem_data, dt, evap_trans_sum, evap_trans);
      }

      /***************************************************************
       * Compute running sum of overland outflow for water balance 
       **************************************************************/
      if(public_xtra -> write_silo_overland_sum) {
	 OverlandSum(problem_data, 
		     instance_xtra -> pressure,
		     dt, 
		     instance_xtra -> overland_sum);
      }

      /***************************************************************/
      /*                 Print the pressure and saturation           */
      /***************************************************************/

      /* Dump the pressure, saturation, surface fluxes at this time-step */
      if ( dump_files )
      {
	 instance_xtra -> dump_index++;
			
	 if(public_xtra -> print_press) {
	    sprintf(file_postfix, "press.%05d", instance_xtra -> file_number);
	    WritePFBinary(file_prefix, file_postfix, instance_xtra -> pressure);
	    any_file_dumped = 1;
	 }

	 if(public_xtra -> write_silo_press) 
	 {
	    sprintf(file_postfix, "press.%05d", instance_xtra -> file_number);
	    WriteSilo(file_prefix, file_postfix, instance_xtra -> pressure,
                      t, instance_xtra -> file_number, "Pressure");
	    any_file_dumped = 1;
	 }
 	
	 if( print_satur ) {
	    sprintf(file_postfix, "satur.%05d", instance_xtra -> file_number );
	    WritePFBinary(file_prefix, file_postfix, instance_xtra -> saturation );
	    any_file_dumped = 1;
	 }

	 if(public_xtra -> write_silo_satur) 
	 {
	    sprintf(file_postfix, "satur.%05d", instance_xtra -> file_number );
	    WriteSilo(file_prefix, file_postfix, instance_xtra -> saturation, 
                      t, instance_xtra -> file_number, "Saturation");
	    any_file_dumped = 1;
	 }

	 if(public_xtra -> write_silo_evaptrans) {
	    sprintf(file_postfix, "evaptrans.%05d", instance_xtra -> file_number );
	    WriteSilo(file_prefix, file_postfix, evap_trans, 
		      t, instance_xtra -> file_number, "EvapTrans");
	    any_file_dumped = 1;
	 }

	 if(public_xtra -> write_silo_evaptrans_sum) {
	    sprintf(file_postfix, "evaptranssum.%05d", instance_xtra -> file_number );
	    WriteSilo(file_prefix, file_postfix, evap_trans_sum, 
		      t, instance_xtra -> file_number, "EvapTransSum");
	    any_file_dumped = 1;
    
	    /* reset sum after output */
	    PFVConstInit(0.0, evap_trans_sum);
	 }

	 if(public_xtra -> write_silo_overland_sum) {
	    sprintf(file_postfix, "overlandsum.%05d", instance_xtra -> file_number );
	    WriteSilo(file_prefix, file_postfix, overland_sum, 
		      t, instance_xtra -> file_number, "OverlandSum");
	    any_file_dumped = 1;
	    
	    /* reset sum after output */
	    PFVConstInit(0.0, overland_sum);
	 }

	 if(public_xtra -> print_lsm_sink) 
	 {
	    /*sk Print the sink terms from the land surface model*/
	    sprintf(file_postfix, "et.%05d", instance_xtra -> file_number );
	    WritePFBinary(file_prefix, file_postfix, evap_trans);

	    /*sk Print the sink terms from the land surface model*/
	    sprintf(file_postfix, "obf.%05d", instance_xtra -> file_number );
	    WritePFBinary(file_prefix, file_postfix, instance_xtra -> ovrl_bc_flx);
	    any_file_dumped = 1;
	 }
      }

      /***************************************************************/
      /*             Compute the l2 error                            */
      /***************************************************************/

      PFModuleInvokeType(L2ErrorNormInvoke, l2_error_norm,
		     (t, instance_xtra -> pressure, problem_data, &err_norm));
      if( (!amps_Rank(amps_CommWorld)) && (err_norm >= 0.0) )
      {
	 amps_Printf("l2-error in pressure: %20.8e\n", err_norm);
	 amps_Printf("tcl: set pressure_l2_error(%d) %20.8e\n", 
		     instance_xtra -> iteration_number, err_norm);
	 fflush(NULL);
      }

      /*******************************************************************/
      /*                   Print the Well Data                           */
      /*******************************************************************/

      if ( print_wells && dump_files )
      {
	 WriteWells(file_prefix,
		    problem,
		    ProblemDataWellData(problem_data),
		    t, 
		    WELLDATA_DONTWRITEHEADER);
      }

      /*-----------------------------------------------------------------
       * Log this step
       *-----------------------------------------------------------------*/

      IfLogging(1)
      {
	 /*
	  * SGS Better error handing should be added 
	  */
	 if(instance_xtra -> number_logged > public_xtra -> max_iterations + 1) {
	    amps_Printf("Error: max_iterations reached, can't log anymore data\n");
	    exit(1);
	 }

	 instance_xtra -> seq_log[instance_xtra -> number_logged]       = instance_xtra -> iteration_number;
	 instance_xtra -> time_log[instance_xtra -> number_logged]      = t;
	 instance_xtra -> dt_log[instance_xtra -> number_logged]        = dt;
	 instance_xtra -> dt_info_log[instance_xtra -> number_logged]   = dt_info;
	 instance_xtra -> outflow_log[instance_xtra -> number_logged]   = outflow;
	 if ( any_file_dumped )
	    instance_xtra -> dumped_log[instance_xtra -> number_logged] = instance_xtra -> file_number;
	 else
	    instance_xtra -> dumped_log[instance_xtra -> number_logged] = -1;
	 instance_xtra -> recomp_log[instance_xtra -> number_logged] = 'y';
	 instance_xtra -> number_logged++;
      }

      if ( any_file_dumped ) 
	 instance_xtra -> file_number++;

      if (take_more_time_steps) {
	 take_more_time_steps = (instance_xtra -> iteration_number < max_iterations) &&
	    (t < stop_time);
      }

   }   /* ends do for time loop */
   while( take_more_time_steps );

   /***************************************************************/
   /*                 Print the pressure and saturation           */
   /***************************************************************/

   /* Dump the pressure values at end if requested */
   if( ProblemDumpAtEnd(problem) )
   {
      if(public_xtra -> print_press) {
	 sprintf(file_postfix, "press.%05d", instance_xtra -> file_number);
	 WritePFBinary(file_prefix, file_postfix, instance_xtra -> pressure);
	 any_file_dumped = 1;
      }
      
      if(public_xtra -> write_silo_press) 
      {
	 sprintf(file_postfix, "press.%05d", instance_xtra -> file_number);
	 WriteSilo(file_prefix, file_postfix, instance_xtra -> pressure,
		   t, instance_xtra -> file_number, "Pressure");
	 any_file_dumped = 1;
      }
      
      if( print_satur ) {
	 sprintf(file_postfix, "satur.%05d", instance_xtra -> file_number );
	 WritePFBinary(file_prefix, file_postfix, instance_xtra -> saturation );
	 any_file_dumped = 1;
      }
      
      if(public_xtra -> write_silo_satur) 
      {
	 sprintf(file_postfix, "satur.%05d", instance_xtra -> file_number );
	 WriteSilo(file_prefix, file_postfix, instance_xtra -> saturation, 
		   t, instance_xtra -> file_number, "Saturation");
	 any_file_dumped = 1;
      }

      if(public_xtra -> write_silo_evaptrans) {
	 sprintf(file_postfix, "evaptrans.%05d", instance_xtra -> file_number );
	 WriteSilo(file_prefix, file_postfix, evap_trans, 
		   t, instance_xtra -> file_number, "EvapTrans");
	 any_file_dumped = 1;
      }

      if(public_xtra -> write_silo_evaptrans_sum) {
	 sprintf(file_postfix, "evaptranssum.%05d", instance_xtra -> file_number );
	 WriteSilo(file_prefix, file_postfix, evap_trans_sum, 
		   t, instance_xtra -> file_number, "EvapTransSum");
	 any_file_dumped = 1;
	 /* reset sum after output */
	 PFVConstInit(0.0, evap_trans_sum);
      }

      if(public_xtra -> write_silo_overland_sum) {
	 sprintf(file_postfix, "overlandsum.%05d", instance_xtra -> file_number );
	 WriteSilo(file_prefix, file_postfix, overland_sum, 
		   t, instance_xtra -> file_number, "OverlandSum");
	 any_file_dumped = 1;
	 
	 /* reset sum after output */
	 PFVConstInit(0.0, overland_sum);
      }

      if(public_xtra -> print_lsm_sink) 
      {
	 /*sk Print the sink terms from the land surface model*/
	 sprintf(file_postfix, "et.%05d", instance_xtra -> file_number );
	 WritePFBinary(file_prefix, file_postfix, evap_trans);
	 
	 /*sk Print the sink terms from the land surface model*/
	 sprintf(file_postfix, "obf.%05d", instance_xtra -> file_number );
	 WritePFBinary(file_prefix, file_postfix, instance_xtra -> ovrl_bc_flx);
	 
	 any_file_dumped = 1;
      }
   }

   *pressure_out =   instance_xtra -> pressure;
   *porosity_out =   porosity;
   *saturation_out = instance_xtra -> saturation;
}


void TeardownRichards(PFModule *this_module) {
   PublicXtra    *public_xtra      = (PublicXtra *)PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra    = (InstanceXtra *)PFModuleInstanceXtra(this_module);

   Problem      *problem             = (public_xtra -> problem);
   ProblemData  *problem_data        = (instance_xtra -> problem_data);


   int           start_count         = ProblemStartCount(problem);

   FreeVector( instance_xtra -> saturation );
   FreeVector( instance_xtra -> density );
   FreeVector( instance_xtra -> old_saturation );
   FreeVector( instance_xtra -> old_pressure );
   FreeVector( instance_xtra -> old_density );
   FreeVector( instance_xtra -> pressure );
   FreeVector( instance_xtra -> ovrl_bc_flx );
   FreeVector( instance_xtra -> mask );

   if(instance_xtra -> evap_trans_sum) {
      FreeVector( instance_xtra -> evap_trans_sum);
   }

   if(instance_xtra -> overland_sum) {
      FreeVector(instance_xtra -> overland_sum);
   }

#ifdef HAVE_CLM   
   if(instance_xtra -> eflx_lh_tot) {
      FreeVector(instance_xtra -> eflx_lh_tot);
      FreeVector(instance_xtra -> eflx_lwrad_out);
      FreeVector(instance_xtra -> eflx_sh_tot);
      FreeVector(instance_xtra -> eflx_soil_grnd);
      FreeVector(instance_xtra -> qflx_evap_tot);
      FreeVector(instance_xtra -> qflx_evap_grnd);
      FreeVector(instance_xtra -> qflx_evap_soi);
      FreeVector(instance_xtra -> qflx_evap_veg);
      FreeVector(instance_xtra -> qflx_tran_veg);
      FreeVector(instance_xtra -> qflx_infl);
      FreeVector(instance_xtra -> swe_out);
      FreeVector(instance_xtra -> t_grnd);
      FreeVector(instance_xtra -> tsoil);

      /*IMF Initialize variables for CLM irrigation output */
      FreeVector(instance_xtra -> qflx_qirr);
      FreeVector(instance_xtra -> qflx_qirr_inst);
      /*IMF Initialize variables for CLM forcing fields
            SW rad, LW rad, precip, T(air), U, V, P(air), q(air) */
      FreeVector(instance_xtra -> sw_forc);
      FreeVector(instance_xtra -> lw_forc);
      FreeVector(instance_xtra -> prcp_forc);
      FreeVector(instance_xtra -> tas_forc);
      FreeVector(instance_xtra -> u_forc);
      FreeVector(instance_xtra -> v_forc);
      FreeVector(instance_xtra -> patm_forc);
      FreeVector(instance_xtra -> qatm_forc);
   }
#endif

   if(!amps_Rank(amps_CommWorld))
   {
      PrintWellData(ProblemDataWellData(problem_data), (WELLDATA_PRINTSTATS));
   }

   /*-----------------------------------------------------------------------
    * Print log
    *-----------------------------------------------------------------------*/

   IfLogging(1)
   {
      FILE*  log_file;
      int        k;

      log_file = OpenLogFile("SolverRichards");

      if ( start_count >= 0 )
      {
	 fprintf(log_file, "Transient Problem Solved.\n");
	 fprintf(log_file, "-------------------------\n");
	 fprintf(log_file, "Sequence #       Time         \\Delta t         Dumpfile #   Recompute?\n");
	 fprintf(log_file, "----------   ------------   ------------ -     ----------   ----------\n");

	 for (k = 0; k < instance_xtra -> number_logged; k++)
	 {
	    if ( instance_xtra -> dumped_log[k] == -1 )
	       fprintf(log_file, "  %06d     %8e   %8e %1c                       %1c\n",
		       k, instance_xtra -> time_log[k], instance_xtra -> dt_log[k], instance_xtra -> dt_info_log[k], instance_xtra -> recomp_log[k]);
	    else
	       fprintf(log_file, "  %06d     %8e   %8e %1c       %06d          %1c\n",
		       k, instance_xtra -> time_log[k], instance_xtra -> dt_log[k], instance_xtra -> dt_info_log[k], instance_xtra -> dumped_log[k], instance_xtra -> recomp_log[k]);
	 }

	 fprintf(log_file, "\n");
	 fprintf(log_file, "Overland flow Results\n");
	 /*fprintf(log_file, "-------------------------\n");
	   fprintf(log_file, "Sequence #       Time         \\Delta t           Outflow [L/T]\n");
	   fprintf(log_file, "----------   ------------   --------------       -------------- \n");*/
	 fprintf(log_file, " %d\n",instance_xtra -> number_logged); 
	 for (k = 0; k < instance_xtra -> number_logged; k++) //sk start
	 {
	    if ( instance_xtra -> dumped_log[k] == -1 )
	       fprintf(log_file, "  %06d     %8e   %8e       %e\n",
		       k, instance_xtra -> time_log[k], instance_xtra -> dt_log[k], instance_xtra -> outflow_log[k]);
	    else
	       fprintf(log_file, "  %06d     %8e   %8e       %e\n",
		       k, instance_xtra -> time_log[k], instance_xtra -> dt_log[k], instance_xtra -> outflow_log[k]);
	 } //sk end
      }
      else
      {
	 fprintf(log_file, "Non-Transient Problem Solved.\n");
	 fprintf(log_file, "-----------------------------\n");
      }

      CloseLogFile(log_file);

      tfree(instance_xtra -> seq_log);
      tfree(instance_xtra -> time_log);
      tfree(instance_xtra -> dt_log);
      tfree(instance_xtra -> dt_info_log);
      tfree(instance_xtra -> dumped_log);
      tfree(instance_xtra -> recomp_log);
      tfree(instance_xtra -> outflow_log);
   }

}

/*--------------------------------------------------------------------------
 * SolverRichardsInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule *SolverRichardsInitInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = (PublicXtra *)PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra;

   Problem      *problem = (public_xtra -> problem);

   Grid         *grid;
   Grid         *grid2d;
   Grid         *x_grid;
   Grid         *y_grid;
   Grid         *z_grid;
#ifdef HAVE_CLM
   Grid         *gridTs;
   Grid         *metgrid;
#endif

   SubgridArray *new_subgrids;
   SubgridArray *all_subgrids, *new_all_subgrids;
   Subgrid      *subgrid, *new_subgrid;
   double       *temp_data, *temp_data_placeholder;
   int           concen_sz, ic_sz, velocity_sz, temp_data_size, sz;
   int           nonlin_sz, parameter_sz;
   int           i;

   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);

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
   new_subgrids  = GetGridSubgrids(new_all_subgrids);
   grid2d        = NewGrid(new_subgrids, new_all_subgrids);
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
   new_subgrids  = GetGridSubgrids(new_all_subgrids);
   x_grid        = NewGrid(new_subgrids, new_all_subgrids);
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
   new_subgrids  = GetGridSubgrids(new_all_subgrids);
   y_grid        = NewGrid(new_subgrids, new_all_subgrids);
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
   new_subgrids  = GetGridSubgrids(new_all_subgrids);
   z_grid        = NewGrid(new_subgrids, new_all_subgrids);
   CreateComputePkgs(z_grid);

   (instance_xtra -> grid)   = grid;
   (instance_xtra -> grid2d) = grid2d;
   (instance_xtra -> x_grid) = x_grid;
   (instance_xtra -> y_grid) = y_grid;
   (instance_xtra -> z_grid) = z_grid;

#ifdef HAVE_CLM
   /* IMF New grid for met forcing (nx*ny*nt) */
   /* NT specified by key CLM.MetForcing3D.NT */
   all_subgrids = GridAllSubgrids(grid);
   new_all_subgrids = NewSubgridArray();
   ForSubgridI(i, all_subgrids)
   {
      subgrid = SubgridArraySubgrid(all_subgrids, i);
      new_subgrid = DuplicateSubgrid(subgrid);
      SubgridNZ(new_subgrid) = public_xtra -> clm_metnt; 
      AppendSubgrid(new_subgrid, new_all_subgrids);
   }
   new_subgrids  = GetGridSubgrids(new_all_subgrids);
   metgrid       = NewGrid(new_subgrids, new_all_subgrids);
   CreateComputePkgs(metgrid);
   (instance_xtra -> metgrid) = metgrid;

   /* IMF New grid for Tsoil (nx*ny*10) */
   all_subgrids = GridAllSubgrids(grid);
   new_all_subgrids = NewSubgridArray();
   ForSubgridI(i, all_subgrids)
   {
      subgrid = SubgridArraySubgrid(all_subgrids, i);
      new_subgrid = DuplicateSubgrid(subgrid);
      SubgridNZ(new_subgrid) = 10;
      AppendSubgrid(new_subgrid, new_all_subgrids);
   }
   new_subgrids  = GetGridSubgrids(new_all_subgrids);
   gridTs        = NewGrid(new_subgrids, new_all_subgrids);
   CreateComputePkgs(gridTs);
   (instance_xtra -> gridTs) = gridTs;
#endif

   /*-------------------------------------------------------------------
    * Create problem_data
    *-------------------------------------------------------------------*/

   (instance_xtra -> problem_data) = NewProblemData(grid,grid2d);
   
   /*-------------------------------------------------------------------
    * Initialize module instances
    *-------------------------------------------------------------------*/

   if ( PFModuleInstanceXtra(this_module) == NULL )
   {
      (instance_xtra -> phase_velocity_face) = NULL;
      /*	 PFModuleNewInstance((public_xtra -> phase_velocity_face),
                 (problem, grid, x_grid, y_grid, z_grid, NULL));
      */
      /* Need to change for rel. perm. and not mobility */
      (instance_xtra -> advect_concen) =
	 PFModuleNewInstanceType(AdvectionConcentrationInitInstanceXtraType,
				 (public_xtra -> advect_concen),
				 (problem, grid, NULL));
      (instance_xtra -> set_problem_data) =
	 PFModuleNewInstanceType(SetProblemDataInitInstanceXtraInvoke,
				 (public_xtra -> set_problem_data),
				 (problem, grid, grid2d, NULL));

      (instance_xtra -> retardation) =
	 PFModuleNewInstanceType(RetardationInitInstanceXtraInvoke,
				 ProblemRetardation(problem), (NULL));
      (instance_xtra -> phase_rel_perm) =
	 PFModuleNewInstanceType(PhaseRelPermInitInstanceXtraInvoke,
				 ProblemPhaseRelPerm(problem), (grid, NULL));
      (instance_xtra -> ic_phase_concen) =
	 PFModuleNewInstance(ProblemICPhaseConcen(problem), ());

      (instance_xtra -> permeability_face) =
	 PFModuleNewInstanceType(PermeabilityFaceInitInstanceXtraInvoke,
				 (public_xtra -> permeability_face),
				 (z_grid));
	
      (instance_xtra -> ic_phase_pressure) =
	 PFModuleNewInstanceType(ICPhasePressureInitInstanceXtraInvoke,
				 ProblemICPhasePressure(problem), 
				 (problem, grid, NULL));
      (instance_xtra -> problem_saturation) =
	 PFModuleNewInstanceType(SaturationInitInstanceXtraInvoke,
				 ProblemSaturation(problem), (grid, NULL));
      (instance_xtra -> phase_density) =
	 PFModuleNewInstance(ProblemPhaseDensity(problem), ());
      (instance_xtra -> select_time_step) =
	 PFModuleNewInstance(ProblemSelectTimeStep(problem), ());
      (instance_xtra -> l2_error_norm) =
	 PFModuleNewInstance(ProblemL2ErrorNorm(problem), ());
      (instance_xtra -> nonlin_solver) =
	 PFModuleNewInstanceType(NonlinSolverInitInstanceXtraInvoke,
				 public_xtra -> nonlin_solver, 
				 (problem, grid, instance_xtra -> problem_data, NULL));
      
   }
   else
   {
      PFModuleReNewInstanceType(PhaseVelocityFaceInitInstanceXtraInvoke,
				(instance_xtra -> phase_velocity_face),
				(problem, grid, x_grid, y_grid, z_grid, NULL));
      PFModuleReNewInstanceType(AdvectionConcentrationInitInstanceXtraType,
				(instance_xtra -> advect_concen),
				(problem, grid, NULL));
      PFModuleReNewInstanceType(SetProblemDataInitInstanceXtraInvoke,
				(instance_xtra -> set_problem_data),
				(problem, grid, grid2d, NULL));
      
      PFModuleReNewInstanceType(RetardationInitInstanceXtraInvoke,
				(instance_xtra -> retardation), (NULL));
      
      PFModuleReNewInstanceType(PhaseRelPermInitInstanceXtraInvoke,
				(instance_xtra -> phase_rel_perm), (grid, NULL));
      PFModuleReNewInstance((instance_xtra -> ic_phase_concen), ());
      
      PFModuleReNewInstanceType(PermeabilityFaceInitInstanceXtraInvoke,
				(instance_xtra -> permeability_face),
				(z_grid));
      
      PFModuleReNewInstanceType(ICPhasePressureInitInstanceXtraInvoke,
				(instance_xtra -> ic_phase_pressure), 
				(problem, grid, NULL));
      PFModuleReNewInstanceType(SaturationInitInstanceXtraInvoke,
				(instance_xtra -> problem_saturation), 
				(grid, NULL)); 
      PFModuleReNewInstance((instance_xtra -> phase_density), ()); 
      PFModuleReNewInstance((instance_xtra -> select_time_step), ()); 
      PFModuleReNewInstance((instance_xtra -> l2_error_norm), ()); 
      PFModuleReNewInstance((instance_xtra -> nonlin_solver), ()); 
   }
   
   /*-------------------------------------------------------------------
    * Set up temporary data
    *-------------------------------------------------------------------*/

   /* May need the temp_mobility size for something later... */

   //sk: I don't have to do this for my instcances, because I allocate memory locally ?!

   /* compute size for velocity computation */
   sz = 0;
   /*   sz = max(sz, PFModuleSizeOfTempData(instance_xtra -> phase_velocity_face)); */
   velocity_sz = sz;

   /* compute size for concentration advection */
   sz = 0;
   sz = max(sz, PFModuleSizeOfTempData(instance_xtra -> retardation));
   sz = max(sz, PFModuleSizeOfTempData(instance_xtra -> advect_concen));
   concen_sz = sz;

   /* compute size for pressure initial condition */
   ic_sz = PFModuleSizeOfTempData(instance_xtra -> ic_phase_pressure);

   /* compute size for initial pressure guess*/
   /*ig_sz = PFModuleSizeOfTempData(instance_xtra -> ig_phase_pressure);*/

   /* Compute size for nonlinear solver */
   nonlin_sz = PFModuleSizeOfTempData(instance_xtra -> nonlin_solver);

   /* Compute size for problem parameters */
   parameter_sz = PFModuleSizeOfTempData(instance_xtra -> problem_saturation)
      + PFModuleSizeOfTempData(instance_xtra -> phase_rel_perm);

   /* set temp_data size to max of velocity_sz, concen_sz, and ic_sz. */
   /* The temp vector space for the nonlinear solver is added in because */
   /* at a later time advection may need to re-solve flow. */
   temp_data_size = parameter_sz 
      + max(max(max(velocity_sz, concen_sz), nonlin_sz), ic_sz);

   /* allocate temporary data */
   temp_data = NewTempData(temp_data_size);
   (instance_xtra -> temp_data) = temp_data;

   PFModuleReNewInstanceType(SaturationInitInstanceXtraInvoke,
			     (instance_xtra -> problem_saturation),
			     (NULL, temp_data));
   temp_data += PFModuleSizeOfTempData(instance_xtra->problem_saturation);

   PFModuleReNewInstanceType(PhaseRelPermInitInstanceXtraInvoke,
			     (instance_xtra -> phase_rel_perm),
			     (NULL, temp_data));
   temp_data += PFModuleSizeOfTempData(instance_xtra->phase_rel_perm);

   /* renew ic_phase_pressure module */
   PFModuleReNewInstanceType(ICPhasePressureInitInstanceXtraInvoke,
			     (instance_xtra -> ic_phase_pressure),
			     (NULL, NULL, temp_data));

   /* renew nonlinear solver module */
   PFModuleReNewInstanceType(NonlinSolverInitInstanceXtraInvoke,
			     (instance_xtra -> nonlin_solver),
			     (NULL, NULL, instance_xtra -> problem_data, temp_data));

   /* renew set_problem_data module */
   PFModuleReNewInstanceType(SetProblemDataInitInstanceXtraInvoke,
			     (instance_xtra -> set_problem_data),
			     (NULL, NULL, NULL, temp_data));

   /* renew velocity computation modules that take temporary data */
   /*   PFModuleReNewInstance((instance_xtra -> phase_velocity_face),
	(NULL, NULL, NULL, NULL, NULL, temp_data)); */


   /* renew concentration advection modules that take temporary data */
   temp_data_placeholder = temp_data;
   PFModuleReNewInstanceType(RetardationInitInstanceXtraInvoke,
			     (instance_xtra -> retardation),
			     (temp_data_placeholder));
   PFModuleReNewInstanceType(AdvectionConcentrationInitInstanceXtraType,
			     (instance_xtra -> advect_concen),
			     (NULL, NULL, temp_data_placeholder));

   temp_data_placeholder += max(PFModuleSizeOfTempData(
				   instance_xtra -> retardation),
				PFModuleSizeOfTempData(
				   instance_xtra -> advect_concen));
   /* set temporary vector data used for advection */

   temp_data += temp_data_size;

   PFModuleInstanceXtra(this_module) = instance_xtra;

   return this_module;
}

/*--------------------------------------------------------------------------
 * SolverRichardsFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  SolverRichardsFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);

   if ( instance_xtra )
   {

      FreeTempData( (instance_xtra -> temp_data) );

      PFModuleFreeInstance((instance_xtra -> ic_phase_concen));
      PFModuleFreeInstance((instance_xtra -> phase_rel_perm));
      PFModuleFreeInstance((instance_xtra -> retardation));

      PFModuleFreeInstance((instance_xtra -> set_problem_data));
      PFModuleFreeInstance((instance_xtra -> advect_concen));
      if (instance_xtra -> phase_velocity_face)
	 PFModuleFreeInstance((instance_xtra -> phase_velocity_face));

      PFModuleFreeInstance((instance_xtra -> ic_phase_pressure));
      PFModuleFreeInstance((instance_xtra -> problem_saturation));
      PFModuleFreeInstance((instance_xtra -> phase_density));
      PFModuleFreeInstance((instance_xtra -> select_time_step));
      PFModuleFreeInstance((instance_xtra -> l2_error_norm));
      PFModuleFreeInstance((instance_xtra -> nonlin_solver));

      PFModuleFreeInstance((instance_xtra -> permeability_face));

      FreeProblemData((instance_xtra -> problem_data));

      FreeGrid((instance_xtra -> z_grid));
      FreeGrid((instance_xtra -> y_grid));
      FreeGrid((instance_xtra -> x_grid));
      FreeGrid((instance_xtra -> grid2d));
      FreeGrid((instance_xtra -> grid));

#ifdef HAVE_CLM
      FreeGrid((instance_xtra -> gridTs));
#endif

      tfree(instance_xtra);
   }
}

/*--------------------------------------------------------------------------
 * SolverRichardsNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *SolverRichardsNewPublicXtra(char *name)
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;
   
   char key[IDB_MAX_KEY_LEN];

   char          *switch_name;
   int            switch_value;
   NameArray      switch_na;
   NameArray      nonlin_switch_na;
   NameArray      lsm_switch_na;

#ifdef HAVE_CLM
   NameArray      beta_switch_na;
   NameArray      vegtype_switch_na;
   NameArray      metforce_switch_na;
   NameArray      irrtype_switch_na;
   NameArray      irrcycle_switch_na;
   NameArray      irrthresholdtype_switch_na;
#endif

   switch_na = NA_NewNameArray("False True");

   public_xtra = ctalloc(PublicXtra, 1);
   
   (public_xtra -> permeability_face) = 
      PFModuleNewModule(PermeabilityFace, ());
   (public_xtra -> phase_velocity_face) = NULL;
   /*     
	  PFModuleNewModule(PhaseVelocityFace, ());
   */
   /* Need to account for rel. perm. and not mobility */

   (public_xtra -> advect_concen) = PFModuleNewModule(Godunov, ());
   (public_xtra -> set_problem_data) = PFModuleNewModule(SetProblemData, ());
   
   (public_xtra -> problem) = NewProblem(RichardsSolve);

   nonlin_switch_na = NA_NewNameArray("KINSol");
   sprintf(key, "%s.NonlinearSolver", name);
   switch_name = GetStringDefault(key, "KINSol");
   switch_value = NA_NameToIndex(nonlin_switch_na, switch_name);
   switch (switch_value)
   {
      case 0:
      {
	 (public_xtra -> nonlin_solver) = 
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
	 public_xtra -> lsm = 0;
	 break;
      }
      case 1:
      {
#ifdef HAVE_CLM
	 public_xtra -> lsm = 1;
#else
         InputError("Error: <%s> used for key <%s> but this version of Parflow is compiled without CLM\n", switch_name, 
		    key);
#endif
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
#ifdef HAVE_CLM
   sprintf(key, "%s.CLM.CLMDumpInterval", name);
   public_xtra -> clm_dump_interval = GetIntDefault(key,1);

   sprintf(key, "%s.CLM.Print1dOut", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid print switch value <%s> for key <%s>\n",
		 switch_name, key );
   }
   public_xtra -> clm_1d_out = switch_value;

   sprintf(key, "%s.CLM.BinaryOutDir", name);
   switch_name = GetStringDefault(key, "True");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid print switch value <%s> for key <%s>\n",
		 switch_name, key );
   }
   public_xtra -> clm_bin_out_dir = switch_value;
	
   sprintf(key, "%s.CLM.CLMFileDir", name);
   public_xtra -> clm_file_dir = GetStringDefault(key,"");

   /* RMM added beta input function for clm */
   beta_switch_na = NA_NewNameArray("none Linear Cosine");
   sprintf(key, "%s.CLM.EvapBeta", name);
   switch_name = GetStringDefault(key, "Linear");
   switch_value = NA_NameToIndex(beta_switch_na, switch_name);
   switch (switch_value)
   {
        case 0:
        {
            public_xtra -> clm_beta_function = 0;
            break;
        }
        case 1:
        {
            public_xtra -> clm_beta_function = 1;
            break;
        }
        case 2:
        {
            public_xtra -> clm_beta_function = 2;
            break;
        }
        default:
        {
            InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
                       key);
        }
   }
   NA_FreeNameArray(beta_switch_na);

   sprintf(key, "%s.CLM.ResSat", name);
   public_xtra -> clm_res_sat = GetDoubleDefault(key, 0.1);

   /* RMM added veg sm stress input function for clm */
   vegtype_switch_na = NA_NewNameArray("none Pressure Saturation");
   sprintf(key, "%s.CLM.VegWaterStress", name);
   switch_name = GetStringDefault(key, "Saturation");
   switch_value = NA_NameToIndex(vegtype_switch_na, switch_name);
   switch (switch_value)
   {
        case 0:
        {
            public_xtra -> clm_veg_function = 0;
            break;
        }
        case 1:
        {
            public_xtra -> clm_veg_function = 1;
            break;
        }
        case 2:
        {
            public_xtra -> clm_veg_function = 2;
            break;
        }
        default:
        {
            InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
                       key);
        }
   }
   NA_FreeNameArray(vegtype_switch_na);

   sprintf(key, "%s.CLM.WiltingPoint", name);
   public_xtra -> clm_veg_wilting = GetDoubleDefault(key, 0.1);

   sprintf(key, "%s.CLM.FieldCapacity", name);
   public_xtra -> clm_veg_fieldc = GetDoubleDefault(key, 1.0);

   /* IMF Write CLM as Silo (default=False) */
   sprintf(key, "%s.WriteSiloCLM", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
                  switch_name, key );
   }
   public_xtra -> write_silo_CLM = switch_value;

   /* IMF Write CLM Binary (default=True) */
   sprintf(key, "%s.WriteCLMBinary", name);
   switch_name = GetStringDefault(key, "True");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
                 switch_name, key );
   }
   public_xtra -> write_CLM_binary = switch_value;

   /* IMF Key for CLM met file path */
   sprintf(key, "%s.CLM.MetFilePath", name);
   public_xtra -> clm_metpath = GetStringDefault(key, ".");

   /* IMF Key for met vars in subdirectories
      If True  -- each variable in it's own subdirectory of MetFilePath (e.g., /Temp, /APCP, etc.)
      If False -- all files in MetFilePath */
   sprintf(key, "%s.CLM.MetFileSubdir", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
                  switch_name, key );
   }
   public_xtra -> clm_metsub = switch_value;

   /* IMF Key for CLM met file name...
      for 1D forcing, is complete file name
      for 2D/3D forcing, is base file name (w/o timestep extension) */
   sprintf(key, "%s.CLM.MetFileName", name);
   public_xtra -> clm_metfile = GetStringDefault(key, "narr_1hr.sc3.txt");

   /* IMF Key for CLM istep (default=1) */
   sprintf(key, "%s.CLM.IstepStart", name);
   public_xtra -> clm_istep_start = GetIntDefault(key, 1);

   /* IMF Key for CLM fstep (default=1) */
   sprintf(key, "%s.CLM.FstepStart", name);
   public_xtra -> clm_fstep_start = GetIntDefault(key, 1);

   /* IMF Switch for 1D (uniform) vs. 2D (distributed) met forcings */
   /* IMF Added 3D option (distributed w/ time axis -- nx*ny*nz; nz=nt) */
   metforce_switch_na = NA_NewNameArray("none 1D 2D 3D");
   sprintf(key, "%s.CLM.MetForcing", name);
   switch_name = GetStringDefault(key, "none");
   switch_value = NA_NameToIndex(metforce_switch_na, switch_name);
   switch (switch_value)
   {
        case 0:
        {
            public_xtra -> clm_metforce = 0;
            printf("TEST: clm_metforce = 0 (%d) \n", public_xtra -> clm_metforce );
            break;
        }
        case 1:
        {
            public_xtra -> clm_metforce = 1;
            printf("TEST: clm_metforce = 1 (%d) \n", public_xtra -> clm_metforce );
            break;
        }
        case 2:
        {
            public_xtra -> clm_metforce = 2;
            printf("TEST: clm_metforce = 2 (%d) \n", public_xtra -> clm_metforce );
            break;
        }
        case 3:
        {
            public_xtra -> clm_metforce = 3;
            printf("TEST: clm_metforce = 3 (%d) \n", public_xtra -> clm_metforce );
            break;
        }
        default:
        {
            InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
                       key);
        }
   }
   NA_FreeNameArray(metforce_switch_na);

   /* IMF added key for nt of 3D met files */
   sprintf(key, "%s.CLM.MetFileNT", name);
   public_xtra -> clm_metnt = GetIntDefault(key, 1);

   /* IMF added irrigation type, rate, value keys for irrigating in CLM */
   /* IrrigationType -- none, Drip, Spray, Instant (default == none) */
   irrtype_switch_na = NA_NewNameArray("none Spray Drip Instant");
   sprintf(key, "%s.CLM.IrrigationType", name);
   switch_name = GetStringDefault(key, "none");
   switch_value = NA_NameToIndex(irrtype_switch_na, switch_name);
   switch (switch_value)
   {
        case 0:     // None
        {
            public_xtra -> clm_irr_type = 0;
            break;
        }
        case 1:     // Spray
        {
            public_xtra -> clm_irr_type = 1;
            break;
        }
        case 2:     // Drip
        {
            public_xtra -> clm_irr_type = 2;
            break;
        }
        case 3:     // Instant
        {
            public_xtra -> clm_irr_type = 3;
            break;
        }
        default:
        {
            InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
                       key);
        }
   }
   NA_FreeNameArray(irrtype_switch_na);

   /* IrrigationCycle -- Constant, Deficit (default == Deficit) */
   /* (Constant = irrigate based on specified time cycle [IrrigationStartTime,IrrigationEndTime]; 
       Deficit  = irrigate based on soil moisture criteria [IrrigationDeficit]) */
   irrcycle_switch_na = NA_NewNameArray("Constant Deficit");
   sprintf(key, "%s.CLM.IrrigationCycle", name);
   switch_name = GetStringDefault(key, "Constant");
   switch_value = NA_NameToIndex(irrcycle_switch_na, switch_name);
   switch (switch_value)
   {
        case 0:
        {
            public_xtra -> clm_irr_cycle = 0;
            break;
        }
        case 1:
        {
            public_xtra -> clm_irr_cycle = 1;
            break;
        }
        default:
        {
            InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
                       key);
        }
   }
   NA_FreeNameArray(irrcycle_switch_na);

   /* IrrigationValue -- Application rate for Drip or Spray irrigation */ 
   sprintf(key, "%s.CLM.IrrigationRate", name);
   public_xtra -> clm_irr_rate = GetDoubleDefault(key,0.0);

   /* IrrigationStartTime -- Start time of daily irrigation if IrrigationCycle == Constant */
   /* IrrigationStopTime  -- Stop time of daily irrigation if IrrigationCycle == Constant  */
   /* Default == start @ 12:00gmt (7am in central US), end @ 20:00gmt (3pm in central US)  */
   /* NOTE: Times in GMT */
   sprintf(key, "%s.CLM.IrrigationStartTime", name);
   public_xtra -> clm_irr_start = GetDoubleDefault(key,12.0);
   sprintf(key, "%s.IrrigationStopTime", name);
   public_xtra -> clm_irr_stop = GetDoubleDefault(key,20.0);

   /* IrrigationThreshold -- Soil moisture threshold for irrigation if IrrigationCycle == Deficit */
   /* CLM applies irrigation whenever soil moisture < threshold */
   sprintf(key, "%s.CLM.IrrigationThreshold", name);
   public_xtra -> clm_irr_threshold = GetDoubleDefault(key,0.5);

   /* IrrigationThresholdType -- Soil moisture threshold for irrigation if IrrigationCycle == Deficit */
   /* Specifies where saturation comparison is made -- top layer, bottom layer, average over column */
   irrthresholdtype_switch_na = NA_NewNameArray("Top Bottom Column");
   sprintf(key, "%s.CLM.IrrigationThresholdType", name);
   switch_name = GetStringDefault(key, "Column");
   switch_value = NA_NameToIndex(irrthresholdtype_switch_na, switch_name);
   switch (switch_value)
   {
        case 0:
        {
            public_xtra -> clm_irr_thresholdtype = 0;    
            break;
        }
        case 1:
        {
            public_xtra -> clm_irr_thresholdtype = 1;
            break;
        }
        case 2:
        {
            public_xtra -> clm_irr_thresholdtype = 2;
            break;
        }
        default:
        {
            InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
                       key);
        }
   }
   NA_FreeNameArray(irrthresholdtype_switch_na);

#endif

   sprintf(key, "%s.MaxIter", name);
   public_xtra -> max_iterations = GetIntDefault(key, 1000000);

   sprintf(key, "%s.MaxConvergenceFailures", name);
   public_xtra -> max_convergence_failures = GetIntDefault(key,3);

   if (public_xtra -> max_convergence_failures > 9) 
   {
      amps_Printf("Warning: Input variable <%s> \n", key);
      amps_Printf("         is set to a large value that may cause problems\n");
      amps_Printf("         with how time cycles calculations are evaluated.  Values\n");
      amps_Printf("         specified via a time cycle may be on/off at the slightly\n"); 
      amps_Printf("         wrong times times due to how Parflow discretizes time.\n");
   }

   sprintf(key, "%s.AdvectOrder", name);
   public_xtra -> advect_order = GetIntDefault(key,2);

   sprintf(key, "%s.CFL", name);
   public_xtra -> CFL = GetDoubleDefault(key, 0.7);

   sprintf(key, "%s.DropTol", name);
   public_xtra -> drop_tol = GetDoubleDefault(key, 1E-8);

   sprintf(key, "%s.PrintSubsurfData", name);
   switch_name = GetStringDefault(key, "True");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid print switch value <%s> for key <%s>\n",
		 switch_name, key);
   }
   public_xtra -> print_subsurf_data = switch_value;

   sprintf(key, "%s.PrintPressure", name);
   switch_name = GetStringDefault(key, "True");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid print switch value <%s> for key <%s>\n",
		 switch_name, key );
   }
   public_xtra -> print_press = switch_value;

   sprintf(key, "%s.PrintVelocities", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid print switch value <%s> for key <%s>\n",
		 switch_name, key );
   }
   public_xtra -> print_velocities = switch_value;

   sprintf(key, "%s.PrintSaturation", name);
   switch_name = GetStringDefault(key, "True");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid print switch value <%s> for key <%s>\n",
		 switch_name, key);
   }
   public_xtra -> print_satur = switch_value;

   sprintf(key, "%s.PrintConcentration", name);
   switch_name = GetStringDefault(key, "True");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid print switch value <%s> for key <%s>\n",
		 switch_name, key );
   }
   public_xtra -> print_concen = switch_value;

   sprintf(key, "%s.PrintWells", name);
   switch_name = GetStringDefault(key, "True");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid print switch value <%s> for key <%s>\n",
		 switch_name, key);
   }
   public_xtra -> print_wells = switch_value;

   // SGS TODO
   // Need to add this to the user manual, this is new for LSM stuff that was added.
   sprintf(key, "%s.PrintLSMSink", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid print switch value <%s> for key <%s>\n",
		 switch_name, key);
   }
   public_xtra -> print_lsm_sink = switch_value;

#ifndef HAVE_CLM
   if(public_xtra -> print_lsm_sink) 
   {
      InputError("Error: setting PrintLSMSink but do not have CLM\n", switch_name, key);
   }
#endif

   /* Silo file writing control */
   sprintf(key, "%s.WriteSiloSubsurfData", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
		 switch_name, key);
   }
   public_xtra -> write_silo_subsurf_data = switch_value;

   sprintf(key, "%s.WriteSiloPressure", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
		 switch_name, key );
   }
   public_xtra -> write_silo_press = switch_value;

   sprintf(key, "%s.WriteSiloVelocities", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
		 switch_name, key );
   }
   public_xtra -> write_silo_velocities = switch_value;

   sprintf(key, "%s.WriteSiloSaturation", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
		 switch_name, key);
   }
   public_xtra -> write_silo_satur = switch_value;

   sprintf(key, "%s.WriteSiloEvapTrans", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
		 switch_name, key);
   }
   public_xtra -> write_silo_evaptrans = switch_value;

   sprintf(key, "%s.WriteSiloEvapTransSum", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
		 switch_name, key);
   }
   public_xtra -> write_silo_evaptrans_sum = switch_value;

   sprintf(key, "%s.WriteSiloOverlandSum", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
		 switch_name, key);
   }
   public_xtra -> write_silo_overland_sum = switch_value;

   sprintf(key, "%s.WriteSiloConcentration", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
		 switch_name, key );
   }
   public_xtra -> write_silo_concen = switch_value;

   sprintf(key, "%s.WriteSiloMask", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
		 switch_name, key);
   }
   public_xtra -> write_silo_mask = switch_value;


   sprintf(key, "%s.WriteSiloSlopes", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
		 switch_name, key );
   }
   public_xtra -> write_silo_slopes = switch_value;

   sprintf(key, "%s.WriteSiloMannings", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
		 switch_name, key );
   }
   public_xtra -> write_silo_mannings = switch_value;

   sprintf(key, "%s.WriteSiloSpecificStorage", name);
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   if(switch_value < 0)
   {
      InputError("Error: invalid value <%s> for key <%s>\n",
		 switch_name, key );
   }
	public_xtra -> write_silo_specific_storage = switch_value;

   /* Initialize silo if necessary */
   if( public_xtra -> write_silo_subsurf_data || 
       public_xtra -> write_silo_press  ||
       public_xtra -> write_silo_velocities ||
       public_xtra -> write_silo_satur ||
       public_xtra -> write_silo_concen ||
       public_xtra -> write_silo_specific_storage ||
       public_xtra -> write_silo_slopes ||
       public_xtra -> write_silo_evaptrans ||
       public_xtra -> write_silo_evaptrans_sum ||
       public_xtra -> write_silo_mannings  ||
       public_xtra -> write_silo_CLM
     ) {

	   WriteSiloInit(GlobalsOutFileName);
   }

   NA_FreeNameArray(switch_na);
   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;

}

/*--------------------------------------------------------------------------
 * SolverRichardsFreePublicXtra
 *--------------------------------------------------------------------------*/

void   SolverRichardsFreePublicXtra()
{
   PFModule      *this_module = ThisPFModule;
   PublicXtra    *public_xtra = (PublicXtra *)PFModulePublicXtra(this_module);

   if ( public_xtra )
   {
      FreeProblem(public_xtra -> problem, RichardsSolve);

      PFModuleFreeModule(public_xtra -> set_problem_data);
      PFModuleFreeModule(public_xtra -> advect_concen);
      if (public_xtra -> phase_velocity_face)
	 PFModuleFreeModule(public_xtra -> phase_velocity_face); 
      PFModuleFreeModule(public_xtra -> permeability_face);
      PFModuleFreeModule(public_xtra -> nonlin_solver);
      tfree( public_xtra );
   }
}

/*--------------------------------------------------------------------------
 * SolverRichardsSizeOfTempData
 *--------------------------------------------------------------------------*/

int  SolverRichardsSizeOfTempData()
{
   /* SGS temp data */

   return 0;
}

/*--------------------------------------------------------------------------
 * SolverRichards
 *--------------------------------------------------------------------------*/
void      SolverRichards() {

   PFModule      *this_module      = ThisPFModule;
   PublicXtra    *public_xtra      = (PublicXtra *)PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra    = (InstanceXtra *)PFModuleInstanceXtra(this_module);

   Problem      *problem           = (public_xtra -> problem);
   
   double        start_time          = ProblemStartTime(problem);
   double        stop_time           = ProblemStopTime(problem);

   Grid         *grid                = (instance_xtra -> grid);

   Vector       *pressure_out;
   Vector       *porosity_out;
   Vector       *saturation_out;

   /* 
    * sk: Vector that contains the sink terms from the land surface model 
    */ 
   Vector       *evap_trans;
   
   SetupRichards(this_module);

   /*sk Initialize LSM terms*/
   evap_trans = NewVector( grid, 1, 1 );
   InitVectorAll(evap_trans, 0.0);
   AdvanceRichards(this_module, 
		   start_time, 
		   stop_time, 
		   NULL,
		   evap_trans,
		   &pressure_out, 
                   &porosity_out,
                   &saturation_out);

   /*
     Record amount of memory in use.
   */
   recordMemoryInfo();
   
   TeardownRichards(this_module);

   FreeVector(evap_trans );
}

 /* 
 * Getter/Setter methods
 */

ProblemData *GetProblemDataRichards(PFModule *this_module) {
   InstanceXtra  *instance_xtra    = (InstanceXtra *)PFModuleInstanceXtra(this_module);
   return (instance_xtra -> problem_data);
}

Problem *GetProblemRichards(PFModule *this_module) {
   PublicXtra    *public_xtra      = (PublicXtra *)PFModulePublicXtra(this_module);
   return (public_xtra -> problem);
}

PFModule *GetICPhasePressureRichards(PFModule *this_module) {
   InstanceXtra  *instance_xtra    = (InstanceXtra *)PFModuleInstanceXtra(this_module);
   return (instance_xtra -> ic_phase_pressure);
}
