/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/*****************************************************************************
 *
 * Top level 
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include "parflow.h"
#include "kinsol_dependences.h"


#define WATER 0
#define ROCK  1

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
  int                max_iterations;
  double             drop_tol;              /* drop tolerance */
  int                print_subsurf_data;    /* print permeability/porosity? */
  int                print_press;           /* print pressures? */
  int                print_velocities;      /* print velocities? */
  int                print_satur;           /* print saturations? */
  int                print_concen;          /* print concentrations? */
  int                print_wells;           /* print well data? */

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
  PFModule          *ic_phase_temperature;
  PFModule          *ic_phase_concen;
  PFModule          *problem_saturation;
  PFModule          *phase_density;
  PFModule          *phase_heat_capacity;
  PFModule          *phase_viscosity;
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

  Vector            *ctemp;

  /*****************************************************************************
   * Local variables that need to be kept around 
   *****************************************************************************/
  Vector       *pressure;
  Vector       *temperature;
  Vector       *saturation;
  Vector       *density;
  Vector       *heat_capacity_water;
  Vector       *heat_capacity_rock;
  Vector       *old_density;
  Vector       *viscosity;
  Vector       *old_viscosity;
  Vector       *old_saturation;
  Vector       *old_pressure;
  Vector       *old_temperature;
  Vector       *mask;

  /* sk: Multispecies N_Vector that is passed to Kinsol; the number of species
     is currently fixed at 2: 1-> pressure, 2-> temperature. */
  N_Vector     multispecies;
  N_VectorContent_Parflow content;


  /* 
   * sk: Vector that contains the outflow at the boundary 
   */
  Vector       *ovrl_bc_flx, *clm_energy_source, *forc_t;

  Vector       *x_velocity, *y_velocity, *z_velocity;

   double       *time_log;
   double       *dt_log;
   double        *outflow_log;
   int          *seq_log;
   int          *dumped_log;
   char         *recomp_log; 
   char         *dt_info_log;

   int file_number;
   int number_logged;

   int iteration_number;

} InstanceXtra; 


void SetupRichards(PFModule *this_module) {
  PublicXtra    *public_xtra      = PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra    = PFModuleInstanceXtra(this_module);

  Problem      *problem             = (public_xtra -> problem);
  ProblemData  *problem_data        = (instance_xtra -> problem_data);


  double        start_time          = ProblemStartTime(problem);

  /*
   * Modules used in setup
   */
  PFModule     *set_problem_data        = (instance_xtra -> set_problem_data);
  PFModule     *ic_phase_temperature    = (instance_xtra -> ic_phase_temperature);
  PFModule     *phase_density           = (instance_xtra -> phase_density);
  PFModule     *ic_phase_pressure       = (instance_xtra -> ic_phase_pressure);
  PFModule     *phase_heat_capacity     = (instance_xtra -> phase_heat_capacity);

  CommHandle   *handle;

  Grid         *grid                = (instance_xtra -> grid);
  Grid         *grid2d              = (instance_xtra -> grid2d);
  Grid         *x_grid              = (instance_xtra -> x_grid);
  Grid         *y_grid              = (instance_xtra -> y_grid);
  Grid         *z_grid              = (instance_xtra -> z_grid);

  double        dtmp, err_norm;

  double        gravity = ProblemGravity(problem);

  int           any_file_dumped;

  char          file_prefix[64];
  char          file_postfix[64];

  sprintf(file_prefix, GlobalsOutFileName);

  IfLogging(1)
    {
       int max_iterations           = (public_xtra -> max_iterations);
      instance_xtra -> seq_log      = talloc(int,    max_iterations + 1);
      instance_xtra -> time_log     = talloc(double, max_iterations + 1);
      instance_xtra -> dt_log       = talloc(double, max_iterations + 1);
      instance_xtra -> dt_info_log  = talloc(char,   max_iterations + 1);
      instance_xtra -> dumped_log   = talloc(int,    max_iterations + 1);
      instance_xtra -> recomp_log   = talloc(char,   max_iterations + 1);
      instance_xtra -> outflow_log  = talloc(double, max_iterations + 1);
      instance_xtra -> number_logged = 0;
    }

  /* do turning bands (and other stuff maybe) */
  PFModuleInvoke(void, set_problem_data, (problem_data));

  if ( public_xtra -> print_subsurf_data )
  {
     sprintf(file_postfix, "perm_x");
     WritePFBinary(file_prefix, file_postfix, 
		   ProblemDataPermeabilityX(problem_data));
     
     sprintf(file_postfix, "perm_y");
     WritePFBinary(file_prefix, file_postfix, 
		   ProblemDataPermeabilityY(problem_data));
     
     sprintf(file_postfix, "perm_z");
     WritePFBinary(file_prefix, file_postfix, 
		   ProblemDataPermeabilityZ(problem_data));
     
     sprintf(file_postfix, "porosity");
     WritePFBinary(file_prefix, file_postfix, 
		   ProblemDataPorosity(problem_data));
  }
  
  if(!amps_Rank(amps_CommWorld))
  {
     PrintWellData(ProblemDataWellData(problem_data), 
		   (WELLDATA_PRINTPHYSICAL | WELLDATA_PRINTVALUES));
  }

  int           start_count         = ProblemStartCount(problem);

  /*
   * Only allocate space if needed. 
   */
  if (start_count >= 0)
  {
     /*-------------------------------------------------------------------
      * Allocate and set up initial values
      *-------------------------------------------------------------------*/

     instance_xtra -> pressure = NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> pressure, 0.0);

     instance_xtra -> temperature = NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> temperature, 0.0);

     instance_xtra -> saturation = NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> saturation, 0.0);

     instance_xtra -> density = NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> density, 0.0);

     instance_xtra -> heat_capacity_water= NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> heat_capacity_water, 0.0);

     instance_xtra -> heat_capacity_rock= NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> heat_capacity_rock, 1.932e+6);

     instance_xtra -> viscosity= NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> viscosity, 0.0);

     instance_xtra -> old_pressure = NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> old_pressure, 0.0);

     instance_xtra -> old_temperature= NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> old_temperature, 0.0);

     instance_xtra -> old_saturation = NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> old_saturation, 0.0);

     instance_xtra -> old_density = NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> old_density, 0.0);

     instance_xtra -> old_viscosity = NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> old_viscosity, 0.0);

     instance_xtra -> clm_energy_source = NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> clm_energy_source, 0.0);

     instance_xtra -> forc_t = NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> forc_t, 0.0);

     /*sk Initialize Overland flow boundary fluxes*/
     instance_xtra -> ovrl_bc_flx = NewVector( grid2d, 1, 1 );
     InitVectorAll(instance_xtra -> ovrl_bc_flx, 0.0);

     /*sk Initialize LSM mask */
     instance_xtra -> mask = NewVector( grid, 1, 1 );
     InitVectorAll(instance_xtra -> mask, 0.0);

     /* Velocity vectors for heat convection */ 
     instance_xtra -> x_velocity = NewVector( x_grid, 1, 1 );
     InitVectorAll(instance_xtra -> x_velocity, 0.0);
     instance_xtra -> y_velocity = NewVector( y_grid, 1, 1 );
     InitVectorAll(instance_xtra -> y_velocity, 0.0);
     instance_xtra -> z_velocity = NewVector( z_grid, 1, 1 );
     InitVectorAll(instance_xtra -> z_velocity, 0.0);

     /* Set initial temperature and pass around ghost data to start */
     PFModuleInvoke(void, ic_phase_temperature, 
		    (instance_xtra -> temperature, problem_data, problem)); 

     handle = InitVectorUpdate(instance_xtra -> temperature, VectorUpdateAll);
     FinalizeVectorUpdate(handle);

     /* Set initial densities and pass around ghost data to start */
     PFModuleInvoke(void, phase_density, 
		    (WATER, instance_xtra -> pressure, instance_xtra -> temperature, 
		     instance_xtra -> density, &dtmp, &dtmp, CALCFCN));

     handle = InitVectorUpdate(instance_xtra -> density, VectorUpdateAll);
     FinalizeVectorUpdate(handle);

     /* Set initial pressures and pass around ghost data to start */
     PFModuleInvoke(void, ic_phase_pressure, 
		    (instance_xtra -> pressure, instance_xtra -> temperature, 
		     instance_xtra -> mask, problem_data, problem));
	 
     handle = InitVectorUpdate(instance_xtra -> pressure, VectorUpdateAll);
     FinalizeVectorUpdate(handle);

     /* Set initial heat capacities and pass around ghost data to start */
     PFModuleInvoke(void, phase_heat_capacity, 
		    (WATER, instance_xtra -> heat_capacity_water, problem_data));

     handle = InitVectorUpdate(instance_xtra -> heat_capacity_water, VectorUpdateAll);
     FinalizeVectorUpdate(handle);

     /* Set initial viscosities and pass around ghost data to start */
     PFModuleInvoke(void, instance_xtra -> phase_viscosity, 
		    (WATER, instance_xtra -> pressure, instance_xtra -> temperature, 
		     instance_xtra -> viscosity, CALCFCN));

     handle = InitVectorUpdate(instance_xtra -> viscosity, VectorUpdateAll);
     FinalizeVectorUpdate(handle);

     /* Set initial saturations */
     PFModuleInvoke(void, instance_xtra -> problem_saturation, 
		    (instance_xtra -> saturation, instance_xtra -> pressure, 
		     instance_xtra -> density, gravity, problem_data, 
		     CALCFCN));

     handle = InitVectorUpdate(instance_xtra -> pressure, VectorUpdateAll);
     FinalizeVectorUpdate(handle);


     /*****************************************************************/
     /*          Print out any of the requested initial data          */
     /*****************************************************************/
     any_file_dumped = 0;

     /*-------------------------------------------------------------------
      * Print out the initial well data?
      *-------------------------------------------------------------------*/

     if ( public_xtra -> print_wells )
     {
        WriteWells(file_prefix,
		   problem,
		   ProblemDataWellData(problem_data),
		   start_time, 
		   WELLDATA_WRITEHEADER);
     }

     /*-----------------------------------------------------------------
      * Print out the initial pressures?
      *-----------------------------------------------------------------*/

     if ( public_xtra -> print_press )
     {
        sprintf(file_postfix, "press.%05d", start_count );
	WritePFBinary(file_prefix, file_postfix, instance_xtra -> pressure);
	any_file_dumped = 1;
     }

     /*-----------------------------------------------------------------
      * Print out the initial temperatures?
      *-----------------------------------------------------------------*/
 
     if ( public_xtra -> print_press )
     {
        sprintf(file_postfix, "temp.%05d", start_count );
        WritePFBinary(file_prefix, file_postfix, instance_xtra -> temperature);
        any_file_dumped = 1;
     }

     /*-----------------------------------------------------------------
      * Print out the initial saturations?
      *-----------------------------------------------------------------*/

     if ( public_xtra -> print_satur )
     {
        sprintf(file_postfix, "satur.%05d", start_count );
	WritePFBinary(file_prefix, file_postfix, instance_xtra -> saturation );
	any_file_dumped = 1;
     }

     /*-----------------------------------------------------------------
      * Print out mask?
      *-----------------------------------------------------------------*/
     /* SGS this flag is wrong? */
     if ( public_xtra -> print_satur )
     {
        sprintf(file_postfix, "mask.%05d", start_count );
        WritePFBinary(file_prefix, file_postfix, instance_xtra -> mask );
        any_file_dumped = 1;
     }

     /*-----------------------------------------------------------------
      * Log this step
      *-----------------------------------------------------------------*/

     IfLogging(1)
     {
	double        outflow = 0.0; //sk Outflow due to overland flow

        instance_xtra -> seq_log[instance_xtra -> number_logged]       = start_count;
	instance_xtra -> time_log[instance_xtra -> number_logged]      = start_time;
	instance_xtra -> dt_log[instance_xtra -> number_logged]        = 0.0;
	instance_xtra -> dt_info_log[instance_xtra -> number_logged]   = 'i';
	instance_xtra -> outflow_log[instance_xtra -> number_logged]   = outflow;
	if ( any_file_dumped )
	  instance_xtra -> dumped_log[instance_xtra -> number_logged] = start_count;
	else
	  instance_xtra -> dumped_log[instance_xtra -> number_logged] = -1;
	instance_xtra -> recomp_log[instance_xtra -> number_logged]   = 'n';
	instance_xtra -> number_logged++;
     }

     instance_xtra -> file_number = start_count;
     if (any_file_dumped) instance_xtra -> file_number++;
  }

  instance_xtra -> iteration_number = start_count;
}

void TeardownRichards(PFModule *this_module) {
  PublicXtra    *public_xtra      = PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra    = PFModuleInstanceXtra(this_module);

  ProblemData  *problem_data        = (instance_xtra -> problem_data);

   if(!amps_Rank(amps_CommWorld))
   {
      PrintWellData(ProblemDataWellData(problem_data), (WELLDATA_PRINTSTATS));
   }

   FreeVector( instance_xtra -> saturation );
   FreeVector( instance_xtra -> density );
   FreeVector( instance_xtra -> heat_capacity_water );
   FreeVector( instance_xtra -> heat_capacity_rock );
   FreeVector( instance_xtra -> viscosity );
   FreeVector( instance_xtra -> old_saturation );
   FreeVector( instance_xtra -> old_pressure );
   FreeVector( instance_xtra -> old_temperature);
   FreeVector( instance_xtra -> old_viscosity );
   FreeVector( instance_xtra -> old_density );
   FreeVector( instance_xtra -> pressure );
   FreeVector( instance_xtra -> temperature );
   FreeVector( instance_xtra -> clm_energy_source );
   FreeVector( instance_xtra -> forc_t );
   FreeVector( instance_xtra -> ovrl_bc_flx );
   FreeVector( instance_xtra -> mask );
   FreeVector( instance_xtra -> x_velocity );
   FreeVector( instance_xtra -> y_velocity );
   FreeVector( instance_xtra -> z_velocity );
   
   /*-----------------------------------------------------------------------
    * Print log
    *-----------------------------------------------------------------------*/

   IfLogging(1)
   {
      FILE*  log_file;
      int        k;

      log_file = OpenLogFile("SolverRichards");

      if ( instance_xtra -> number_logged > 0 )
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
	 fprintf(log_file, " %d\n", instance_xtra -> number_logged); 
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

   }

   tfree(instance_xtra -> seq_log);
   tfree(instance_xtra -> time_log);
   tfree(instance_xtra -> dt_log);
   tfree(instance_xtra -> dt_info_log);
   tfree(instance_xtra -> dumped_log);
   tfree(instance_xtra -> recomp_log);
   tfree(instance_xtra -> outflow_log);
}

// SGS start, stop and dt seem a bit redundant.
void AdvanceRichards(PFModule *this_module, 
		     double start_time,      /* Starting time */
		     double stop_time,       /* Stopping time */
		     double dt,              /* Suggested dt, may be overridden */
		     int compute_time_step,  /* Flag if true (!= 0)
						then compute timestep
						else use provided
						dt */
		     Vector *evap_trans,     /* Flux from land surface model */
		     
		                             /* Output */
		     Vector **pressure_out,  
		     Vector **porosity_out,
		     Vector **saturation_out
   ) 
{
  PublicXtra    *public_xtra      = PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra    = PFModuleInstanceXtra(this_module);

//  sk: For the couple with CLM 
  int p = GetInt("Process.Topology.P");
  int q = GetInt("Process.Topology.Q");
  int r = GetInt("Process.Topology.R");

  Problem      *problem           = (public_xtra -> problem);

  int           max_iterations      = (public_xtra -> max_iterations);
  int           print_press         = (public_xtra -> print_press);
  int           print_velocities    = (public_xtra -> print_velocities);
  int           print_satur         = (public_xtra -> print_satur);
  int           print_wells         = (public_xtra -> print_wells);

  PFModule     *problem_saturation      = (instance_xtra -> problem_saturation);
  PFModule     *phase_density           = (instance_xtra -> phase_density);
  PFModule     *phase_viscosity         = (instance_xtra -> phase_viscosity);
  PFModule     *select_time_step        = (instance_xtra -> select_time_step);
  PFModule     *l2_error_norm           = (instance_xtra -> l2_error_norm);
  PFModule     *nonlin_solver           = (instance_xtra -> nonlin_solver);

  ProblemData  *problem_data        = (instance_xtra -> problem_data);

  double        gravity             = ProblemGravity(problem);

  Grid         *grid                = (instance_xtra -> grid);
  Grid         *grid2d              = (instance_xtra -> grid2d);
  Grid         *x_grid              = (instance_xtra -> x_grid);
  Grid         *y_grid              = (instance_xtra -> y_grid);
  Grid         *z_grid              = (instance_xtra -> z_grid);

  double        dump_interval       = ProblemDumpInterval(problem);

  Subgrid      *subgrid;
  Subvector    *p_sub, *s_sub, *t_sub, *et_sub, *es_sub, *ft_sub, *m_sub, *po_sub; 
  double       *pp, *sp, *tp, *et, *es, *ft, *ms, *po_dat;
  int          is,nx,ny,nz,nx_f,ny_f,nz_f,nch,ip,ix,iy,iz; 
  double       dx,dy,dz;
  int          rank;

  int           dump_index;
  int           any_file_dumped;
  int           dump_files;
  int           retval;
  int           converged, take_more_time_steps;
  int           conv_failures;
  int           max_failures = 60;

  double        t;
  double        print_dt;
  double        dtmp, err_norm;


  double        outflow = 0.0; //sk Outflow due to overland flow

  CommHandle   *handle;

  char          dt_info;
  char          file_prefix[64], file_postfix[64];

  sprintf(file_prefix, GlobalsOutFileName);

  Vector *porosity  = ProblemDataPorosity(problem_data);

// SGS Reworking here
  t = start_time;

  /* Check to see if pressure solves are requested */
  /* start_count < 0 implies that subsurface data ONLY is requested */
  /*    Thus, we do not want to allocate memory or initialize storage for */
  /*    other variables.  */
  if ( instance_xtra -> iteration_number < 0 )
  {
     take_more_time_steps = 0;
  }
  else  
  {
     take_more_time_steps = 1;
  }

  dump_index = 1;

  if ( ( (t >= stop_time) || (instance_xtra -> iteration_number > max_iterations) ) 
       && ( take_more_time_steps == 1) )
  {
     take_more_time_steps = 0;

     print_press           = 0;
     print_satur           = 0;
     print_wells           = 0;
  }
         
  /***********************************************************************/
  /*                                                                     */
  /*                Begin the main computational section                 */
  /*                                                                     */
  /***********************************************************************/
  
  /***************************************************************************
   sk: Here, I make the multispecies N_Vector from pressure and temperature 
   ***************************************************************************/
  instance_xtra -> multispecies = N_VMake_Parflow(instance_xtra -> pressure, instance_xtra -> temperature);
  instance_xtra -> content = NV_CONTENT_PF(instance_xtra -> multispecies);

  rank = amps_Rank(amps_CommWorld);
  dump_files = 1;

  do  /* while take_more_time_steps */
  {

     converged = 1;
     conv_failures = 0;

     do  /* while not converged */
     {
        /*******************************************************************/
        /*                  Compute time step                              */
        /*******************************************************************/
	  
	if (converged)
	{
	   if(compute_time_step) {
	      PFModuleInvoke(void, select_time_step, (&dt, &dt_info, t, problem,
                                                   problem_data) );
 
	      PFVCopy(instance_xtra -> density,    instance_xtra -> old_density);
	      PFVCopy(instance_xtra -> viscosity,  instance_xtra -> old_viscosity);
	      PFVCopy(instance_xtra -> saturation, instance_xtra -> old_saturation);
	      PFVCopy(instance_xtra -> content->specie[0], instance_xtra -> old_pressure);
	      PFVCopy(instance_xtra -> content->specie[1], instance_xtra -> old_temperature);
	      
	      /* sk: call to the land surface model/subroutine*/
	      if (dump_files==1) 
	      {
		 ForSubgridI(is, GridSubgrids(grid))
		 {
		    subgrid = GridSubgrid(grid, is);
		    p_sub = VectorSubvector(instance_xtra -> pressure, is);
		    s_sub  = VectorSubvector(instance_xtra -> saturation, is);
		    t_sub  = VectorSubvector(instance_xtra -> temperature, is);
		    et_sub = VectorSubvector(evap_trans, is);
		    es_sub = VectorSubvector(instance_xtra -> clm_energy_source, is);
		    ft_sub = VectorSubvector(instance_xtra -> forc_t, is);
		    m_sub = VectorSubvector(instance_xtra -> mask, is);
		    po_sub = VectorSubvector(porosity, is);
		    
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
		    
		    sp  = SubvectorData(s_sub);
		    pp = SubvectorData(p_sub);
		    tp = SubvectorData(t_sub);
		    et = SubvectorData(et_sub);
		    es = SubvectorData(es_sub);
		    ft = SubvectorData(ft_sub);
		    ms = SubvectorData(m_sub);
		    po_dat = SubvectorData(po_sub);
		    
		    ip = SubvectorEltIndex(p_sub, ix, iy, iz);
#ifdef HAVE_CLM
		    // printf("Before %d %d %d \n",nx, ny,nz);
		    CALL_CLM_LSM(pp,sp,tp,et,es,ft,ms,po_dat,dt,t,dx,dy,dz,ix,iy,nx,ny,nz,nx_f,ny_f,nz_f,ip,p,q,r,rank);
		    // printf("After\n");
#endif
		 }
		 handle = InitVectorUpdate(evap_trans, VectorUpdateAll);
		 FinalizeVectorUpdate(handle);
		 
		 handle = InitVectorUpdate(instance_xtra -> clm_energy_source, VectorUpdateAll);
		 FinalizeVectorUpdate(handle);
		 
		 handle = InitVectorUpdate(instance_xtra -> forc_t, VectorUpdateAll);
		 FinalizeVectorUpdate(handle);
	      }
	   } else  // Do not compute timestep
	   {
	      // Simply use DT provided; don't use select_time_step module.
	      // Note DT will still be reduced if solution does not converge.
	   }
	}
	else  /* Not converged, so decrease time step */
	{
	   t = t - dt;
	   dt = 0.5 * dt;
	   PFVCopy(instance_xtra -> old_density,    instance_xtra -> density);
	   PFVCopy(instance_xtra -> old_viscosity,  instance_xtra -> viscosity);
	   PFVCopy(instance_xtra -> old_saturation, instance_xtra -> saturation);
	   PFVCopy(instance_xtra -> old_pressure,   instance_xtra -> content->specie[0]);
	   PFVCopy(instance_xtra -> old_temperature,instance_xtra -> content->specie[1]);
	   //if(!amps_Rank(amps_CommWorld)) printf("Decreasing step size for step taken at time %12.4e.\n",t);
	}
 
         

	      //printf("Current time step size  %12.4e.\n",dt); 
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
	  
	if ( print_press || print_velocities || print_satur || print_wells )
	{
           dump_files = 0;

	   if ( dump_interval > 0 )
	   {
	      print_dt = start_time + (double)dump_index*dump_interval - t;

	      if ( dt >= print_dt )
	      {
	         dt = print_dt;
		 dt_info = 'p';
		 dump_files = 1;
		 dump_index++;
	      }
	   }
	   else
	   {
	      if ( (instance_xtra -> iteration_number % (-(int)dump_interval)) == 0 )
	      {
	         dump_files = 1;
	      }
	   }
	}


        /*--------------------------------------------------------------
         * If this is the last iteration, set appropriate variables. 
         *--------------------------------------------------------------*/
         
        printf("SolverRichard::Advance :  stop time %40.20f \n", stop_time);
        if ( (t + dt) > stop_time )
        {   
           dt = stop_time - t;
           dt_info = 'f';
        }
         
        t += dt;
 
	/*******************************************************************/
	/*          Solve the nonlinear system for this time step          */
	/*******************************************************************/
	  
	retval = PFModuleInvoke(int, nonlin_solver, 
	                        (instance_xtra -> multispecies, 
				 instance_xtra -> density, 
				 instance_xtra -> old_density, 
				 instance_xtra -> heat_capacity_water, 
				 instance_xtra -> heat_capacity_rock, 
                                 instance_xtra -> viscosity, 
				 instance_xtra -> old_viscosity, 
				 instance_xtra -> saturation, 
				 instance_xtra -> old_saturation, 
				 t, dt, problem_data, 
				 instance_xtra -> old_pressure, 
                                 instance_xtra -> old_temperature, 
				 &outflow, 
				 evap_trans, 
				 instance_xtra -> clm_energy_source, 
				 instance_xtra -> forc_t, 
				 instance_xtra -> ovrl_bc_flx,
                                 instance_xtra -> x_velocity, 
				 instance_xtra -> y_velocity, 
				 instance_xtra -> z_velocity));

	printf("SolverRichard::Advance : Outflow , %e\n",outflow);

	if (retval != 0)
	{
	   converged = 0;
	   conv_failures++;
	}
	else 
	   converged = 1;

        if (retval != 0 && dump_files == 1)
        {
           dump_files = 0;
           dump_index--;
        }

	if (conv_failures == max_failures)
	{
	   take_more_time_steps = 0;
	   if(!amps_Rank(amps_CommWorld))
	   { 
	      printf("Time step failed for time %12.4e.\n", t);
	      printf("Shutting down.\n");
	   }
	}

     }  /* Ends do for convergence of time step loop */
     while ( (!converged) && (conv_failures < max_failures) );

     instance_xtra -> iteration_number++;

     
     /* Calculate densities, viscosities and saturations for the new pressure. */
     PFModuleInvoke(void, phase_density, 
		    (0, instance_xtra -> pressure, instance_xtra -> temperature, instance_xtra -> density, &dtmp, &dtmp, CALCFCN));
     handle = InitVectorUpdate(instance_xtra -> density, VectorUpdateAll);
     FinalizeVectorUpdate(handle);

     PFModuleInvoke(void, phase_viscosity, 
		    (0, instance_xtra -> pressure, instance_xtra -> temperature, instance_xtra -> viscosity, CALCFCN));
     handle = InitVectorUpdate(instance_xtra -> viscosity, VectorUpdateAll);
     FinalizeVectorUpdate(handle);


     PFModuleInvoke(void, problem_saturation, 
		    (instance_xtra -> saturation, instance_xtra -> pressure, instance_xtra -> density, gravity, problem_data,
		     CALCFCN));

     any_file_dumped = 0;

     /***************************************************************/
     /*                 Print the pressure and saturation           */
     /***************************************************************/

     /* Dump the pressure values at this time-step */
     if ( ( print_press ) && ( dump_files ) )
     {
        sprintf(file_postfix, "press.%05d", instance_xtra -> file_number);
	WritePFBinary(file_prefix, file_postfix, instance_xtra -> pressure);

        sprintf(file_postfix, "temp.%05d", instance_xtra -> file_number);
	WritePFBinary(file_prefix, file_postfix, instance_xtra -> temperature);

	any_file_dumped = 1;
     }

     if ( ( print_satur ) && ( dump_files ) )
     {
	sprintf(file_postfix, "satur.%05d", instance_xtra -> file_number );
	WritePFBinary(file_prefix, file_postfix, instance_xtra -> saturation );

     /*sk Print the sink terms from the land surface model*/
        sprintf(file_postfix, "et.%05d", instance_xtra -> file_number );
        WritePFBinary(file_prefix, file_postfix, evap_trans);

     /*sk Print the sink terms from the land surface model*/
        sprintf(file_postfix, "obf.%05d", instance_xtra -> file_number );
        WritePFBinary(file_prefix, file_postfix, instance_xtra -> ovrl_bc_flx);

	any_file_dumped = 1;
     }

     /***************************************************************/
     /*             Compute the l2 error                            */
     /***************************************************************/

     PFModuleInvoke(void, l2_error_norm,
		    (t, instance_xtra -> pressure, problem_data, &err_norm));
     if( (!amps_Rank(amps_CommWorld)) && (err_norm >= 0.0) )
     {
        printf("l2-error in pressure: %20.8e\n", err_norm);
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

     if ( any_file_dumped ) instance_xtra -> file_number++;

     if (take_more_time_steps == 1)
        take_more_time_steps = (    (instance_xtra -> iteration_number < max_iterations) 
				 && (t < stop_time) );

  }   /* ends do for time loop */
  while( take_more_time_steps );

  *pressure_out = instance_xtra -> pressure;
  *porosity_out = ProblemDataPorosity(problem_data);
  *saturation_out = instance_xtra -> saturation;


  N_VDestroy_Parflow(instance_xtra -> multispecies);
}

/*--------------------------------------------------------------------------
 * SolverRichardsInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule *SolverRichardsInitInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra;

   Problem      *problem = (public_xtra -> problem);

   Grid         *grid;
   Grid         *grid2d;
   Grid         *x_grid;
   Grid         *y_grid;
   Grid         *z_grid;

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
      instance_xtra = PFModuleInstanceXtra(this_module);

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

   (instance_xtra -> grid) = grid;
   (instance_xtra -> grid2d) = grid2d;
   (instance_xtra -> x_grid) = x_grid;
   (instance_xtra -> y_grid) = y_grid;
   (instance_xtra -> z_grid) = z_grid;

   /*-------------------------------------------------------------------
    * Create problem_data
    *-------------------------------------------------------------------*/

   (instance_xtra -> problem_data) = NewProblemData(grid,grid2d);
   
   /*-------------------------------------------------------------------
    * Set up temporary vectors
    *-------------------------------------------------------------------*/

   (instance_xtra -> ctemp)           = NewTempVector(grid, 1, 3);

   /*-------------------------------------------------------------------
    * Initialize module instances
    *-------------------------------------------------------------------*/

   if ( PFModuleInstanceXtra(this_module) == NULL )
   {
      (instance_xtra -> phase_velocity_face) =
	/*	 PFModuleNewInstance((public_xtra -> phase_velocity_face),
                 (problem, grid, x_grid, y_grid, z_grid, NULL));
		 */
	NULL;  /* Need to change for rel. perm. and not mobility */
      (instance_xtra -> advect_concen) =
	 PFModuleNewInstance((public_xtra -> advect_concen),
                 (problem, grid, NULL));
      (instance_xtra -> set_problem_data) =
	 PFModuleNewInstance((public_xtra -> set_problem_data),
			     (problem, grid, grid2d, NULL));

      (instance_xtra -> retardation) =
	 PFModuleNewInstance(ProblemRetardation(problem), (NULL));
      (instance_xtra -> phase_rel_perm) =
	 PFModuleNewInstance(ProblemPhaseRelPerm(problem), (grid, NULL));
      (instance_xtra -> ic_phase_concen) =
	 PFModuleNewInstance(ProblemICPhaseConcen(problem), ());

      (instance_xtra -> permeability_face) =
	PFModuleNewInstance((public_xtra -> permeability_face),
			    (z_grid));
	
      (instance_xtra -> ic_phase_pressure) =
	PFModuleNewInstance(ProblemICPhasePressure(problem),(problem, grid, NULL));
      (instance_xtra -> ic_phase_temperature) =
	PFModuleNewInstance(ProblemICPhaseTemperature(problem),(problem, grid, NULL));
	  (instance_xtra -> problem_saturation) =
	PFModuleNewInstance(ProblemSaturation(problem), (grid, NULL));
      (instance_xtra -> phase_density) =
	PFModuleNewInstance(ProblemPhaseDensity(problem), ());
      (instance_xtra -> phase_heat_capacity) =
	PFModuleNewInstance(ProblemPhaseHeatCapacity(problem), ());
      (instance_xtra -> phase_viscosity) =
	PFModuleNewInstance(ProblemPhaseViscosity(problem), ());
      (instance_xtra -> select_time_step) =
	PFModuleNewInstance(ProblemSelectTimeStep(problem), ());
      (instance_xtra -> l2_error_norm) =
	PFModuleNewInstance(ProblemL2ErrorNorm(problem), ());
      (instance_xtra -> nonlin_solver) =
	PFModuleNewInstance(public_xtra -> nonlin_solver, 
			    (problem, grid, NULL, NULL));

   }
   else
   {
      PFModuleReNewInstance((instance_xtra -> phase_velocity_face),
                (problem, grid, x_grid, y_grid, z_grid, NULL));
      PFModuleReNewInstance((instance_xtra -> advect_concen),
                (problem, grid, NULL));
      PFModuleReNewInstance((instance_xtra -> set_problem_data),
			    (problem, grid, grid2d, NULL));

      PFModuleReNewInstance((instance_xtra -> retardation), (NULL));

      PFModuleReNewInstance((instance_xtra -> phase_rel_perm), (grid, NULL));
      PFModuleReNewInstance((instance_xtra -> ic_phase_concen), ());

      PFModuleReNewInstance((instance_xtra -> permeability_face),
			    (z_grid));

      PFModuleReNewInstance((instance_xtra -> ic_phase_pressure), 
			    (problem, grid, NULL));
      PFModuleReNewInstance((instance_xtra -> ic_phase_temperature), 
			    (problem, grid, NULL));
      PFModuleReNewInstance((instance_xtra -> problem_saturation), 
			    (grid, NULL)); 
      PFModuleReNewInstance((instance_xtra -> phase_density), ()); 
      PFModuleReNewInstance((instance_xtra -> phase_heat_capacity), ()); 
      PFModuleReNewInstance((instance_xtra -> phase_viscosity), ()); 
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
   sz += SizeOfVector(instance_xtra -> ctemp);
   concen_sz = sz;

   /* compute size for pressure initial condition */
   ic_sz = PFModuleSizeOfTempData(instance_xtra -> ic_phase_pressure);

   /* compute size for temperature initial condition */
   ic_sz = PFModuleSizeOfTempData(instance_xtra -> ic_phase_temperature);

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

   PFModuleReNewInstance((instance_xtra -> problem_saturation),
                         (NULL, temp_data));
   temp_data += PFModuleSizeOfTempData(instance_xtra->problem_saturation);

   PFModuleReNewInstance((instance_xtra -> phase_rel_perm),
                         (NULL, temp_data));
   temp_data += PFModuleSizeOfTempData(instance_xtra->phase_rel_perm);

   /* renew ic_phase_pressure module */
   PFModuleReNewInstance((instance_xtra -> ic_phase_pressure),
                         (NULL, NULL, temp_data));

   /* renew ic_phase_temperature module */
   PFModuleReNewInstance((instance_xtra -> ic_phase_temperature),
                         (NULL, NULL, temp_data));
   /* renew nonlinear solver module */
   PFModuleReNewInstance((instance_xtra -> nonlin_solver),
                         (NULL, NULL, NULL, temp_data));

   /* renew set_problem_data module */
   PFModuleReNewInstance((instance_xtra -> set_problem_data),
                         (NULL, NULL, NULL, temp_data));

   /* renew velocity computation modules that take temporary data */
   /*   PFModuleReNewInstance((instance_xtra -> phase_velocity_face),
             (NULL, NULL, NULL, NULL, NULL, temp_data)); */


   /* renew concentration advection modules that take temporary data */
   temp_data_placeholder = temp_data;
   PFModuleReNewInstance((instance_xtra -> retardation),
             (temp_data_placeholder));
   PFModuleReNewInstance((instance_xtra -> advect_concen),
             (NULL, NULL, temp_data_placeholder));

   temp_data_placeholder += max(PFModuleSizeOfTempData(
				   instance_xtra -> retardation),
                                PFModuleSizeOfTempData(
				   instance_xtra -> advect_concen));
   /* set temporary vector data used for advection */
   SetTempVectorData((instance_xtra -> ctemp), temp_data_placeholder);
   /*   temp_data += SizeOfVector(instance_xtra -> ctemp);  */

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
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

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
      PFModuleFreeInstance((instance_xtra -> ic_phase_temperature));
      PFModuleFreeInstance((instance_xtra -> problem_saturation));
      PFModuleFreeInstance((instance_xtra -> phase_viscosity));
      PFModuleFreeInstance((instance_xtra -> phase_density));
      PFModuleFreeInstance((instance_xtra -> phase_heat_capacity));
      PFModuleFreeInstance((instance_xtra -> select_time_step));
      PFModuleFreeInstance((instance_xtra -> l2_error_norm));
      PFModuleFreeInstance((instance_xtra -> nonlin_solver));

      PFModuleFreeInstance((instance_xtra -> permeability_face));

      FreeTempVector((instance_xtra -> ctemp));

      FreeProblemData((instance_xtra -> problem_data));

      FreeGrid((instance_xtra -> z_grid));
      FreeGrid((instance_xtra -> y_grid));
      FreeGrid((instance_xtra -> x_grid));
      FreeGrid((instance_xtra -> grid2d));
      FreeGrid((instance_xtra -> grid));

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
   switch_na = NA_NewNameArray("False True");

   public_xtra = ctalloc(PublicXtra, 1);
   
   (public_xtra -> permeability_face) = 
           PFModuleNewModule(PermeabilityFace, ());
   (public_xtra -> phase_velocity_face) = 
     /*     PFModuleNewModule(PhaseVelocityFace, ());
      */
     NULL; /* Need to account for rel. perm. and not mobility */

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

   sprintf(key, "%s.MaxIter", name);
   public_xtra -> max_iterations = GetIntDefault(key, 1000000);
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
   PublicXtra    *public_xtra = PFModulePublicXtra(this_module);

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
   return 0;
}

/*--------------------------------------------------------------------------
 * SolverRichards
 *--------------------------------------------------------------------------*/
void      SolverRichards() {

   PFModule      *this_module      = ThisPFModule;
   PublicXtra    *public_xtra      = PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra    = PFModuleInstanceXtra(this_module);

   Problem      *problem           = (public_xtra -> problem);
   
   int           start_count         = ProblemStartCount(problem);

   double        start_time          = ProblemStartTime(problem);
   double        stop_time           = ProblemStopTime(problem);
   double        dt                  = 0.0;

   Grid         *grid                = (instance_xtra -> grid);

   Vector       *pressure_out;
   Vector       *porosity_out;
   Vector       *saturation_out;

   // AdvanceRichards should use select_time_step module to compute dt.
   int compute_time_step = 1; 
                           
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
		   dt, 
		   compute_time_step, 
		   evap_trans,
		   &pressure_out, 
                   &porosity_out,
                   &saturation_out);
   
   TeardownRichards(this_module);

   FreeVector(evap_trans );
}


/* 
 * Getter/Setter methods
 */
ProblemData *GetProblemDataRichards(PFModule *this_module) {
  InstanceXtra  *instance_xtra    = PFModuleInstanceXtra(this_module);
  return (instance_xtra -> problem_data);
}

Problem *GetProblemRichards(PFModule *this_module) {
  PublicXtra    *public_xtra      = PFModulePublicXtra(this_module);
  return (public_xtra -> problem);
}

PFModule *GetICPhasePressureRichards(PFModule *this_module) {
  InstanceXtra  *instance_xtra    = PFModuleInstanceXtra(this_module);
  return (instance_xtra -> ic_phase_pressure);
}
