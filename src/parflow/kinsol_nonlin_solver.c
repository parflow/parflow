/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "parflow.h"
#include "kinsol_dependences.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{
   int       max_iter;
   int       krylov_dimension;
   int       max_restarts;
   int       print_flag;
   int       eta_choice;
   int       globalization;
   int       neq;
   int       time_index;

   double    residual_tol;
   double    step_tol;
   double    eta_value;
   double    eta_alpha;
   double    eta_gamma;
   double    derivative_epsilon;
   
   PFModule *precond_pressure;
   PFModule *precond_temperature;
   PFModule *press_function_eval;
   PFModule *temp_function_eval;
   PFModule *richards_jacobian_eval;
   PFModule *temperature_jacobian_eval;

//   KINSpgmruserAtimesFn   matvec;
   KINSpilsJacTimesVecFn   matvec;
//   KINSpgmrPrecondFn      pcinit;
   KINSpilsPrecSetupFn      pcinit;
//   KINSpgmrPrecondSolveFn pcsolve;
   KINSpilsPrecSolveFn pcsolve;

} PublicXtra;

typedef struct
{
   PFModule  *precond_pressure;
   PFModule  *precond_temperature;
   PFModule  *press_function_eval;
   PFModule  *temp_function_eval;
   PFModule  *richards_jacobian_eval;
   PFModule  *temperature_jacobian_eval;

   N_Vector  uscalen;
   N_Vector  fscalen;

   Matrix   *jacobian_matrix_press;
   Matrix   *jacobian_matrix_temp;

   State    *current_state;

   void     *kin_mem;
   FILE     *kinsol_file;

} InstanceXtra;


/*--------------------------------------------------------------------------
 * Auxilliary functions for interfacing with outside software
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * KINSolInitPC
 *--------------------------------------------------------------------------*/
int  KINSolInitPC(multispecies, uscale, fval, fscale, current_state, vtemp1, vtemp2)
N_Vector  multispecies;
N_Vector  uscale;
N_Vector  fval;
N_Vector  fscale;
void     *current_state;
N_Vector  vtemp1;
N_Vector  vtemp2;
{
   PFModule    *precond_pressure    = StatePrecondPressure(    ((State*)current_state) );
   PFModule    *precond_temperature = StatePrecondTemperature( ((State*)current_state) );
   ProblemData *problem_data        = StateProblemData(        ((State*)current_state) );
   Vector      *saturation          = StateSaturation(         ((State*)current_state) );
   Vector      *density             = StateDensity(            ((State*)current_state) );
   Vector      *heat_capacity_water = StateHeatCapacityWater(  ((State*)current_state) );
   Vector      *heat_capacity_rock  = StateHeatCapacityRock(   ((State*)current_state) );
   Vector      *viscosity           = StateViscosity(          ((State*)current_state) );
   Vector      *x_velocity          = StateXVelocity(          ((State*)current_state) );
   Vector      *y_velocity          = StateYVelocity(          ((State*)current_state) );
   Vector      *z_velocity          = StateZVelocity(          ((State*)current_state) );
   double       dt                  = StateDt(                 ((State*)current_state) );
   double       time                = StateTime(               ((State*)current_state) );

   Vector      *pressure;
   Vector      *temperature;

   N_VectorContent_Parflow  content;
   int          n, nspecies;
   
   content = NV_CONTENT_PF(multispecies);
   nspecies = content->num_species;

   pressure = content->specie[0];
   temperature = content->specie[1];

   for (n = 0; n < nspecies; n++)
   {
     /* The preconditioner module initialized here is the KinsolPC module
        itself */
     if (n == 0)
     {
       PFModuleReNewInstance(precond_pressure, (NULL, NULL, problem_data, NULL, 
				     pressure, temperature, saturation, density, viscosity, dt, time));
     } 
     else if (n == (nspecies-1))
     {
       PFModuleReNewInstance(precond_temperature, (NULL, NULL, problem_data, NULL, 
				     temperature, pressure, saturation, density, heat_capacity_water, heat_capacity_rock, 
                                     x_velocity, y_velocity, z_velocity, dt, time));
     }

   }
   return(0);
}


/*--------------------------------------------------------------------------
 * KINSolCallPC
 *--------------------------------------------------------------------------*/
int   KINSolCallPC(multispecies, uscale, fval, fscale, vtem, current_state, ftem)
N_Vector  multispecies;
N_Vector  uscale;
N_Vector  fval;
N_Vector  fscale;
N_Vector  vtem;
void     *current_state;
N_Vector  ftem;
{
   PFModule *precond_pressure    = StatePrecondPressure( (State*)current_state );
   PFModule *precond_temperature = StatePrecondTemperature( (State*)current_state );
   N_VectorContent_Parflow content;
   Vector   *vtem_specie;
   int i, j, nspecies;
   
   content = NV_CONTENT_PF(vtem);
   nspecies = content->num_species;

   for (i = 0; i < nspecies; i++)
   {
     vtem_specie = content->specie[i];
   
     /* The preconditioner module invoked here is the KinsolPC module
        itself */

     if (i == 0)
       PFModuleInvoke(void, precond_pressure, (vtem_specie));
     else if (i == (nspecies-1))
       PFModuleInvoke(void, precond_temperature, (vtem_specie));

    }

   return(0);
}

void PrintFinalStats(out_file)
FILE       *out_file;
{
  fprintf(out_file, "\n-------------------------------------------------- \n");
  fprintf(out_file, "                    Iteration             Total\n");
  
/*  fprintf(out_file, "Nonlin. Its.:           %5ld             %5ld\n", 
	  integer_outputs_now[NNI], integer_outputs_total[NNI]);
  fprintf(out_file, "Lin. Its.:              %5ld             %5ld\n", 
	  integer_outputs_now[SPGMR_NLI], integer_outputs_total[SPGMR_NLI]);
  fprintf(out_file, "Func. Evals.:           %5ld             %5ld\n", 
	  integer_outputs_now[NFE], integer_outputs_total[NFE]);
  fprintf(out_file, "PC Evals.:              %5ld             %5ld\n", 
	  integer_outputs_now[SPGMR_NPE], integer_outputs_total[SPGMR_NPE]);
  fprintf(out_file, "PC Solves:              %5ld             %5ld\n", 
	  integer_outputs_now[SPGMR_NPS], integer_outputs_total[SPGMR_NPS]);
  fprintf(out_file, "Lin. Conv. Fails:       %5ld             %5ld\n", 
	  integer_outputs_now[SPGMR_NCFL], integer_outputs_total[SPGMR_NCFL]);
  fprintf(out_file, "Beta Cond. Fails:       %5ld             %5ld\n", 
	  integer_outputs_now[NBCF], integer_outputs_total[NBCF]);
  fprintf(out_file, "Backtracks:             %5ld             %5ld\n", 
	  integer_outputs_now[NBKTRK], integer_outputs_total[NBKTRK]);
  fprintf(out_file,   "-------------------------------------------------- \n"); */
}

/*--------------------------------------------------------------------------
 * KinsolNonlinSolver
 *--------------------------------------------------------------------------*/

int          KinsolNonlinSolver(multispecies, density, old_density, heat_capacity_water, heat_capacity_rock, viscosity, old_viscosity, saturation, 
				old_saturation, t, dt, problem_data, old_pressure, 
                                old_temperature,outflow,evap_trans,clm_energy_source,forc_t,ovrl_bc_flx,x_velocity,y_velocity,z_velocity)
N_Vector    multispecies;
Vector      *density;
Vector      *old_density;
Vector      *heat_capacity_water;
Vector      *heat_capacity_rock;
Vector      *viscosity;
Vector      *old_viscosity;
Vector      *saturation;
Vector      *old_saturation;
double       t;
double       dt;
ProblemData *problem_data;
Vector      *old_pressure;
Vector      *old_temperature;
double       *outflow; //sk
Vector      *evap_trans;
Vector      *clm_energy_source;
Vector      *forc_t;
Vector      *ovrl_bc_flx;
Vector      *x_velocity;
Vector      *y_velocity;
Vector      *z_velocity;
{
   PFModule     *this_module      = ThisPFModule;
   PublicXtra   *public_xtra      = PFModulePublicXtra(this_module);
   InstanceXtra *instance_xtra    = PFModuleInstanceXtra(this_module);

   Matrix       *jacobian_matrix_press  = (instance_xtra -> jacobian_matrix_press);
   Matrix       *jacobian_matrix_temp   = (instance_xtra -> jacobian_matrix_temp);
   N_Vector      uscale           = (instance_xtra -> uscalen);
   N_Vector      fscale           = (instance_xtra -> fscalen);

   PFModule  *press_function_eval       = instance_xtra -> press_function_eval;
   PFModule  *temp_function_eval        = instance_xtra -> temp_function_eval;
   PFModule  *richards_jacobian_eval    = instance_xtra -> richards_jacobian_eval;
   PFModule  *temperature_jacobian_eval = instance_xtra -> temperature_jacobian_eval;
   PFModule  *precond_pressure          = instance_xtra -> precond_pressure;
   PFModule  *precond_temperature       = instance_xtra -> precond_temperature;

   State        *current_state    = (instance_xtra -> current_state);

   int           globalization    = (public_xtra -> globalization);
   int           neq              = (public_xtra -> neq);

   double        residual_tol     = (public_xtra -> residual_tol);
   double        step_tol         = (public_xtra -> step_tol);

   void         *kin_mem          = (instance_xtra -> kin_mem);
   FILE         *kinsol_file      = (instance_xtra -> kinsol_file);

   int           ret              = 0;
   
   N_VectorContent_Parflow content= NV_CONTENT_PF(multispecies);

   StateFuncPress(current_state)          = press_function_eval;
   StateFuncTemp(current_state)           = temp_function_eval;
   StateProblemData(current_state)        = problem_data;
   StateTime(current_state)               = t;
   StateDt(current_state)                 = dt;
   StateDensity(current_state)            = density;
   StateOldDensity(current_state)         = old_density;
   StateHeatCapacityWater(current_state)  = heat_capacity_water;
   StateHeatCapacityRock(current_state)   = heat_capacity_rock;
   StateViscosity(current_state)          = viscosity;
   StateOldViscosity(current_state)       = old_viscosity;
   StateOldPressure(current_state)        = old_pressure;
   StateOldTemperature(current_state)     = old_temperature;
   StateSaturation(current_state)         = saturation;
   StateOldSaturation(current_state)      = old_saturation;
   StateJacEvalPressure(current_state)    = richards_jacobian_eval;
   StateJacEvalTemperature(current_state) = temperature_jacobian_eval;
   StateJacPressure(current_state)        = jacobian_matrix_press;
   StateJacTemperature(current_state)     = jacobian_matrix_temp;
   StatePrecondPressure(current_state)    = precond_pressure;
   StatePrecondTemperature(current_state) = precond_temperature;
   StateOutflow(current_state)            = outflow; /*sk*/
   StateEvapTrans(current_state)          = evap_trans; /*sk*/
   StateClmEnergySource(current_state)    = clm_energy_source; /*sk*/
   StateForcT(current_state)              = forc_t; /*sk*/
   StateOvrlBcFlx(current_state)          = ovrl_bc_flx; /*sk*/
   StateXVelocity(current_state)          = x_velocity; /*sk*/
   StateYVelocity(current_state)          = y_velocity; /*sk*/
   StateZVelocity(current_state)          = z_velocity; /*sk*/

   if (!amps_Rank(amps_CommWorld))
      fprintf(kinsol_file,"\nKINSOL starting step for time %f\n",t);

   BeginTiming(public_xtra -> time_index);

   /* Specfies pointer to specie vector */
   content = NV_CONTENT_PF(multispecies);
   ret = KINSol( (void*)kin_mem,        /* Memory allocated above */
	         multispecies,          /* Initial guess @ this was "pressure before" */
	         globalization,         /* Globalization method */
	         uscale,                /* Scalings for the variable */
	         fscale                /* Scalings for the function */
	       );

   EndTiming(public_xtra -> time_index);


   if (!amps_Rank(amps_CommWorld))
      PrintFinalStats(kinsol_file);

   if ( ret == KIN_SUCCESS || ret == KIN_INITIAL_GUESS_OK ) 
   {
      ret = 0;
   }

   return(ret);

}

/*--------------------------------------------------------------------------
 * KinsolNonlinSolverInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *KinsolNonlinSolverInitInstanceXtra(problem, grid, problem_data, 
					      temp_data)
Problem     *problem;
Grid        *grid;
ProblemData *problem_data;
double      *temp_data;
{
   PFModule      *this_module        = ThisPFModule;
   PublicXtra    *public_xtra        = PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra;

   int           neq                 = public_xtra -> neq;
   int           max_restarts        = public_xtra -> max_restarts;
   int           krylov_dimension    = public_xtra -> krylov_dimension;
   int           max_iter            = public_xtra -> max_iter;
   int           print_flag          = public_xtra -> print_flag;
   int           eta_choice          = public_xtra -> eta_choice;

   double        eta_value           = public_xtra -> eta_value;
   double        eta_alpha           = public_xtra -> eta_alpha;
   double        eta_gamma           = public_xtra -> eta_gamma;
   double        derivative_epsilon  = public_xtra -> derivative_epsilon;
   double        residual_tol        = public_xtra -> residual_tol;
   double        step_tol            = public_xtra -> step_tol;
   

   N_Vector     templ;
   N_Vector     fscalen, uscalen;
   N_VectorContent_Parflow content;
   Vector       *template;
   Vector       *fscale;
   Vector       *uscale;
   int          flag;

   State        *current_state;

   //KINSpgmruserAtimesFn   matvec      = public_xtra -> matvec;
   KINSpilsJacTimesVecFn    matvec      = public_xtra -> matvec;
   //KINSpgmrPrecondFn      pcinit      = public_xtra -> pcinit;
   KINSpilsPrecSetupFn      pcinit      = public_xtra -> pcinit;
   //KINSpgmrPrecondSolveFn pcsolve     = public_xtra -> pcsolve;
   KINSpilsPrecSolveFn      pcsolve     = public_xtra -> pcsolve;

   void                  *kin_mem = NULL;
   FILE                  *kinsol_file;
   char                   filename[255];

   int                    i;


   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   /*-----------------------------------------------------------------------
    * Initialize module instances
    *-----------------------------------------------------------------------*/

   if ( PFModuleInstanceXtra(this_module) == NULL )
   {
      if (public_xtra -> precond_pressure != NULL)
	 instance_xtra -> precond_pressure =
	    PFModuleNewInstance(public_xtra -> precond_pressure,
				(problem, grid, problem_data, temp_data,
				 NULL, NULL, NULL, NULL, NULL));
      else
	 instance_xtra -> precond_pressure = NULL;

      if (public_xtra -> precond_temperature != NULL)
         instance_xtra -> precond_temperature =
            PFModuleNewInstance(public_xtra -> precond_temperature,
                                (problem, grid, problem_data, temp_data,
                                 NULL, NULL, NULL, NULL, NULL));
      else
         instance_xtra -> precond_temperature = NULL;

      instance_xtra -> press_function_eval = 
	 PFModuleNewInstance(public_xtra -> press_function_eval, 
			     (problem, grid, temp_data));

      instance_xtra -> temp_function_eval = 
	 PFModuleNewInstance(public_xtra -> temp_function_eval, 
			     (problem, grid, temp_data));

      if (public_xtra -> richards_jacobian_eval != NULL)
	 /* Initialize instance for nonsymmetric matrix */
	 instance_xtra -> richards_jacobian_eval = 
	    PFModuleNewInstance(public_xtra -> richards_jacobian_eval, 
				(problem, grid, temp_data, 0));
      else
	 instance_xtra -> richards_jacobian_eval = NULL;

      if (public_xtra -> temperature_jacobian_eval != NULL)
         /* Initialize instance for nonsymmetric matrix */
         instance_xtra -> temperature_jacobian_eval =
            PFModuleNewInstance(public_xtra -> temperature_jacobian_eval,
                                (problem, grid, temp_data, 0));
      else
         instance_xtra -> temperature_jacobian_eval = NULL;

   }
   else
   {
      if (instance_xtra -> precond_pressure != NULL)
	 PFModuleReNewInstance(instance_xtra -> precond_pressure,
			        (problem, grid, problem_data, temp_data,
				 NULL, NULL, NULL, NULL, NULL));

      if (instance_xtra -> precond_temperature != NULL)
         PFModuleReNewInstance(instance_xtra -> precond_temperature,
                                (problem, grid, problem_data, temp_data,
                                 NULL, NULL, NULL, NULL, NULL));

      PFModuleReNewInstance(instance_xtra -> press_function_eval, 
			    (problem, grid, temp_data));

      PFModuleReNewInstance(instance_xtra -> temp_function_eval, 
			    (problem, grid, temp_data));

      if (instance_xtra -> richards_jacobian_eval != NULL)
	 PFModuleReNewInstance(instance_xtra -> richards_jacobian_eval, 
			       (problem, grid, temp_data, 0));

      if (instance_xtra -> temperature_jacobian_eval != NULL)
	 PFModuleReNewInstance(instance_xtra -> temperature_jacobian_eval, 
			       (problem, grid, temp_data, 0));
   }

   /*-----------------------------------------------------------------------
    * Initialize KINSol input parameters and memory instance
    *-----------------------------------------------------------------------*/

   if ( PFModuleInstanceXtra(this_module) == NULL )
   {
      current_state = ctalloc( State, 1 );

      /* Initialize KINSol parameters */
      sprintf(filename, "%s.%s", GlobalsOutFileName, "kinsol.log");
      if (!amps_Rank(amps_CommWorld))
	 kinsol_file = fopen( filename, "w" );
      else
	 kinsol_file = NULL;
      instance_xtra -> kinsol_file = kinsol_file;

      /* Scaling vectors*/
      uscalen = N_VNew_Parflow(grid); 
      N_VConst_Parflow(1.0,uscalen);
      instance_xtra -> uscalen = uscalen;
      content = NV_CONTENT_PF(uscalen);

      fscalen = N_VNew_Parflow(grid); 
      N_VConst_Parflow(1.0,fscalen);
      instance_xtra -> fscalen = fscalen;

      /* Initialize KINSol memory */
      kin_mem = KINCreate();
      if (kin_mem == NULL) printf ("\n KINSOL COULD NOT CREATE MEMORY BLOCK \n");

      flag = KINSetInfoFile(kin_mem, kinsol_file);
      flag = KINSetErrFile(kin_mem, kinsol_file);
      flag = KINSetPrintLevel(kin_mem, 0);
      if (!amps_Rank(amps_CommWorld))
        flag = KINSetPrintLevel(kin_mem, print_flag);

      flag = KINSetNumMaxIters(kin_mem, max_iter);
      flag = KINSetEtaForm(kin_mem, eta_choice);
      if (eta_choice == KIN_ETACONSTANT) flag = KINSetEtaConstValue(kin_mem, eta_value);
      if (eta_choice == KIN_ETACHOICE2)  flag = KINSetEtaParams(kin_mem, eta_gamma, eta_alpha);
      
     
      flag = KINMalloc(kin_mem,KINSolFunctionEval,uscalen);
      if (flag == KIN_SUCCESS) printf("\nKINSOL ALLOCATED SUCCESSFULLY \n");
      if (flag == KIN_MEM_NULL) {
        printf("\nKINSOL COULD NOT ALLOCATE MEMORY \n");
        return;}
      if (flag == KIN_MEM_FAIL) {
        printf("\nMEMEORY ALLOCATION REQUEST FAILED\n");
        return;}
      if (flag == KIN_ILL_INPUT) {
        printf("\nINPUT ARGUMENT HAS ILLEGAL VALUE\n");
        return;}

      /* Specifies pointer to user-defined memory */
      instance_xtra -> kin_mem = kin_mem;
      instance_xtra -> current_state = current_state;

      flag = KINSetFdata(kin_mem, instance_xtra->current_state);
 
      flag = KINSetFuncNormTol(kin_mem,residual_tol);
      flag = KINSetScaledStepTol(kin_mem, step_tol);

      /*Initialize Preconditioner */
      

      /* Initialize the gmres linear solver in KINSol */
      flag = KINSpgmr(kin_mem,        /* Memory allocated above */
		krylov_dimension      /* Max. Krylov dimension */
		);
      if (flag < 0) printf("\n LINEAR SOLVER COULD NOT ALLOCATE MEMORY \n");

      /* Initialize optional arguments for KINSol */
      if (public_xtra->precond_pressure != NULL) {
        flag = KINSpilsSetPreconditioner(kin_mem, pcinit, pcsolve, current_state);
        flag = KINSpilsSetJacTimesVecFn(kin_mem, matvec, current_state);
      } 

      /* Put in conditional assignment of eta_gamma since KINSOL aliases */
      /* ETAGAMMA and ETACONST */

      /* Initialize iteration counts */

   }


   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * KinsolNonlinSolverFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  KinsolNonlinSolverFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);
   void *kin_mem;


   if (instance_xtra)
   {
      PFModuleFreeInstance((instance_xtra -> press_function_eval));
      PFModuleFreeInstance((instance_xtra -> temp_function_eval));
      if (instance_xtra -> richards_jacobian_eval != NULL)
      {
         PFModuleFreeInstance((instance_xtra -> richards_jacobian_eval));
      }

      if (instance_xtra -> precond_pressure != NULL)
         PFModuleFreeInstance((instance_xtra -> precond_pressure));

      if (instance_xtra -> precond_temperature != NULL)
         PFModuleFreeInstance((instance_xtra -> precond_temperature));

      N_VDestroy_Parflow(instance_xtra -> uscalen);
      N_VDestroy_Parflow(instance_xtra -> fscalen);

      tfree(instance_xtra -> current_state);

      kin_mem = (instance_xtra -> kin_mem);
      KINFree(&kin_mem);

      if (instance_xtra->kinsol_file) 
	 fclose((instance_xtra -> kinsol_file));

      tfree(instance_xtra);
   }
}

/*--------------------------------------------------------------------------
 * KinsolNonlinSolverNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *KinsolNonlinSolverNewPublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   char          *switch_name;
   char           key[IDB_MAX_KEY_LEN];
   int            switch_value;

   NameArray      switch_na;
   NameArray      verbosity_switch_na;
   NameArray      eta_switch_na;
   NameArray      globalization_switch_na;
   NameArray      precond_switch_na;

   public_xtra = ctalloc(PublicXtra, 1);

   sprintf(key, "Solver.Nonlinear.ResidualTol");
   (public_xtra -> residual_tol) = GetDoubleDefault(key, 1e-7);
   sprintf(key, "Solver.Nonlinear.StepTol");
   (public_xtra -> step_tol)     = GetDoubleDefault(key, 1e-7);

   sprintf(key, "Solver.Nonlinear.MaxIter");
   (public_xtra -> max_iter)     = GetIntDefault(key, 15);
   sprintf(key, "Solver.Linear.KrylovDimension");
   (public_xtra -> krylov_dimension) = GetIntDefault(key, 10);
   sprintf(key, "Solver.Linear.MaxRestarts");
   (public_xtra -> max_restarts) = GetIntDefault(key, 0);

   verbosity_switch_na = NA_NewNameArray("NoVerbosity LowVerbosity "
	                                 "NormalVerbosity HighVerbosity");
   sprintf(key, "Solver.Nonlinear.PrintFlag");
   switch_name = GetStringDefault(key, "HighVerbosity");
   (public_xtra -> print_flag) = NA_NameToIndex(verbosity_switch_na, 
                                                switch_name);
   NA_FreeNameArray(verbosity_switch_na);

   eta_switch_na = NA_NewNameArray("EtaConstant Walker1 Walker2");
   sprintf(key, "Solver.Nonlinear.EtaChoice");
   switch_name = GetStringDefault(key, "Walker2");
   switch_value = NA_NameToIndex(eta_switch_na, switch_name);
   switch (switch_value)
   {
      case 0:
      {
	 public_xtra -> eta_choice = KIN_ETACONSTANT;
	 public_xtra -> eta_value  
	                 = GetDoubleDefault("Solver.Nonlinear.EtaValue", 1e-4);
	 public_xtra -> eta_alpha = 0.0;
	 public_xtra -> eta_gamma = 0.0;
	 break;
      }
      case 1:
      {
	 public_xtra -> eta_choice = KIN_ETACHOICE1;
	 public_xtra -> eta_alpha = 0.0;
	 public_xtra -> eta_gamma = 0.0;
	 break;
      }
      case 2:
      {
	 public_xtra -> eta_choice = KIN_ETACHOICE2;
	 public_xtra -> eta_alpha 
                         = GetDoubleDefault("Solver.Nonlinear.EtaAlpha", 2.0);
	 public_xtra -> eta_gamma 
                         = GetDoubleDefault("Solver.Nonlinear.EtaGamma", 0.9);
	 public_xtra -> eta_value = 0.0;
	 break;
      }
      default:
      {
	 InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
		     key);
      }
   }
   NA_FreeNameArray(eta_switch_na);

   switch_na = NA_NewNameArray("False True");
   sprintf(key, "Solver.Nonlinear.UseJacobian");
   switch_name = GetStringDefault(key, "False");
   switch_value = NA_NameToIndex(switch_na, switch_name);
   switch (switch_value)
   {
      case 0:
      {
         (public_xtra -> matvec) = NULL;
         break;
      }
      case 1:
      {
         (public_xtra -> matvec) = KINSolMatVec;
         break;
      }
      default:
      {
	 InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
		     key);
      }
   }
   NA_FreeNameArray(switch_na);

   sprintf(key, "Solver.Nonlinear.DerivativeEpsilon");
   (public_xtra -> derivative_epsilon) = GetDoubleDefault(key, 1e-7);

   globalization_switch_na = NA_NewNameArray("InexactNewton LineSearch");
   sprintf(key, "Solver.Nonlinear.Globalization");
   switch_name = GetStringDefault(key, "LineSearch");
   switch_value = NA_NameToIndex(globalization_switch_na, switch_name);
   switch (switch_value)
   {
      case 0:
      {
	 (public_xtra -> globalization) = KIN_NONE;
	 break;
      }
      case 1:
      {
	 (public_xtra -> globalization) = KIN_LINESEARCH;
	 break;
      }
      default:
      {
         InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
		     key);
      }
   }
   NA_FreeNameArray(globalization_switch_na);
 
   precond_switch_na = NA_NewNameArray("NoPC MGSemi SMG PFMG");
   sprintf(key, "Solver.Linear.Preconditioner");
   switch_name = GetStringDefault(key, "MGSemi");
   switch_value = NA_NameToIndex(precond_switch_na, switch_name);
   if (switch_value == 0)
   {
      (public_xtra -> precond_pressure) = NULL;
      (public_xtra -> precond_temperature) = NULL;
      (public_xtra -> pcinit)  = NULL;
      (public_xtra -> pcsolve) = NULL;
   }
   else if ( switch_value > 0 )
   {
      (public_xtra -> precond_pressure) = PFModuleNewModule(KinsolPCPressure, 
						   (key, switch_name));
      (public_xtra -> precond_temperature) = PFModuleNewModule(KinsolPCTemperature, 
						   (key, switch_name));
      (public_xtra -> pcinit)  = (KINSpilsPrecSetupFn)KINSolInitPC;
      (public_xtra -> pcsolve) = (KINSpilsPrecSolveFn)KINSolCallPC;
   }
   else
   {
      InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
		 key);
   }
   NA_FreeNameArray(precond_switch_na);

   public_xtra -> press_function_eval = PFModuleNewModule(PressFunctionEval, ());
   public_xtra -> temp_function_eval = PFModuleNewModule(TempFunctionEval, ());
   public_xtra -> neq = ((public_xtra -> max_restarts)+1)
                           *(public_xtra -> krylov_dimension);

   if (public_xtra -> matvec != NULL)
   {
      public_xtra -> richards_jacobian_eval = 
                                 PFModuleNewModule(RichardsJacobianEval, ());
      public_xtra -> temperature_jacobian_eval = 
                                 PFModuleNewModule(TemperatureJacobianEval, ());
   } else {
      public_xtra -> richards_jacobian_eval = NULL;
      public_xtra -> temperature_jacobian_eval = NULL;
   }

   (public_xtra -> time_index) = RegisterTiming("KINSol");
   
   PFModulePublicXtra(this_module) = public_xtra;

   return this_module;
}

/*-------------------------------------------------------------------------
 * KinsolNonlinSolverFreePublicXtra
 *-------------------------------------------------------------------------*/

void  KinsolNonlinSolverFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);

   if ( public_xtra )
   {
      if (public_xtra -> richards_jacobian_eval != NULL)
         PFModuleFreeModule(public_xtra -> richards_jacobian_eval);

      if (public_xtra -> temperature_jacobian_eval != NULL)
         PFModuleFreeModule(public_xtra -> temperature_jacobian_eval);

      if (public_xtra -> precond_pressure != NULL)
         PFModuleFreeModule(public_xtra -> precond_pressure);

      if (public_xtra -> precond_temperature!= NULL)
         PFModuleFreeModule(public_xtra -> precond_temperature);

      PFModuleFreeModule(public_xtra -> press_function_eval);
      PFModuleFreeModule(public_xtra -> temp_function_eval);

      tfree(public_xtra);
   }
}

/*--------------------------------------------------------------------------
 * KinsolNonlinSolverSizeOfTempData
 *--------------------------------------------------------------------------*/

int  KinsolNonlinSolverSizeOfTempData()
{
   PFModule             *this_module   = ThisPFModule;
   InstanceXtra         *instance_xtra = PFModuleInstanceXtra(this_module);

   PFModule             *precond_pressure          = (instance_xtra -> precond_pressure);
   PFModule             *precond_temperature       = (instance_xtra -> precond_temperature);
   PFModule             *pressure_jacobian_eval    = (instance_xtra -> richards_jacobian_eval);
   PFModule             *temperature_jacobian_eval = (instance_xtra -> temperature_jacobian_eval);

   int sz = 0;
   
   if (pressure_jacobian_eval != NULL)
      sz += PFModuleSizeOfTempData(pressure_jacobian_eval);

   if (temperature_jacobian_eval != NULL)
      sz += PFModuleSizeOfTempData(temperature_jacobian_eval);

   if (precond_pressure != NULL)
      sz += PFModuleSizeOfTempData(precond_pressure);

   if (precond_temperature != NULL)
      sz += PFModuleSizeOfTempData(precond_temperature);

   return sz;
}
