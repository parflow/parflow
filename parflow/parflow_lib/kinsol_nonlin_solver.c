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
   
   PFModule *precond;
   PFModule *nl_function_eval;
   PFModule *richards_jacobian_eval;

   KINSpgmruserAtimesFn   matvec;
   KINSpgmrPrecondFn      pcinit;
   KINSpgmrPrecondSolveFn pcsolve;

} PublicXtra;

typedef struct
{
   PFModule  *precond;
   PFModule  *nl_function_eval;
   PFModule  *richards_jacobian_eval;

   Vector   *uscale;
   Vector   *fscale;

   Matrix   *jacobian_matrix;

   long int  integer_outputs[OPT_SIZE];

   long int  int_optional_input[OPT_SIZE];
   double    real_optional_input[OPT_SIZE];

   State    *current_state;

   KINMem    kin_mem;
   FILE     *kinsol_file;
   SysFn     feval;

} InstanceXtra;


/*--------------------------------------------------------------------------
 * Auxilliary functions for interfacing with outside software
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * KINSolInitPC
 *--------------------------------------------------------------------------*/
int  KINSolInitPC(neq, pressure, uscale, fval, fscale, vtemp1, 
		  vtemp2, nl_function, uround, nfePtr, current_state)
int       neq;
N_Vector  pressure;
N_Vector  uscale;
N_Vector  fval;
N_Vector  fscale;
N_Vector  vtemp1;
N_Vector  vtemp2;
void     *nl_function;
double    uround;
long int *nfePtr;
void     *current_state;
{
   PFModule    *precond      = StatePrecond( ((State*)current_state) );
   ProblemData *problem_data = StateProblemData( ((State*)current_state) );
   Vector      *saturation   = StateSaturation( ((State*)current_state) );
   Vector      *density      = StateDensity( ((State*)current_state) );
   double       dt           = StateDt( ((State*)current_state) );
   double       time         = StateTime( ((State*)current_state) );

   /* The preconditioner module initialized here is the KinsolPC module
      itself */

   PFModuleReNewInstance(precond, (NULL, NULL, problem_data, NULL, 
				   pressure, saturation, density, dt, time));
   return(0);
}


/*--------------------------------------------------------------------------
 * KINSolCallPC
 *--------------------------------------------------------------------------*/
int   KINSolCallPC(neq, pressure, uscale, fval, fscale, vtem, ftem, 
		   nl_function, uround, nfePtr, current_state)
int       neq;	
N_Vector  pressure;
N_Vector  uscale;
N_Vector  fval;
N_Vector  fscale;
N_Vector  vtem;
N_Vector  ftem;
void     *nl_function;
double    uround;
long int *nfePtr;
void     *current_state;
{
   PFModule *precond = StatePrecond( (State*)current_state );

   /* The preconditioner module invoked here is the KinsolPC module
      itself */

   PFModuleInvoke(void, precond, (vtem));

   return(0);
}

void PrintFinalStats(out_file, integer_outputs_now, integer_outputs_total)
FILE       *out_file;
long int   *integer_outputs_now;
long int   *integer_outputs_total;
{
  fprintf(out_file, "\n-------------------------------------------------- \n");
  fprintf(out_file, "                    Iteration             Total\n");
  fprintf(out_file, "Nonlin. Its.:           %5ld             %5ld\n", 
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
  fprintf(out_file,   "-------------------------------------------------- \n");
}

/*--------------------------------------------------------------------------
 * KinsolNonlinSolver
 *--------------------------------------------------------------------------*/

int          KinsolNonlinSolver(pressure, density, old_density, saturation, 
				old_saturation, t, dt, problem_data, old_pressure, 
                                outflow,evap_trans,ovrl_bc_flx)
Vector      *pressure;
Vector      *old_pressure;
Vector      *density;
Vector      *old_density;
Vector      *saturation;
Vector      *old_saturation;
Vector      *evap_trans;
Vector      *ovrl_bc_flx;
double       t;
double       dt;
double       *outflow; //sk
ProblemData *problem_data;
{
   PFModule     *this_module      = ThisPFModule;
   PublicXtra   *public_xtra      = PFModulePublicXtra(this_module);
   InstanceXtra *instance_xtra    = PFModuleInstanceXtra(this_module);

   Matrix       *jacobian_matrix  = (instance_xtra -> jacobian_matrix);
   Vector       *uscale           = (instance_xtra -> uscale);
   Vector       *fscale           = (instance_xtra -> fscale);

   PFModule  *nl_function_eval       = instance_xtra -> nl_function_eval;
   PFModule  *richards_jacobian_eval = instance_xtra -> richards_jacobian_eval;
   PFModule  *precond                = instance_xtra -> precond;

   State        *current_state    = (instance_xtra -> current_state);

   int           globalization    = (public_xtra -> globalization);
   int           neq              = (public_xtra -> neq);

   double        residual_tol     = (public_xtra -> residual_tol);
   double        step_tol         = (public_xtra -> step_tol);

   SysFn         feval            = (instance_xtra -> feval);
   KINMem        kin_mem          = (instance_xtra -> kin_mem);
   FILE         *kinsol_file      = (instance_xtra -> kinsol_file);

   long int     *integer_outputs  = (instance_xtra -> integer_outputs);
   long int     *iopt             = (instance_xtra -> int_optional_input);
   double       *ropt             = (instance_xtra -> real_optional_input);

   int           ret              = 0;

   StateFunc(current_state)          = nl_function_eval;
   StateProblemData(current_state)   = problem_data;
   StateTime(current_state)          = t;
   StateDt(current_state)            = dt;
   StateOldDensity(current_state)    = old_density;
   StateOldPressure(current_state)   = old_pressure;
   StateOldSaturation(current_state) = old_saturation;
   StateDensity(current_state)       = density;
   StateSaturation(current_state)    = saturation;
   StateJacEval(current_state)       = richards_jacobian_eval;
   StateJac(current_state)           = jacobian_matrix;
   StatePrecond(current_state)       = precond;
   StateOutflow(current_state)       = outflow; /*sk*/
   StateEvapTrans(current_state)     = evap_trans; /*sk*/
   StateOvrlBcFlx(current_state)     = ovrl_bc_flx; /*sk*/

   if (!amps_Rank(amps_CommWorld))
      fprintf(kinsol_file,"\nKINSOL starting step for time %f\n",t);

   BeginTiming(public_xtra -> time_index);

   ret = KINSol( (void*)kin_mem,        /* Memory allocated above */
	         neq,                   /* Dummy variable here */
	         pressure,               /* Initial guess @ this was "pressure before" */
	         feval,                 /* Nonlinear function */
	         globalization,         /* Globalization method */
	         uscale,                /* Scalings for the variable */
	         fscale,                /* Scalings for the function */
	         residual_tol,          /* Stopping tolerance on func */
	         step_tol,              /* Stop tol. for sucessive steps */
	         NULL,                  /* Constraints */
	         TRUE,                  /* Optional inputs */
	         iopt,                  /* Opt. integer inputs */
	         ropt,                  /* Opt. double inputs */
	         current_state          /* User-supplied input */
	       );

   EndTiming(public_xtra -> time_index);

   integer_outputs[NNI]        += iopt[NNI];
   integer_outputs[NFE]        += iopt[NFE];
   integer_outputs[NBCF]       += iopt[NBCF];
   integer_outputs[NBKTRK]     += iopt[NBKTRK];
   integer_outputs[SPGMR_NLI]  += iopt[SPGMR_NLI];
   integer_outputs[SPGMR_NPE]  += iopt[SPGMR_NPE];
   integer_outputs[SPGMR_NPS]  += iopt[SPGMR_NPS];
   integer_outputs[SPGMR_NCFL] += iopt[SPGMR_NCFL];

   if (!amps_Rank(amps_CommWorld))
      PrintFinalStats(kinsol_file, iopt, integer_outputs);

   if ( ret == KINSOL_SUCCESS || ret == KINSOL_INITIAL_GUESS_OK ) 
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

   long int     *iopt;
   double       *ropt;

   double        eta_value           = public_xtra -> eta_value;
   double        eta_alpha           = public_xtra -> eta_alpha;
   double        eta_gamma           = public_xtra -> eta_gamma;
   double        derivative_epsilon  = public_xtra -> derivative_epsilon;

   Vector       *fscale;
   Vector       *uscale;

   State        *current_state;

   KINSpgmruserAtimesFn   matvec      = public_xtra -> matvec;
   KINSpgmrPrecondFn      pcinit      = public_xtra -> pcinit;
   KINSpgmrPrecondSolveFn pcsolve     = public_xtra -> pcsolve;

   KINMem                 kin_mem;
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
      if (public_xtra -> precond != NULL)
	 instance_xtra -> precond =
	    PFModuleNewInstance(public_xtra -> precond,
				(problem, grid, problem_data, temp_data,
				 NULL, NULL, NULL, NULL, NULL));
      else
	 instance_xtra -> precond = NULL;

      instance_xtra -> nl_function_eval = 
	 PFModuleNewInstance(public_xtra -> nl_function_eval, 
			     (problem, grid, temp_data));

      if (public_xtra -> richards_jacobian_eval != NULL)
	 /* Initialize instance for nonsymmetric matrix */
	 instance_xtra -> richards_jacobian_eval = 
	    PFModuleNewInstance(public_xtra -> richards_jacobian_eval, 
				(problem, grid, temp_data, 0));
      else
	 instance_xtra -> richards_jacobian_eval = NULL;
   }
   else
   {
      if (instance_xtra -> precond != NULL)
	 PFModuleReNewInstance(instance_xtra -> precond,
			        (problem, grid, problem_data, temp_data,
				 NULL, NULL, NULL, NULL, NULL));

      PFModuleReNewInstance(instance_xtra -> nl_function_eval, 
			    (problem, grid, temp_data));

      if (instance_xtra -> richards_jacobian_eval != NULL)
	 PFModuleReNewInstance(instance_xtra -> richards_jacobian_eval, 
			       (problem, grid, temp_data, 0));
   }

   /*-----------------------------------------------------------------------
    * Initialize KINSol input parameters and memory instance
    *-----------------------------------------------------------------------*/

   if ( PFModuleInstanceXtra(this_module) == NULL )
   {
      current_state = ctalloc( State, 1 );

      /* Set up the grid data for the kinsol stuff */
      SetPf2KinsolData(grid, 1);

      /* Initialize KINSol parameters */
      sprintf(filename, "%s.%s", GlobalsOutFileName, "kinsol.log");
      if (!amps_Rank(amps_CommWorld))
	 kinsol_file = fopen( filename, "w" );
      else
	 kinsol_file = NULL;
      instance_xtra -> kinsol_file = kinsol_file;

      /* Initialize KINSol memory */
      kin_mem = (KINMem)KINMalloc(neq, kinsol_file, NULL);

      /* Initialize the gmres linear solver in KINSol */
      KINSpgmr( (void*)kin_mem,        /* Memory allocated above */
		krylov_dimension,      /* Max. Krylov dimension */
		max_restarts,          /* Max. no. of restarts - 0 is none */
		1,                     /* Max. calls to PC Solve w/o PC Set */
		pcinit,                /* PC Set function */
		pcsolve,               /* PC Solve function */
		matvec,                /* ATimes routine */
		current_state          /* User data for PC stuff */
		);

      /* Initialize optional arguments for KINSol */
      iopt = instance_xtra -> int_optional_input;
      ropt = instance_xtra -> real_optional_input;

      iopt[PRINTFL]         = print_flag;
      iopt[MXITER]          = max_iter;
      iopt[PRECOND_NO_INIT] = 0;
      iopt[NNI]             = 0;
      iopt[NFE]             = 0;
      iopt[NBCF]            = 0;
      iopt[NBKTRK]          = 0;
      iopt[ETACHOICE]       = eta_choice;
      iopt[NO_MIN_EPS]      = 0;

      ropt[MXNEWTSTEP]      = 0.0;
      ropt[RELFUNC]         = derivative_epsilon;
      ropt[RELU]            = 0.0;
      ropt[FNORM]           = 0.0;
      ropt[STEPL]           = 0.0;
      ropt[ETACONST]        = eta_value;
      ropt[ETAALPHA]        = eta_alpha;
      /* Put in conditional assignment of eta_gamma since KINSOL aliases */
      /* ETAGAMMA and ETACONST */
      if (eta_value == 0.0) ropt[ETAGAMMA]        = eta_gamma;

      /* Initialize iteration counts */
      for (i=0; i< OPT_SIZE; i++)
	 instance_xtra->integer_outputs[i] = 0;

      /* Scaling vectors*/
      uscale = NewVector(grid, 1, 1);
      InitVectorAll(uscale, 1.0);
      instance_xtra -> uscale = uscale;

      fscale = NewVector(grid, 1, 1);
      InitVectorAll(fscale, 1.0);
      instance_xtra -> fscale = fscale;

      instance_xtra -> feval = KINSolFunctionEval;
      instance_xtra -> kin_mem = kin_mem;
      instance_xtra -> current_state = current_state;
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


   if (instance_xtra)
   {
      PFModuleFreeInstance((instance_xtra -> nl_function_eval));
      if (instance_xtra -> richards_jacobian_eval != NULL)
      {
         PFModuleFreeInstance((instance_xtra -> richards_jacobian_eval));
      }
      if (instance_xtra -> precond != NULL)
      {
         PFModuleFreeInstance((instance_xtra -> precond));
      }

      FreeVector(instance_xtra -> uscale);
      FreeVector(instance_xtra -> fscale);

      tfree(instance_xtra -> current_state);

      KINFree((instance_xtra -> kin_mem));

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
	 public_xtra -> eta_choice = ETACONSTANT;
	 public_xtra -> eta_value  
	                 = GetDoubleDefault("Solver.Nonlinear.EtaValue", 1e-4);
	 public_xtra -> eta_alpha = 0.0;
	 public_xtra -> eta_gamma = 0.0;
	 break;
      }
      case 1:
      {
	 public_xtra -> eta_choice = ETACHOICE1;
	 public_xtra -> eta_alpha = 0.0;
	 public_xtra -> eta_gamma = 0.0;
	 break;
      }
      case 2:
      {
	 public_xtra -> eta_choice = ETACHOICE2;
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
	 (public_xtra -> globalization) = INEXACT_NEWTON;
	 break;
      }
      case 1:
      {
	 (public_xtra -> globalization) = LINESEARCH;
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
      (public_xtra -> precond) = NULL;
      (public_xtra -> pcinit)  = NULL;
      (public_xtra -> pcsolve) = NULL;
   }
   else if ( switch_value > 0 )
   {
      (public_xtra -> precond) = PFModuleNewModule(KinsolPC, 
						   (key, switch_name));
      (public_xtra -> pcinit)  = (KINSpgmrPrecondFn)KINSolInitPC;
      (public_xtra -> pcsolve) = (KINSpgmrPrecondSolveFn)KINSolCallPC;
   }
   else
   {
      InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
		 key);
   }
   NA_FreeNameArray(precond_switch_na);

   public_xtra -> nl_function_eval = PFModuleNewModule(NlFunctionEval, ());
   public_xtra -> neq = ((public_xtra -> max_restarts)+1)
                           *(public_xtra -> krylov_dimension);

   if (public_xtra -> matvec != NULL)
      public_xtra -> richards_jacobian_eval = 
                                 PFModuleNewModule(RichardsJacobianEval, ());
   else 
      public_xtra -> richards_jacobian_eval = NULL;

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
      {
         PFModuleFreeModule(public_xtra -> richards_jacobian_eval);
      }
      if (public_xtra -> precond != NULL)
      {
         PFModuleFreeModule(public_xtra -> precond);
      }
      PFModuleFreeModule(public_xtra -> nl_function_eval);

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

   PFModule             *precond       = (instance_xtra -> precond);
   PFModule             *jacobian_eval = (instance_xtra -> 
					  richards_jacobian_eval);

   int sz = 0;
   
   if (jacobian_eval != NULL)
   {
      sz += PFModuleSizeOfTempData(jacobian_eval);
   }
   if (precond != NULL)
   {
      sz += PFModuleSizeOfTempData(precond);
   }

   return sz;
}
