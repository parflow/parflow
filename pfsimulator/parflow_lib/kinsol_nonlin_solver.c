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

#include "parflow.h"
#include "kinsol_dependences.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int max_iter;
  int krylov_dimension;
  int max_restarts;
  int print_flag;
  int eta_choice;
  int globalization;
  int neq;
  int time_index;

  double residual_tol;
  double step_tol;
  double eta_value;
  double eta_alpha;
  double eta_gamma;
  double derivative_epsilon;

  PFModule *precond;
  PFModule *nl_function_eval;
  PFModule *richards_jacobian_eval;

#if defined (PARFLOW_HAVE_SUNDIALS)
  KINLsJacTimesVecFn matvec;
  KINLsPrecSetupFn pcinit;
  KINLsPrecSolveFn pcsolve;
#else
  KINSpgmruserAtimesFn matvec;
  KINSpgmrPrecondFn pcinit;
  KINSpgmrPrecondSolveFn pcsolve;
#endif
} PublicXtra;

typedef struct {
  PFModule  *precond;
  PFModule  *nl_function_eval;
  PFModule  *richards_jacobian_eval;

  Matrix   *jacobian_matrix;
  Matrix   *jacobian_matrix_C;

  State    *current_state;
  FILE     *kinsol_file;

#if defined (PARFLOW_HAVE_SUNDIALS)
/* SUNDIALS context object */
  SUNContext sunctx;
/* SUNDIALS uses (void *) for Kinsol memory block */
  void * kin_mem;
/* function eval */
  KINSysFn feval;

/* Linear Solver */
  SUNLinearSolver LS;

/* output statistics variables - current and total statistics */
  long int num_nonlin_iters, tot_nonlin_iters;
  long int num_lin_iters, tot_lin_iters;
  long int num_fevals, tot_fevals;
  long int num_pc_setups, tot_pc_setups;
  long int num_pc_solves, tot_pc_solves;
  long int num_lin_conv_fails, tot_lin_conv_fails;
  long int num_beta_cond_fails, tot_beta_cond_fails;
  long int num_backtracks, tot_backtracks;

/* N_Vector objects for kinsol */
  N_Vector uscale;
  N_Vector fscale;
/* N_Vector for pressure variable.
 * This acts as a container for Parflow's Vector pointer
 * for the pressure variable.
 */
  N_Vector pf_n_pressure;
#else
  long int integer_outputs[OPT_SIZE];

  long int int_optional_input[OPT_SIZE];
  double real_optional_input[OPT_SIZE];

  KINMem kin_mem;
  SysFn feval;

  Vector   *uscale;
  Vector   *fscale;
#endif
} InstanceXtra;


/*--------------------------------------------------------------------------
 * Auxiliary functions for interfacing with outside software
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * KINSolInitPC
 *--------------------------------------------------------------------------*/
#if defined (PARFLOW_HAVE_SUNDIALS)
int  KINSolInitPC(
                  N_Vector pf_n_pressure,
                  N_Vector pf_n_uscale,
                  N_Vector pf_n_fval,
                  N_Vector pf_n_fscale,
                  void *   current_state)
{
  Vector      *pressure = N_VectorData(pf_n_pressure);

  (void)pf_n_uscale;
  (void)pf_n_fval;
  (void)pf_n_fscale;
#else
int  KINSolInitPC(
                  int       neq,
                  N_Vector  pressure,
                  N_Vector  uscale,
                  N_Vector  fval,
                  N_Vector  fscale,
                  N_Vector  vtemp1,
                  N_Vector  vtemp2,
                  void *    nl_function,
                  double    uround,
                  long int *nfePtr,
                  void *    current_state)
{
  (void)neq;
  (void)uscale;
  (void)fval;
  (void)fscale;
  (void)vtemp1;
  (void)vtemp2;
  (void)nl_function;
  (void)uround;
  (void)nfePtr;
#endif

  PFModule    *precond = StatePrecond(((State*)current_state));
  ProblemData *problem_data = StateProblemData(((State*)current_state));
  Vector      *saturation = StateSaturation(((State*)current_state));
  Vector      *density = StateDensity(((State*)current_state));
  Vector      *old_pressure = StateOldPressure(((State*)current_state));
  double dt = StateDt(((State*)current_state));
  double time = StateTime(((State*)current_state));

  /* The preconditioner module initialized here is the KinsolPC module
   * itself */

  PFModuleReNewInstanceType(KinsolPCInitInstanceXtraInvoke, precond, (NULL, NULL, NULL, problem_data, NULL,
                                                                      pressure, old_pressure, saturation, density, dt, time));
  return(0);
}


/*--------------------------------------------------------------------------
 * KINSolCallPC
 * This is actually doing an in-place precon solve. Are all the other arguments
 * needed? - DOK
 *--------------------------------------------------------------------------*/
#if defined (PARFLOW_HAVE_SUNDIALS)
int   KINSolCallPC(
                   N_Vector pf_n_pressure,
                   N_Vector pf_n_uscale,
                   N_Vector pf_n_fval,
                   N_Vector pf_n_fscale,
                   N_Vector pf_n_vtem,
                   void *   current_state)
{
  (void)pf_n_pressure;
  (void)pf_n_uscale;
  (void)pf_n_fval;
  (void)pf_n_fscale;
  Vector      *vtem = N_VectorData(pf_n_vtem);
#else
int   KINSolCallPC(
                   int       neq,
                   N_Vector  pressure,
                   N_Vector  uscale,
                   N_Vector  fval,
                   N_Vector  fscale,
                   N_Vector  vtem,
                   N_Vector  ftem,
                   void *    nl_function,
                   double    uround,
                   long int *nfePtr,
                   void *    current_state)
{
  (void)neq;
  (void)pressure;
  (void)uscale;
  (void)fval;
  (void)fscale;
  (void)ftem;
  (void)nl_function;
  (void)uround;
  (void)nfePtr;
#endif
  PFModule *precond = StatePrecond((State*)current_state);

  /* The preconditioner module invoked here is the KinsolPC module
   * itself */

  PFModuleInvokeType(KinsolPCInvoke, precond, (vtem));

  return(0);
}

#if defined (PARFLOW_HAVE_SUNDIALS)
static void PrintFinalStats(FILE * out_file, InstanceXtra *instance_xtra)
{
  if (instance_xtra)
  {
    fprintf(out_file, "\n-------------------------------------------------- \n");
    fprintf(out_file, "                    Iteration             Total\n");
    fprintf(out_file, "Nonlin. Its.:           %5ld             %5ld\n",
            instance_xtra->num_nonlin_iters, instance_xtra->tot_nonlin_iters);
    fprintf(out_file, "Lin. Its.:              %5ld             %5ld\n",
            instance_xtra->num_lin_iters, instance_xtra->tot_lin_iters);
    fprintf(out_file, "Func. Evals.:           %5ld             %5ld\n",
            instance_xtra->num_fevals, instance_xtra->tot_fevals);
    fprintf(out_file, "PC Evals.:              %5ld             %5ld\n",
            instance_xtra->num_pc_setups, instance_xtra->tot_pc_setups);
    fprintf(out_file, "PC Solves:              %5ld             %5ld\n",
            instance_xtra->num_pc_solves, instance_xtra->tot_pc_solves);
    fprintf(out_file, "Lin. Conv. Fails:       %5ld             %5ld\n",
            instance_xtra->num_lin_conv_fails, instance_xtra->tot_lin_conv_fails);
    fprintf(out_file, "Beta Cond. Fails:       %5ld             %5ld\n",
            instance_xtra->num_beta_cond_fails, instance_xtra->num_beta_cond_fails);
    fprintf(out_file, "Backtracks:             %5ld             %5ld\n",
            instance_xtra->num_backtracks, instance_xtra->tot_backtracks);
    fprintf(out_file, "-------------------------------------------------- \n");
    fflush(out_file);
  }
}
#else
void PrintFinalStats(
                     FILE *    out_file,
                     long int *integer_outputs_now,
                     long int *integer_outputs_total)
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
  fprintf(out_file, "-------------------------------------------------- \n");
  fflush(out_file);
}
#endif
/*--------------------------------------------------------------------------
 * KinsolNonlinSolver
 *--------------------------------------------------------------------------*/

int KinsolNonlinSolver(Vector *pressure, Vector *density, Vector *old_density, Vector *saturation, Vector *old_saturation, double t, double dt, ProblemData *problem_data, Vector *old_pressure, Vector *evap_trans, Vector *ovrl_bc_flx, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity)
{
  PFModule     *this_module = ThisPFModule;
  PublicXtra   *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  Matrix       *jacobian_matrix = (instance_xtra->jacobian_matrix);
  Matrix       *jacobian_matrix_C = (instance_xtra->jacobian_matrix_C);

  PFModule  *nl_function_eval = instance_xtra->nl_function_eval;
  PFModule  *richards_jacobian_eval = instance_xtra->richards_jacobian_eval;
  PFModule  *precond = instance_xtra->precond;

  State        *current_state = (instance_xtra->current_state);

  int globalization = (public_xtra->globalization);

#if defined (PARFLOW_HAVE_SUNDIALS)
  N_Vector uscale = (instance_xtra->uscale);
  N_Vector fscale = (instance_xtra->fscale);

  /* SUNDIALS context object */
  void * kin_mem = (instance_xtra->kin_mem);

  /* N_Vector for pressure variable */
  N_Vector pf_n_pressure = (instance_xtra->pf_n_pressure);
#else
  Vector       *uscale = (instance_xtra->uscale);
  Vector       *fscale = (instance_xtra->fscale);

  int neq = (public_xtra->neq);

  double residual_tol = (public_xtra->residual_tol);
  double step_tol = (public_xtra->step_tol);

  SysFn feval = (instance_xtra->feval);
  KINMem kin_mem = (instance_xtra->kin_mem);

  long int     *integer_outputs = (instance_xtra->integer_outputs);
  long int     *iopt = (instance_xtra->int_optional_input);
  double       *ropt = (instance_xtra->real_optional_input);
#endif

  FILE         *kinsol_file = (instance_xtra->kinsol_file);

  int ret = 0;

  StateFunc(current_state) = nl_function_eval;
  StateProblemData(current_state) = problem_data;
  StateTime(current_state) = t;
  StateDt(current_state) = dt;
  StateOldDensity(current_state) = old_density;
  StateOldPressure(current_state) = old_pressure;
  StateOldSaturation(current_state) = old_saturation;
  StateDensity(current_state) = density;
  StateSaturation(current_state) = saturation;
  StateJacEval(current_state) = richards_jacobian_eval;
  StateJac(current_state) = jacobian_matrix;
  StateJacC(current_state) = jacobian_matrix_C;           //dok
  StatePrecond(current_state) = precond;
  StateEvapTrans(current_state) = evap_trans;       /*sk*/
  StateOvrlBcFlx(current_state) = ovrl_bc_flx;      /*sk*/
  StateXvel(current_state) = x_velocity;           //jjb
  StateYvel(current_state) = y_velocity;           //jjb
  StateZvel(current_state) = z_velocity;           //jjb

  if (!amps_Rank(amps_CommWorld))
    fprintf(kinsol_file, "\nKINSOL starting step for time %f\n", t);

  BeginTiming(public_xtra->time_index);

#if defined (PARFLOW_HAVE_SUNDIALS)
  /* Attach parflow Vector to N_Vector object */
  N_VectorData(pf_n_pressure) = pressure;

  /* Call KINSol */
  ret = KINSol(kin_mem,                      /* Memory allocated above */
               pf_n_pressure,      /* Initial guess @ this was "pressure before" */
               globalization,                /* NonLin. solver strategy. Here we use Newton with globalization */
               uscale,             /* Scalings for the variable */
               fscale              /* Scalings for the function */
               );

  EndTiming(public_xtra->time_index);

  /* update statistics */
  KINGetNumNonlinSolvIters(kin_mem, &(instance_xtra->num_nonlin_iters));
  KINGetNumLinIters(kin_mem, &(instance_xtra->num_lin_iters));
  KINGetNumFuncEvals(kin_mem, &(instance_xtra->num_fevals));
  KINGetNumPrecEvals(kin_mem, &(instance_xtra->num_pc_setups));
  KINGetNumPrecSolves(kin_mem, &(instance_xtra->num_pc_solves));
  KINGetNumLinConvFails(kin_mem, &(instance_xtra->num_lin_conv_fails));
  KINGetNumBetaCondFails(kin_mem, &(instance_xtra->num_beta_cond_fails));
  KINGetNumBacktrackOps(kin_mem, &(instance_xtra->num_backtracks));

  /* running totals */
  instance_xtra->tot_nonlin_iters += instance_xtra->num_nonlin_iters;
  instance_xtra->tot_lin_iters += instance_xtra->num_lin_iters;
  instance_xtra->tot_fevals += instance_xtra->num_fevals;
  instance_xtra->tot_pc_setups += instance_xtra->num_pc_setups;
  instance_xtra->tot_pc_solves += instance_xtra->num_pc_solves;
  instance_xtra->tot_lin_conv_fails += instance_xtra->num_lin_conv_fails;
  instance_xtra->tot_beta_cond_fails += instance_xtra->num_beta_cond_fails;
  instance_xtra->tot_backtracks += instance_xtra->num_backtracks;

  if (!amps_Rank(amps_CommWorld))
    PrintFinalStats(kinsol_file, instance_xtra);

  if (ret == KIN_SUCCESS || ret == KIN_INITIAL_GUESS_OK)
  {
    ret = 0;
  }
#else
  ret = KINSol((void*)kin_mem,          /* Memory allocated above */
               neq,                     /* Dummy variable here */
               pressure,                /* Initial guess @ this was "pressure before" */
               feval,                   /* Nonlinear function */
               globalization,           /* Globalization method */
               uscale,                  /* Scalings for the variable */
               fscale,                  /* Scalings for the function */
               residual_tol,            /* Stopping tolerance on func */
               step_tol,                /* Stop tol. for successive steps */
               NULL,                    /* Constraints */
               TRUE,                    /* Optional inputs */
               iopt,                    /* Opt. integer inputs */
               ropt,                    /* Opt. double inputs */
               current_state            /* User-supplied input */
               );

  EndTiming(public_xtra->time_index);

  integer_outputs[NNI] += iopt[NNI];
  integer_outputs[NFE] += iopt[NFE];
  integer_outputs[NBCF] += iopt[NBCF];
  integer_outputs[NBKTRK] += iopt[NBKTRK];
  integer_outputs[SPGMR_NLI] += iopt[SPGMR_NLI];
  integer_outputs[SPGMR_NPE] += iopt[SPGMR_NPE];
  integer_outputs[SPGMR_NPS] += iopt[SPGMR_NPS];
  integer_outputs[SPGMR_NCFL] += iopt[SPGMR_NCFL];

  if (!amps_Rank(amps_CommWorld))
    PrintFinalStats(kinsol_file, iopt, integer_outputs);

  if (ret == KINSOL_SUCCESS || ret == KINSOL_INITIAL_GUESS_OK)
  {
    ret = 0;
  }
#endif

  return(ret);
}

/*--------------------------------------------------------------------------
 * KinsolNonlinSolverInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *KinsolNonlinSolverInitInstanceXtra(
                                              Problem *    problem,
                                              Grid *       grid,
                                              Grid *       grid2d,
                                              ProblemData *problem_data,
                                              double *     temp_data)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;

  int max_restarts = public_xtra->max_restarts;
  int krylov_dimension = public_xtra->krylov_dimension;
  int max_iter = public_xtra->max_iter;
  int eta_choice = public_xtra->eta_choice;

  double eta_value = public_xtra->eta_value;
  double eta_alpha = public_xtra->eta_alpha;
  double eta_gamma = public_xtra->eta_gamma;
  double derivative_epsilon = public_xtra->derivative_epsilon;

  State        *current_state;

#if defined (PARFLOW_HAVE_SUNDIALS)
  N_Vector fscale;
  N_Vector uscale;
  N_Vector pf_n_pressure;

  KINLsJacTimesVecFn matvec = public_xtra->matvec;
  KINLsPrecSetupFn pcinit = public_xtra->pcinit;
  KINLsPrecSolveFn pcsolve = public_xtra->pcsolve;
  /* Sundials context, pointer to memory and linear solver */
  SUNContext sunctx;
  void *kin_mem;
  SUNLinearSolver LS;
#else
  int neq = public_xtra->neq;
  int print_flag = public_xtra->print_flag;

  long int     *iopt;
  double       *ropt;

  Vector       *fscale;
  Vector       *uscale;

  KINSpgmruserAtimesFn matvec = public_xtra->matvec;
  KINSpgmrPrecondFn pcinit = public_xtra->pcinit;
  KINSpgmrPrecondSolveFn pcsolve = public_xtra->pcsolve;

  /* pointer to memory */
  KINMem kin_mem;
#endif

  FILE                  *kinsol_file;
  char filename[1024];

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize module instances
   *-----------------------------------------------------------------------*/

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    if (public_xtra->precond != NULL)
      instance_xtra->precond =
        PFModuleNewInstanceType(KinsolPCInitInstanceXtraInvoke, public_xtra->precond,
                                (problem, grid, grid2d, problem_data, temp_data,
                                 NULL, NULL, NULL, NULL, 0, 0));
    else
      instance_xtra->precond = NULL;

    instance_xtra->nl_function_eval =
      PFModuleNewInstanceType(NlFunctionEvalInitInstanceXtraInvoke, public_xtra->nl_function_eval,
                              (problem, grid, grid2d, temp_data));

    if (public_xtra->richards_jacobian_eval != NULL)
      /* Initialize instance for nonsymmetric matrix */
      instance_xtra->richards_jacobian_eval =
        PFModuleNewInstanceType(RichardsJacobianEvalInitInstanceXtraInvoke, public_xtra->richards_jacobian_eval,
                                (problem, grid, grid2d, problem_data, temp_data, 0));
    else
      instance_xtra->richards_jacobian_eval = NULL;
  }
  else
  {
    if (instance_xtra->precond != NULL)
      PFModuleReNewInstanceType(KinsolPCInitInstanceXtraInvoke,
                                instance_xtra->precond,
                                (problem, grid, grid2d, problem_data, temp_data,
                                 NULL, NULL, NULL, NULL, 0, 0));

    PFModuleReNewInstanceType(NlFunctionEvalInitInstanceXtraInvoke, instance_xtra->nl_function_eval,
                              (problem, grid, grid2d, temp_data));

    if (instance_xtra->richards_jacobian_eval != NULL)
      PFModuleReNewInstanceType(RichardsJacobianEvalInitInstanceXtraInvoke, instance_xtra->richards_jacobian_eval,
                                (problem, grid, grid2d, problem_data, temp_data, 0));
  }

  /*-----------------------------------------------------------------------
   * Initialize KINSol input parameters and memory instance
   *-----------------------------------------------------------------------*/

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    current_state = ctalloc(State, 1);

    /* Initialize KINSol parameters */

    /* kinsol log file */
    sprintf(filename, "%s.%s", GlobalsOutFileName, "kinsol.log");
    if (!amps_Rank(amps_CommWorld))
      kinsol_file = fopen(filename, "w");
    else
      kinsol_file = NULL;
    instance_xtra->kinsol_file = kinsol_file;

#if defined (PARFLOW_HAVE_SUNDIALS)
    /* Create the SUNDIALS context that all SUNDIALS objects require */
    /* This needs to be created once? So perhaps should be created elsewhere upstream */
    SUNContext_Create(amps_CommWorld, &sunctx);

    /* Initialize empty N_Vector container for pressure variable */
    pf_n_pressure = PF_NVNewEmpty(sunctx);
    instance_xtra->pf_n_pressure = pf_n_pressure;

    /* Initialize KINSol memory and allocate KINSol vectors */
    /* Initialize scaling vectors now, so we can use as template to initialize kinsol */
    uscale = PF_NVNew(sunctx, grid, 1);
    InitVectorAll(N_VectorData(uscale), 1.0);
    instance_xtra->uscale = uscale;

    fscale = PF_NVNew(sunctx, grid, 1);
    InitVectorAll(N_VectorData(fscale), 1.0);
    instance_xtra->fscale = fscale;

    /* Create KINSol memory */
    kin_mem = KINCreate(sunctx);
    /* Initialize KINSol memory. Use uscale as N_Vector template */
    KINInit(kin_mem, KINSolFunctionEval, uscale);

    /* Set KINSol options */
    /* set user data for problem function */
    KINSetUserData(kin_mem, current_state);
    /* Max. number of nonlinear iterations */
    KINSetNumMaxIters(kin_mem, max_iter);
    /* No initial call to the preconditioner setup */
    KINSetNoInitSetup(kin_mem, 0); /* same as default */
    /* Max nonlin. iterations without PC setup - default is 10 */
    KINSetMaxSetupCalls(kin_mem, 1);
    /* method for computing eta */
    KINSetEtaForm(kin_mem, eta_choice);
    KINSetEtaConstValue(kin_mem, eta_value);
    KINSetEtaParams(kin_mem, eta_gamma, eta_alpha);
    /* Max. scaled length of Newton step. Set to 0.0 to use default */
    KINSetMaxNewtonStep(kin_mem, 0.0);
    /* Relative error in Jacobian Approximation by difference quotient */
    KINSetRelErrFunc(kin_mem, derivative_epsilon);
    /* Flag to constrain residual tolerance.
     * Lower bound on epsilon - default is 0 == false == constrain
     */
    KINSetNoMinEps(kin_mem, 0);
    /* NL function norm stopping tolerance. set to 0.0 for default */
    KINSetFuncNormTol(kin_mem, public_xtra->residual_tol);

    KINSetScaledStepTol(kin_mem, public_xtra->step_tol);

    /* Create SUNDIALS linear solver object for kinsol */
    LS = SUNLinSol_SPGMR(uscale, SUN_PREC_RIGHT, krylov_dimension, sunctx);
    SUNLinSol_SPGMRSetMaxRestarts(LS, max_restarts);
    /* Attach linear solver to KINSol */
    KINSetLinearSolver(kin_mem, LS, NULL);
    KINSetPreconditioner(kin_mem, pcinit, pcsolve);
    KINSetJacTimesVecFn(kin_mem, matvec);

    /* Initialize total statistics counts*/
    instance_xtra->tot_nonlin_iters = 0;
    instance_xtra->tot_lin_iters = 0;
    instance_xtra->tot_fevals = 0;
    instance_xtra->tot_pc_setups = 0;
    instance_xtra->tot_pc_solves = 0;
    instance_xtra->tot_lin_conv_fails = 0;
    instance_xtra->tot_beta_cond_fails = 0;
    instance_xtra->tot_backtracks = 0;

    instance_xtra->sunctx = sunctx;
    instance_xtra->kin_mem = kin_mem;
    instance_xtra->LS = LS;
#else
    /* Set up the grid data for the kinsol stuff */
    SetPf2KinsolData(grid, 1);
    /* Initialize KINSol memory and allocate KINSol vectors*/
    kin_mem = (KINMem)KINMalloc(neq, kinsol_file, NULL);

    /* Initialize the gmres linear solver in KINSol */
    KINSpgmr((void*)kin_mem,           /* Memory allocated above */
             krylov_dimension,         /* Max. Krylov dimension */
             max_restarts,             /* Max. no. of restarts - 0 is none */
             1,                        /* Max. calls to PC Solve w/o PC Set */
             pcinit,                   /* PC Set function */
             pcsolve,                  /* PC Solve function */
             matvec,                   /* ATimes routine */
             current_state             /* User data for PC stuff */
             );

    /* Initialize optional arguments for KINSol */
    iopt = instance_xtra->int_optional_input;
    ropt = instance_xtra->real_optional_input;

    // Only print on rank 0
    iopt[PRINTFL] = amps_Rank(amps_CommWorld) ? 0 : print_flag;
    iopt[MXITER] = max_iter;
    iopt[PRECOND_NO_INIT] = 0;
    iopt[NNI] = 0;
    iopt[NFE] = 0;
    iopt[NBCF] = 0;
    iopt[NBKTRK] = 0;
    iopt[ETACHOICE] = eta_choice;
    iopt[NO_MIN_EPS] = 0;

    ropt[MXNEWTSTEP] = 0.0;
    ropt[RELFUNC] = derivative_epsilon;
    ropt[RELU] = 0.0;
    ropt[FNORM] = 0.0;
    ropt[STEPL] = 0.0;
    ropt[ETACONST] = eta_value;
    ropt[ETAALPHA] = eta_alpha;
    /* Put in conditional assignment of eta_gamma since KINSOL aliases */
    /* ETAGAMMA and ETACONST */
    if (eta_value == 0.0)
      ropt[ETAGAMMA] = eta_gamma;

    /* Initialize iteration counts */
    for (int i = 0; i < OPT_SIZE; i++)
      instance_xtra->integer_outputs[i] = 0;

    /* Scaling vectors*/
    uscale = NewVectorType(grid, 1, 1, vector_cell_centered);
    InitVectorAll(uscale, 1.0);
    instance_xtra->uscale = uscale;

    fscale = NewVectorType(grid, 1, 1, vector_cell_centered);
    InitVectorAll(fscale, 1.0);
    instance_xtra->fscale = fscale;

    instance_xtra->kin_mem = kin_mem;
#endif

    instance_xtra->feval = KINSolFunctionEval;
    instance_xtra->current_state = current_state;
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * KinsolNonlinSolverFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  KinsolNonlinSolverFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    PFModuleFreeInstance((instance_xtra->nl_function_eval));
    if (instance_xtra->richards_jacobian_eval != NULL)
    {
      PFModuleFreeInstance((instance_xtra->richards_jacobian_eval));
    }
    if (instance_xtra->precond != NULL)
    {
      PFModuleFreeInstance((instance_xtra->precond));
    }

#if defined (PARFLOW_HAVE_SUNDIALS)
    PF_NVDestroy(instance_xtra->uscale);
    PF_NVDestroy(instance_xtra->fscale);
    PF_NVDestroy(instance_xtra->pf_n_pressure);

    /* free SUNDIALS context, memory and linear solver */
    KINFree(&(instance_xtra->kin_mem));
    SUNLinSolFree((instance_xtra->LS));
    SUNContext_Free(&(instance_xtra->sunctx));
#else
    FreeVector(instance_xtra->uscale);
    FreeVector(instance_xtra->fscale);

    /* free kinsol memory */
    KINFree((instance_xtra->kin_mem));
#endif
    tfree(instance_xtra->current_state);

    if (instance_xtra->kinsol_file)
      fclose((instance_xtra->kinsol_file));

    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * KinsolNonlinSolverNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *KinsolNonlinSolverNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  char          *switch_name;
  char key[IDB_MAX_KEY_LEN];
  int switch_value;

  NameArray switch_na;
  NameArray verbosity_switch_na;
  NameArray eta_switch_na;
  NameArray globalization_switch_na;
  NameArray precond_switch_na;

  public_xtra = ctalloc(PublicXtra, 1);

  sprintf(key, "Solver.Nonlinear.ResidualTol");
  (public_xtra->residual_tol) = GetDoubleDefault(key, 1e-7);
  sprintf(key, "Solver.Nonlinear.StepTol");
  (public_xtra->step_tol) = GetDoubleDefault(key, 1e-7);

  sprintf(key, "Solver.Nonlinear.MaxIter");
  (public_xtra->max_iter) = GetIntDefault(key, 15);
  sprintf(key, "Solver.Linear.KrylovDimension");
  (public_xtra->krylov_dimension) = GetIntDefault(key, 10);
  sprintf(key, "Solver.Linear.MaxRestarts");
  (public_xtra->max_restarts) = GetIntDefault(key, 0);

  verbosity_switch_na = NA_NewNameArray("NoVerbosity LowVerbosity "
                                        "NormalVerbosity HighVerbosity");
  sprintf(key, "Solver.Nonlinear.PrintFlag");
  switch_name = GetStringDefault(key, "LowVerbosity");
  (public_xtra->print_flag) = NA_NameToIndexExitOnError(verbosity_switch_na,
                                                        switch_name,
                                                        key);
  NA_FreeNameArray(verbosity_switch_na);

  eta_switch_na = NA_NewNameArray("EtaConstant Walker1 Walker2");
  sprintf(key, "Solver.Nonlinear.EtaChoice");
  switch_name = GetStringDefault(key, "Walker2");
  switch_value = NA_NameToIndexExitOnError(eta_switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
#if defined (PARFLOW_HAVE_SUNDIALS)
      public_xtra->eta_choice = KIN_ETACONSTANT;
#else
      public_xtra->eta_choice = ETACONSTANT;
#endif
      public_xtra->eta_value
        = GetDoubleDefault("Solver.Nonlinear.EtaValue", 1e-4);
      public_xtra->eta_alpha = 0.0;
      public_xtra->eta_gamma = 0.0;
      break;
    }

    case 1:
    {
#if defined (PARFLOW_HAVE_SUNDIALS)
      public_xtra->eta_choice = KIN_ETACHOICE1;
#else
      public_xtra->eta_choice = ETACHOICE1;
#endif

      public_xtra->eta_alpha = 0.0;
      public_xtra->eta_gamma = 0.0;
      break;
    }

    case 2:
    {
#if defined (PARFLOW_HAVE_SUNDIALS)
      public_xtra->eta_choice = KIN_ETACHOICE2;
#else
      public_xtra->eta_choice = ETACHOICE2;
#endif
      public_xtra->eta_alpha
        = GetDoubleDefault("Solver.Nonlinear.EtaAlpha", 2.0);
      public_xtra->eta_gamma
        = GetDoubleDefault("Solver.Nonlinear.EtaGamma", 0.9);
      public_xtra->eta_value = 0.0;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(eta_switch_na);

  switch_na = NA_NewNameArray("False True");
  sprintf(key, "Solver.Nonlinear.UseJacobian");
  switch_name = GetStringDefault(key, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      (public_xtra->matvec) = NULL;
      break;
    }

    case 1:
    {
      (public_xtra->matvec) = KINSolMatVec;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(switch_na);

  sprintf(key, "Solver.Nonlinear.DerivativeEpsilon");
  (public_xtra->derivative_epsilon) = GetDoubleDefault(key, 1e-7);

  globalization_switch_na = NA_NewNameArray("InexactNewton LineSearch");
  sprintf(key, "Solver.Nonlinear.Globalization");
  switch_name = GetStringDefault(key, "LineSearch");
  switch_value = NA_NameToIndexExitOnError(globalization_switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
#if defined (PARFLOW_HAVE_SUNDIALS)
      (public_xtra->globalization) = KIN_NONE;
#else
      (public_xtra->globalization) = INEXACT_NEWTON;
#endif
      break;
    }

    case 1:
    {
#if defined (PARFLOW_HAVE_SUNDIALS)
      (public_xtra->globalization) = KIN_LINESEARCH;
#else
      (public_xtra->globalization) = LINESEARCH;
#endif
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(globalization_switch_na);

  precond_switch_na = NA_NewNameArray("NoPC MGSemi SMG PFMG PFMGOctree");
  sprintf(key, "Solver.Linear.Preconditioner");
  switch_name = GetStringDefault(key, "MGSemi");
  switch_value = NA_NameToIndexExitOnError(precond_switch_na, switch_name, key);
  if (switch_value == 0)
  {
    (public_xtra->precond) = NULL;
    (public_xtra->pcinit) = NULL;
    (public_xtra->pcsolve) = NULL;
  }
  else if (switch_value > 0)
  {
    (public_xtra->precond) = PFModuleNewModuleType(
                                                   KinsolPCNewPublicXtraInvoke,
                                                   KinsolPC,
                                                   (key, switch_name));
#if defined (PARFLOW_HAVE_SUNDIALS)
    (public_xtra->pcinit) = KINSolInitPC;
    (public_xtra->pcsolve) = KINSolCallPC;
#else
    (public_xtra->pcinit) = (KINSpgmrPrecondFn)KINSolInitPC;
    (public_xtra->pcsolve) = (KINSpgmrPrecondSolveFn)KINSolCallPC;
#endif
  }
  else
  {
    InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
  }
  NA_FreeNameArray(precond_switch_na);

  public_xtra->nl_function_eval = PFModuleNewModule(NlFunctionEval, ());
  public_xtra->neq = ((public_xtra->max_restarts) + 1)
                     * (public_xtra->krylov_dimension);

  if (public_xtra->matvec != NULL)
    public_xtra->richards_jacobian_eval =
      PFModuleNewModuleType(
                            RichardsJacobianEvalNewPublicXtraInvoke,
                            RichardsJacobianEval,
                            ("Solver.Nonlinear.Jacobian"));
  else
    public_xtra->richards_jacobian_eval = NULL;

  (public_xtra->time_index) = RegisterTiming("KINSol");

  PFModulePublicXtra(this_module) = public_xtra;

  return this_module;
}

/*-------------------------------------------------------------------------
 * KinsolNonlinSolverFreePublicXtra
 *-------------------------------------------------------------------------*/

void  KinsolNonlinSolverFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    if (public_xtra->richards_jacobian_eval != NULL)
    {
      PFModuleFreeModule(public_xtra->richards_jacobian_eval);
    }
    if (public_xtra->precond != NULL)
    {
      PFModuleFreeModule(public_xtra->precond);
    }
    PFModuleFreeModule(public_xtra->nl_function_eval);

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * KinsolNonlinSolverSizeOfTempData
 *--------------------------------------------------------------------------*/

int  KinsolNonlinSolverSizeOfTempData()
{
  PFModule             *this_module = ThisPFModule;
  InstanceXtra         *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  PFModule             *precond = (instance_xtra->precond);
  PFModule             *jacobian_eval = (instance_xtra->
                                         richards_jacobian_eval);

  int sz = 0;

  /* SGS temp data */

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
