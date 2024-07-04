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
/*****************************************************************************
*
* Chebyshev polynomial preconditioned conjugate gradient solver (Omin).
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  PFModule  *precond;

  int max_iter;
  int degree;
  double ia, ib;           /* NOT USED */
  int power;               /* NOT USED */
  int two_norm;

  int time_index;
} PublicXtra;

typedef struct {
  PFModule *precond;

  /* InitInstanceXtra arguments */
  Grid     *grid;
  Matrix   *A;
  double    *temp_data;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * PPCG
 *--------------------------------------------------------------------------
 *
 * This function implements a Chebyshev preconditioned conjugate
 * gradient algorithm to solve the linear system Ax=b. The initial
 * preconditioning matrix, C(A), is the identity. The resulting
 * iterations are simply standard CG iterations. Using intermediate
 * results from these iterations, a tridiagonal matrix is built
 * from which estimates of the minimum and maximum eigenvalues of
 * A can be determined (Lanczos algorithm). Using these eigenvalues,
 * a Chebyshev preconditioning matrix of order 3 is derived. After
 * a fixed number of iterations with this system (currently 5),
 * eigenvalues of A are again estimated, and a new Chebyshev polynomial
 * of degree specified by the user in the run-time parameter file is
 * determined. From this point on, new eigenvalue estimates are
 * made every 5 iterations. The degree of the polynomial does not
 * change. Iterations continue in this way until convergence.
 *
 * We use the following convergence test (see Ashby, Holst, Manteuffel, and
 * Saylor):
 *
 *       ||e||_A                           ||r||_C
 *       -------  <=  [kappa_A(C*A)]^(1/2) -------  < tol
 *       ||x||_A                           ||b||_C
 *
 * where we let ||b||_C = ||b||_2.
 * We implement the test as:
 *
 *       cond * gamma =  [kappa_A(C*A)] * <C*r,r> <  (tol^2)*<b,b> = eps
 *
 *--------------------------------------------------------------------------*/

void     PPCG(
              Vector *x,
              Vector *b,
              double  tol,
              int     zero)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int max_iter = (public_xtra->max_iter);
  int degree = (public_xtra->degree);
  double ia = (public_xtra->ia);
  double ib = (public_xtra->ib);
  int two_norm = (public_xtra->two_norm);

  PFModule  *precond = (instance_xtra->precond);

  Matrix    *A = (instance_xtra->A);

  Vector    *r;
  Vector    *p = NULL;
  Vector    *s = NULL;

  double    *alpha_vec;
  double    *beta_vec;
  double    *pdotAp_vec;
  double prod;

  double alpha, beta;
  double gamma, gamma_old;
  double i_prod = 0.0, eps;
  double cond = 1.0;

  int size;

  double b_dot_b;

  int n_degree = 3;
  int degree_array[3];
  int stride[3];
  int degree_index = 0;

  int i = 0;

  double    *norm_log = NULL;
  double    *rel_norm_log = NULL;
  double    *cond_log = NULL;
  int       *restart_log = NULL;


  /*-----------------------------------------------------------------------
   * Initialize some logging variables
   *-----------------------------------------------------------------------*/

  IfLogging(1)
  {
    norm_log = talloc(double, max_iter);
    rel_norm_log = talloc(double, max_iter);
    cond_log = talloc(double, max_iter);
    restart_log = ctalloc(int, max_iter);
  }

  /*-----------------------------------------------------------------------
   * Begin timing
   *-----------------------------------------------------------------------*/

  BeginTiming(public_xtra->time_index);


  /*-----------------------------------------------------------------------
   * Allocate temp vectors
   *-----------------------------------------------------------------------*/
  p = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
  s = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);

  /*-----------------------------------------------------------------------
   * Start ppcg solve
   *-----------------------------------------------------------------------*/

  if (zero)
    InitVector(x, 0.0);

  /* Set up the polynomial degree arrays */
  degree_array[0] = public_xtra->degree;
  degree_array[1] = public_xtra->degree;
  degree_array[2] = public_xtra->degree;
  stride[0] = 5;
  stride[1] = 5;
  stride[2] = 5;
  degree = degree_array[degree_index];

  /* Allocate memory for dynamic arrays */
  alpha_vec = talloc(double, max_iter + 1);
  beta_vec = talloc(double, max_iter + 1);
  pdotAp_vec = talloc(double, max_iter + 1);

  /* The size variable is the number of iterations since last restart */
  size = 0;

  /* Compute endpoints ia=ib = <Ab,b>/<b,b>  */
  Matvec(1.0, A, b, 0.0, s);
  b_dot_b = InnerProd(b, b);
  ia = InnerProd(s, b) / b_dot_b;
  ib = ia;

  /* eps = (tol^2)*<b,b> */
  eps = (tol * tol) * b_dot_b;

  /* r = b - Ax,  (overwrite b with r) */
  Matvec(-1.0, A, x, 1.0, (r = b));

  /* p = C*r */
  PFModuleInvokeType(ChebyshevInvoke, precond, (p, r, 0.0, 1, ia, ib, degree));

  /* gamma = <r,p> */
  gamma = InnerProd(r, p);

  /* Main interation loop */
  while (((i + 1) <= max_iter) && (gamma > 0))
  {
    i++;

    /* Keep track of how many alphas since last restart */
    size++;
    /* s = A*p */
    Matvec(1.0, A, p, 0.0, s);

    /* alpha = gamma / <s,p> */
    prod = InnerProd(s, p);
    alpha = gamma / prod;
    alpha_vec[size - 1] = alpha;
    pdotAp_vec[size - 1] = prod;

    gamma_old = gamma;

    /* x = x + alpha*p */
    Axpy(alpha, p, x);

    /* r = r - alpha*s */
    Axpy(-alpha, s, r);

    /* s = C*r */
    PFModuleInvokeType(ChebyshevInvoke, precond, (s, r, 0.0, 1, ia, ib, degree));

    /* gamma = <r,s> */
    gamma = InnerProd(r, s);

    /* beta = gamma / gamma_old */
    beta = gamma / gamma_old;
    beta_vec[size - 1] = beta;

    /* p = s + beta p */
    Scale(beta, p);
    Axpy(1.0, s, p);

    /* set i_prod for convergence test */
    if (two_norm)
      i_prod = InnerProd(r, r);
    else
      i_prod = gamma * cond;


#if 0
    if (!amps_Rank(amps_CommWorld))
    {
      if (two_norm)
        amps_Printf("Iter (%d): ||r||_2 = %e, ||r||_2/||b||_2 = le\n",
                    i, sqrt(i_prod), (b_dot_b ? sqrt(i_prod / b_dot_b) : 0));
      else
        amps_Printf("Iter (%d): ||Kr||_C = %e, ||Kr||_C/||b||_2 = %e\n",
                    i, sqrt(i_prod), (b_dot_b ? sqrt(i_prod / b_dot_b) : 0));
    }
#endif

    /* log norm info */
    IfLogging(1)
    {
      norm_log[i - 1] = sqrt(i_prod);
      rel_norm_log[i - 1] = b_dot_b ? sqrt(i_prod / b_dot_b) : 0;
      cond_log[i - 1] = cond;
    }

    /* check for convergence */
    if (i_prod < eps)
      break;

    /* Compute new interval [ia, ib]  */
    if ((i % stride[degree_index]) == 0)
    {
      if (degree_index < n_degree - 1)
        degree_index++;
      degree = degree_array[degree_index];
      if (degree > 1)
        NewEndpts(alpha_vec, beta_vec, pdotAp_vec, &size,
                  degree, &ia, &ib, &cond, eps);
    }

    /* If restarting, size will be zero */
    if (size == 0)
    {
#if 1
      if (!amps_Rank(amps_CommWorld))
        amps_Printf(">> Restart: a, b (iter) = %f, %f (%d)\n",
                    ia, ib, i);
#endif

      IfLogging(1)
      restart_log[i] = 1;

      /* p = C*r */
      PFModuleInvokeType(ChebyshevInvoke, precond, (p, r, 0.0, 1, ia, ib, degree));

      /* gamma = <r,p> */
      gamma = InnerProd(r, p);
    }
  }

#if 1
  if (!amps_Rank(amps_CommWorld))
  {
    if (two_norm)
      amps_Printf("Iterations = %d: ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
                  i, sqrt(i_prod), (b_dot_b ? sqrt(i_prod / b_dot_b) : 0));
    else
      amps_Printf("Iterations = %d: ||Kr||_C = %e, ||Kr||_C/||b||_2 = %e\n",
                  i, sqrt(i_prod), (b_dot_b ? sqrt(i_prod / b_dot_b) : 0));
  }
#endif

  /*-----------------------------------------------------------------------
   * Free temp vectors
   *-----------------------------------------------------------------------*/
  FreeVector(s);
  FreeVector(p);

  /*-----------------------------------------------------------------------
   * End timing
   *-----------------------------------------------------------------------*/

  IncFLOPCount(i * 2 - 1);
  EndTiming(public_xtra->time_index);

  /*-----------------------------------------------------------------------
   * Print log.
   *-----------------------------------------------------------------------*/

  IfLogging(1)
  {
    FILE  *log_file;
    int j;

    log_file = OpenLogFile("PCG");

    if (two_norm)
    {
      fprintf(log_file, "Iters       ||r||_2    ||r||_2/||b||_2     Condition    Restart\n");
      fprintf(log_file, "-----    ------------    ------------    ------------   -------\n");
    }
    else
    {
      fprintf(log_file, "Iters     ||K r||_C   ||K r||_C/||b||_2    Condition    Restart\n");
      fprintf(log_file, "-----    ------------    ------------    ------------   -------\n");
    }

    for (j = 0; j < i; j++)
    {
      fprintf(log_file, "% 5d    %e    %e    %e    %d\n",
              (j + 1), norm_log[j], rel_norm_log[j], cond_log[j],
              restart_log[j]);
    }

    CloseLogFile(log_file);

    tfree(norm_log);
    tfree(rel_norm_log);
    tfree(cond_log);
    tfree(restart_log);
  }

  tfree(alpha_vec);
  tfree(beta_vec);
  tfree(pdotAp_vec);
}


/*--------------------------------------------------------------------------
 * PPCGInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *PPCGInitInstanceXtra(
                                Problem *    problem,
                                Grid *       grid,
                                ProblemData *problem_data,
                                Matrix *     A,
                                double *     temp_data)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;


  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
  {
    /* set new data */
    (instance_xtra->grid) = grid;
  }

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `A'
   *-----------------------------------------------------------------------*/

  if (A != NULL)
    (instance_xtra->A) = A;

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `temp_data'
   *-----------------------------------------------------------------------*/

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;
  }

  /*-----------------------------------------------------------------------
   * Initialize module instances
   *-----------------------------------------------------------------------*/

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->precond) =
      PFModuleNewInstanceType(ChebyshevInitInstanceXtraInvoke,
                              (public_xtra->precond),
                              (problem, grid, problem_data, A, temp_data));
  }
  else
  {
    PFModuleReNewInstanceType(ChebyshevInitInstanceXtraInvoke,
                              (instance_xtra->precond),
                              (problem, grid, problem_data, A, temp_data));
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * PPCGFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void   PPCGFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    PFModuleFreeInstance(instance_xtra->precond);

    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * PPCGNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *PPCGNewPublicXtra(char *name)

{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  int two_norm;

  char          *switch_name;
  int switch_value;
  char key[IDB_MAX_KEY_LEN];
  NameArray switch_na;

  NameArray precond_na;

  switch_na = NA_NewNameArray("True False");

  public_xtra = ctalloc(PublicXtra, 1);

  precond_na = NA_NewNameArray("Chebyshev");
  sprintf(key, "%s.PolyPC", name);
  switch_name = GetStringDefault(key, "Chebyshev");
  switch_value = NA_NameToIndexExitOnError(precond_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      (public_xtra->precond) = PFModuleNewModuleType(ChebyshevNewPublicXtraInvoke,
                                                     Chebyshev, (key));
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(precond_na);

  sprintf(key, "%s.MaxIter", name);
  public_xtra->max_iter = GetIntDefault(key, 1000);

  sprintf(key, "%s.PolyDegree", name);
  public_xtra->degree = GetIntDefault(key, 3);

  sprintf(key, "%s.TwoNorm", name);
  switch_name = GetStringDefault(key, "False");

  two_norm = NA_NameToIndexExitOnError(switch_na, switch_name, key);

  switch (two_norm)
  {
    /* True */
    case 0:
    {
      public_xtra->two_norm = 1;
      break;
    }

    /* False */
    case 1:
    {
      public_xtra->two_norm = 0;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
      break;
    }
  }

  (public_xtra->time_index) = RegisterTiming("PPCG");

  PFModulePublicXtra(this_module) = public_xtra;

  NA_FreeNameArray(switch_na);
  return this_module;
}


/*--------------------------------------------------------------------------
 * PPCGFreePublicXtra
 *--------------------------------------------------------------------------*/

void   PPCGFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (public_xtra)
  {
    PFModuleFreeModule(public_xtra->precond);
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * PPCGSizeOfTempData
 *--------------------------------------------------------------------------*/

int  PPCGSizeOfTempData()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int sz = 0;


  /* set `sz' to max of each of the called modules */
  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->precond));

  return sz;
}
