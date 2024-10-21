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
* Preconditioned conjugate gradient solver (Omin).
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  PFModule  *precond;

  int max_iter;
  int two_norm;

  int time_index;
} PublicXtra;

typedef struct {
  PFModule  *precond;

  /* InitInstanceXtra arguments */
  Grid     *grid;
  Matrix   *A;
  double    *temp_data;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * PCG
 *--------------------------------------------------------------------------
 *
 * We use the following convergence test as the default (see Ashby, Holst,
 * Manteuffel, and Saylor):
 *
 *       ||e||_A                           ||r||_C
 *       -------  <=  [kappa_A(C*A)]^(1/2) -------  < tol
 *       ||x||_A                           ||b||_C
 *
 * where we let (for the time being) kappa_A(CA) = 1.
 * We implement the test as:
 *
 *       gamma = <C*r,r>  <  (tol^2)*<C*b,b> = eps
 *
 *--------------------------------------------------------------------------*/

void     PCG(
             Vector *x,
             Vector *b,
             double  tol,
             int     zero)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int max_iter = (public_xtra->max_iter);
  int two_norm = (public_xtra->two_norm);

  PFModule  *precond = (instance_xtra->precond);

  Matrix    *A = (instance_xtra->A);

  Vector    *r;
  Vector    *p = NULL;
  Vector    *s = NULL;

  double alpha, beta;
  double gamma, gamma_old;
  double bi_prod, i_prod = 0.0, eps;

  int i = 0;

  double    *norm_log = NULL;
  double    *rel_norm_log = NULL;


  /*-----------------------------------------------------------------------
   * Initialize some logging variables
   *-----------------------------------------------------------------------*/

  IfLogging(1)
  {
    norm_log = talloc(double, max_iter);
    rel_norm_log = talloc(double, max_iter);
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
   * Start pcg solve
   *-----------------------------------------------------------------------*/

  if (zero)
    InitVector(x, 0.0);

  if (two_norm)
  {
    /* eps = (tol^2)*<b,b> */
    bi_prod = InnerProd(b, b);
    eps = (tol * tol) * bi_prod;
  }
  else
  {
    /* eps = (tol^2)*<C*b,b> */
    PFModuleInvokeType(PrecondInvoke, precond, (p, b, 0.0, 1));
    bi_prod = InnerProd(p, b);
    eps = (tol * tol) * bi_prod;
  }

  /* r = b - Ax,  (overwrite b with r) */
  Matvec(-1.0, A, x, 1.0, (r = b));

  /* p = C*r */
  PFModuleInvokeType(PrecondInvoke, precond, (p, r, 0.0, 1));

  /* gamma = <r,p> */
  gamma = InnerProd(r, p);

  while (((i + 1) <= max_iter) && (gamma > 0))
  {
    i++;

    /* s = A*p */
    Matvec(1.0, A, p, 0.0, s);

    /* alpha = gamma / <s,p> */
    alpha = gamma / InnerProd(s, p);

    gamma_old = gamma;

    /* x = x + alpha*p */
    Axpy(alpha, p, x);

    /* r = r - alpha*s */
    Axpy(-alpha, s, r);

    /* s = C*r */
    PFModuleInvokeType(PrecondInvoke, precond, (s, r, 0.0, 1));

    /* gamma = <r,s> */
    gamma = InnerProd(r, s);

    /* set i_prod for convergence test */
    if (two_norm)
      i_prod = InnerProd(r, r);
    else
      i_prod = gamma;

#if 1
    if (!amps_Rank(amps_CommWorld))
    {
      if (two_norm)
        amps_Printf("Iter (%d): ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
                    i, sqrt(i_prod), (bi_prod ? sqrt(i_prod / bi_prod) : 0));
      else
        amps_Printf("Iter (%d): ||r||_C = %e, ||r||_C/||b||_C = %e\n",
                    i, sqrt(i_prod), (bi_prod ? sqrt(i_prod / bi_prod) : 0));

      fflush(NULL);
    }
#endif

    /* log norm info */
    IfLogging(1)
    {
      norm_log[i - 1] = sqrt(i_prod);
      rel_norm_log[i - 1] = bi_prod ? sqrt(i_prod / bi_prod) : 0;
    }

    /* check for convergence */
    if (i_prod < eps)
      break;

    /* beta = gamma / gamma_old */
    beta = gamma / gamma_old;

    /* p = s + beta p */
    Scale(beta, p);
    Axpy(1.0, s, p);
  }

#if 1
  if (!amps_Rank(amps_CommWorld))
  {
    if (two_norm)
      amps_Printf("Iterations = %d: ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
                  i, sqrt(i_prod), (bi_prod ? sqrt(i_prod / bi_prod) : 0));
    else
      amps_Printf("Iterations = %d: ||r||_C = %e, ||r||_C/||b||_C = %e\n",
                  i, sqrt(i_prod), (bi_prod ? sqrt(i_prod / bi_prod) : 0));
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
   * Print log
   *-----------------------------------------------------------------------*/

  IfLogging(1)
  {
    FILE *log_file;
    int j;

    log_file = OpenLogFile("PCG");

    if (two_norm)
    {
      fprintf(log_file, "Iters       ||r||_2    ||r||_2/||b||_2\n");
      fprintf(log_file, "-----    ------------    ------------\n");
    }
    else
    {
      fprintf(log_file, "Iters       ||r||_C    ||r||_C/||b||_C\n");
      fprintf(log_file, "-----    ------------    ------------\n");
    }

    for (j = 0; j < i; j++)
    {
      fprintf(log_file, "% 5d    %e    %e\n",
              (j + 1), norm_log[j], rel_norm_log[j]);
    }

    CloseLogFile(log_file);

    tfree(norm_log);
    tfree(rel_norm_log);
  }
}


/*--------------------------------------------------------------------------
 * PCGInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *PCGInitInstanceXtra(
                               Problem *    problem,
                               Grid *       grid,
                               ProblemData *problem_data,
                               Matrix *     A,
                               Matrix *     C,
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
    /* free old data */
    if ((instance_xtra->grid) != NULL)
    {
    }

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
      PFModuleNewInstanceType(PrecondInitInstanceXtraInvoke,
                              (public_xtra->precond),
                              (problem, grid, problem_data, A, C, temp_data));
  }
  else
  {
    PFModuleReNewInstanceType(PrecondInitInstanceXtraInvoke,
                              (instance_xtra->precond),
                              (problem, grid, problem_data, A, C, temp_data));
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * PCGFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void   PCGFreeInstanceXtra()
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
 * PCGNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *PCGNewPublicXtra(char *name)
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

  precond_na = NA_NewNameArray("MGSemi WJacobi");
  sprintf(key, "%s.Preconditioner", name);
  switch_name = GetStringDefault(key, "MGSemi");
  switch_value = NA_NameToIndexExitOnError(precond_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      public_xtra->precond = PFModuleNewModuleType(PrecondNewPublicXtra, MGSemi, (key));
      break;
    }

    case 1:
    {
      public_xtra->precond = PFModuleNewModuleType(PrecondNewPublicXtra, WJacobi, (key));
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
    }
  }

  (public_xtra->time_index) = RegisterTiming("PCG");

  PFModulePublicXtra(this_module) = public_xtra;

  NA_FreeNameArray(switch_na);
  return this_module;
}


/*--------------------------------------------------------------------------
 * PCGFreePublicXtra
 *--------------------------------------------------------------------------*/

void  PCGFreePublicXtra()
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
 * PCGSizeOfTempData
 *--------------------------------------------------------------------------*/

int  PCGSizeOfTempData()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int sz = 0;


  /* set `sz' to max of each of the called modules */
  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->precond));


  return sz;
}
