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
  int pc_matrix_type;

  PFModule  *precond;
  PFModule  *discretization;
} PublicXtra;

typedef struct {
  PFModule  *precond;
  PFModule  *discretization;

  Matrix    *PC, *JC;

  Grid      *grid;

  double    *temp_data;
} InstanceXtra;

/*--------------------------------------------------------------------------
 * KinsolPC
 *--------------------------------------------------------------------------*/

void         KinsolPC(Vector *rhs)
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  PFModule     *precond = instance_xtra->precond;
  Vector       *soln = NULL;
  double tol = 0.0;
  int zero = 1;

  /* Allocate temp vector */
  soln = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);

  /* Invoke the preconditioner using a zero initial guess */
  PFModuleInvokeType(PrecondInvoke, precond, (soln, rhs, tol, zero));

  /* Copy solution from soln to the rhs vector. */
  PFVCopy(soln, rhs);

  /* Free temp vector */
  FreeVector(soln);
}

/*--------------------------------------------------------------------------
 * KinsolPCInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *KinsolPCInitInstanceXtra(
                                    Problem *    problem,
                                    Grid *       grid,
                                    Grid *       grid2d,
                                    ProblemData *problem_data,
                                    double *     temp_data,
                                    Vector *     pressure,
                                    Vector *     old_pressure,
                                    Vector *     saturation,
                                    Vector *     density,
                                    double       dt,
                                    double       time)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;

  int pc_matrix_type = public_xtra->pc_matrix_type;

  PFModule      *discretization = public_xtra->discretization;
  PFModule      *precond = public_xtra->precond;

  Matrix        *PC, *JC;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (grid != NULL)
  {
    /* set new data */
    instance_xtra->grid = grid;
  }

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;
  }

  /*-----------------------------------------------------------------------
   * Initialize module instances
   *-----------------------------------------------------------------------*/

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->discretization) =
      PFModuleNewInstanceType(
                              RichardsJacobianEvalInitInstanceXtraInvoke,
                              discretization, (problem, grid, grid2d, problem_data, temp_data,
                                               pc_matrix_type));
    (instance_xtra->precond) =
      PFModuleNewInstanceType(PrecondInitInstanceXtraInvoke,
                              precond, (problem, grid, problem_data, NULL, NULL,
                                        temp_data));
/*
 *    (instance_xtra -> precond) =
 *       PFModuleNewInstanceType(PrecondInitInstanceXtraInvoke,
 *                               precond,(problem, grid, problem_data, NULL,NULL,
 *                                        temp_data));
 */
  }
  else if (pressure != NULL)
  {
    PFModuleInvokeType(RichardsJacobianEvalInvoke, (instance_xtra->discretization),
                       (pressure, old_pressure, &PC, &JC, saturation, density, problem_data, dt,
                        time, pc_matrix_type));
    PFModuleReNewInstanceType(PrecondInitInstanceXtraInvoke,
                              (instance_xtra->precond),
                              (NULL, NULL, problem_data, PC, JC, temp_data));
/*
 *    PFModuleReNewInstanceType(PrecondInitInstanceXtraInvoke,
 *                              (instance_xtra -> precond),
 *                              (NULL, NULL, problem_data, PC,JC, temp_data));
 */
  }
  else
  {
    PFModuleReNewInstanceType(RichardsJacobianEvalInitInstanceXtraInvoke,
                              (instance_xtra->discretization),
                              (problem, grid, grid2d, problem_data, temp_data, pc_matrix_type));
    PFModuleReNewInstanceType(PrecondInitInstanceXtraInvoke,
                              (instance_xtra->precond),
                              (NULL, NULL, problem_data, NULL, NULL, temp_data));
/*
 *    PFModuleReNewInstanceType(PrecondInitInstanceXtraInvoke,
 *                          (instance_xtra -> precond),
 *                          (NULL, NULL, problem_data, NULL, NULL, temp_data));
 */
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * KinsolPCFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  KinsolPCFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    PFModuleFreeInstance((instance_xtra->precond));
    PFModuleFreeInstance((instance_xtra->discretization));

    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * KinsolPCNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *KinsolPCNewPublicXtra(char *name, char *pc_name)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  char          *switch_name;
  int switch_value;
  char key[IDB_MAX_KEY_LEN];
  NameArray precond_na, precond_switch_na;

  public_xtra = ctalloc(PublicXtra, 1);

  precond_na = NA_NewNameArray("FullJacobian PFSymmetric SymmetricPart Picard");
  sprintf(key, "%s.PCMatrixType", name);
  switch_name = GetStringDefault(key, "PFSymmetric");
  switch_value = NA_NameToIndexExitOnError(precond_na, switch_name, key);
  switch (switch_value)
  {
    case 0:
    {
      public_xtra->pc_matrix_type = 0;
      break;
    }

    case 1:
    {
      public_xtra->pc_matrix_type = 1;
      break;
    }

    case 2:
    {
      public_xtra->pc_matrix_type = 2;
      break;
    }

    case 3:
    {
      public_xtra->pc_matrix_type = 3;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(precond_na);

  precond_switch_na = NA_NewNameArray("NoPC MGSemi SMG PFMG PFMGOctree");
  sprintf(key, "%s.%s", name, pc_name);
  switch_value = NA_NameToIndexExitOnError(precond_switch_na, pc_name, key);
  switch (switch_value)
  {
    case 0:
    {
      public_xtra->precond = NULL;
      break;
    }

    case 1:
    {
      public_xtra->precond = PFModuleNewModuleType(
                                                   LinearSolverNewPublicXtraInvoke,
                                                   MGSemi, (key));
      break;
    }

    case 2:
    {
#ifdef HAVE_HYPRE
      public_xtra->precond = PFModuleNewModuleType(
                                                   LinearSolverNewPublicXtraInvoke, SMG, (key));
#else
      InputError("Error: Invalid value <%s> for key <%s>.\n"
                 "SMG code not compiled in.\n", switch_name, key);
#endif
      break;
    }

    case 3:
    {
#ifdef HAVE_HYPRE
      public_xtra->precond = PFModuleNewModuleType(
                                                   LinearSolverNewPublicXtraInvoke, PFMG, (key));
#else
      InputError("Error: Invalid value <%s> for key <%s>.\n"
                 "Hypre PFMG code not compiled in.\n", switch_name, key);
#endif
      break;
    }

    case 4:
    {
#ifdef HAVE_HYPRE
      public_xtra->precond = PFModuleNewModuleType(
                                                   LinearSolverNewPublicXtraInvoke, PFMGOctree, (key));
#else
      InputError("Error: Invalid value <%s> for key <%s>.\n"
                 "Hypre PFMG code not compiled in.\n", switch_name, key);
#endif
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(precond_switch_na);

  sprintf(key, "%s.Jacobian", name);
  public_xtra->discretization = PFModuleNewModuleType(
                                                      RichardsJacobianEvalNewPublicXtraInvoke, RichardsJacobianEval,
                                                      (key));

  PFModulePublicXtra(this_module) = public_xtra;

  return this_module;
}

/*-------------------------------------------------------------------------
 * KinsolPCFreePublicXtra
 *-------------------------------------------------------------------------*/

void  KinsolPCFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    PFModuleFreeModule(public_xtra->precond);
    PFModuleFreeModule(public_xtra->discretization);

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * KinsolPCSizeOfTempData
 *--------------------------------------------------------------------------*/

int  KinsolPCSizeOfTempData()
{
  PFModule             *this_module = ThisPFModule;
  InstanceXtra         *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PFModule             *precond = (instance_xtra->precond);

  int sz = 0;

  sz += PFModuleSizeOfTempData(precond);

  return sz;
}
