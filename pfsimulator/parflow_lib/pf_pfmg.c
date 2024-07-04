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

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

#ifdef HAVE_HYPRE
#include "pf_hypre.h"
#include "hypre_dependences.h"

typedef struct {
  int max_iter;
  int num_pre_relax;
  int num_post_relax;
  int smoother;
  int raptype;

  int time_index_pfmg;
  int time_index_copy_hypre;
} PublicXtra;

typedef struct {
  double dxyz[3];

  HYPRE_StructGrid hypre_grid;
  HYPRE_StructMatrix hypre_mat;
  HYPRE_StructVector hypre_b, hypre_x;
  HYPRE_StructStencil hypre_stencil;

  HYPRE_StructSolver hypre_pfmg_data;
} InstanceXtra;

#endif

/*--------------------------------------------------------------------------
 * PFMG
 *--------------------------------------------------------------------------*/

void         PFMG(
                  Vector *soln,
                  Vector *rhs,
                  double  tol,
                  int     zero)
{
  (void)zero;

#ifdef HAVE_HYPRE
  PFModule           *this_module = ThisPFModule;
  InstanceXtra       *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra         *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  HYPRE_StructMatrix hypre_mat = instance_xtra->hypre_mat;
  HYPRE_StructVector hypre_b = instance_xtra->hypre_b;
  HYPRE_StructVector hypre_x = instance_xtra->hypre_x;

  HYPRE_StructSolver hypre_pfmg_data = instance_xtra->hypre_pfmg_data;

  int num_iterations;
  double rel_norm;

  /* Copy rhs to hypre_b vector. */
  BeginTiming(public_xtra->time_index_copy_hypre);

  CopyParFlowVectorToHypreVector(rhs, &hypre_b);

  EndTiming(public_xtra->time_index_copy_hypre);

  if (tol > 0.0)
  {
    IfLogging(1)
    {
      HYPRE_StructPFMGSetLogging(instance_xtra->hypre_pfmg_data, 1);
    }
  }

  /* Invoke the preconditioner using a zero initial guess */
  HYPRE_StructPFMGSetZeroGuess(hypre_pfmg_data);

  BeginTiming(public_xtra->time_index_pfmg);

  HYPRE_StructPFMGSolve(hypre_pfmg_data, hypre_mat, hypre_b, hypre_x);

  EndTiming(public_xtra->time_index_pfmg);

  if (tol > 0.0)
  {
    IfLogging(1)
    {
      FILE  *log_file;

      HYPRE_StructPFMGGetNumIterations(hypre_pfmg_data, &num_iterations);
      HYPRE_StructPFMGGetFinalRelativeResidualNorm(hypre_pfmg_data,
                                                   &rel_norm);

      log_file = OpenLogFile("PFMG");
      fprintf(log_file, "PFMG num. its: %i  PFMG Final norm: %12.4e\n",
              num_iterations, rel_norm);
      CloseLogFile(log_file);
    }
  }

  /* Copy solution from hypre_x vector to the soln vector. */
  BeginTiming(public_xtra->time_index_copy_hypre);

  CopyHypreVectorToParflowVector(&hypre_x, soln);

  EndTiming(public_xtra->time_index_copy_hypre);
#else
  amps_Printf("Error: Parflow not compiled with hypre, can't use pfmg\n");
#endif
}

/*--------------------------------------------------------------------------
 * PFMGInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *PFMGInitInstanceXtra(
                                Problem *    problem,
                                Grid *       grid,
                                ProblemData *problem_data,
                                Matrix *     pf_Bmat,
                                Matrix *     pf_Cmat,
                                double *     temp_data)
{
#ifdef HAVE_HYPRE
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;

  int max_iter = public_xtra->max_iter;
  int num_pre_relax = public_xtra->num_pre_relax;
  int num_post_relax = public_xtra->num_post_relax;
  int smoother = public_xtra->smoother;
  int raptype = public_xtra->raptype;

  (void)problem;
  (void)problem_data;
  (void)temp_data;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  HypreAssembleGrid(grid, &(instance_xtra->hypre_grid), instance_xtra->dxyz);

  /* Reset the HYPRE solver for each recompute of the PC matrix.
   * This reset will require a matrix copy from PF format to HYPRE format. */
  if (pf_Bmat != NULL)
  {
    /* Free old solver data because HYPRE requires a new solver if
     * matrix values change */
    if (instance_xtra->hypre_pfmg_data)
    {
      HYPRE_StructPFMGDestroy(instance_xtra->hypre_pfmg_data);
      instance_xtra->hypre_pfmg_data = NULL;
    }

    HypreInitialize(pf_Bmat,
		    &(instance_xtra -> hypre_grid),
		    &(instance_xtra -> hypre_stencil),
		    &(instance_xtra -> hypre_mat),
		    &(instance_xtra -> hypre_b),
		    &(instance_xtra -> hypre_x)
		    );

    /* Copy the matrix entries */
    BeginTiming(public_xtra->time_index_copy_hypre);

    HypreAssembleMatrixAsElements(pf_Bmat,
				  pf_Cmat,
				  &(instance_xtra -> hypre_mat),
				  problem_data);
    
    EndTiming(public_xtra->time_index_copy_hypre);

    /* Set up the PFMG preconditioner */
    HYPRE_StructPFMGCreate(amps_CommWorld,
                           &(instance_xtra->hypre_pfmg_data));

    HYPRE_StructPFMGSetTol(instance_xtra->hypre_pfmg_data, 1.0e-30);
    /* Set user parameters for PFMG */
    HYPRE_StructPFMGSetMaxIter(instance_xtra->hypre_pfmg_data, max_iter);
    HYPRE_StructPFMGSetNumPreRelax(instance_xtra->hypre_pfmg_data,
                                   num_pre_relax);
    HYPRE_StructPFMGSetNumPostRelax(instance_xtra->hypre_pfmg_data,
                                    num_post_relax);
    /* Jacobi = 0; weighted Jacobi = 1; red-black GS symmetric = 2; red-black GS non-symmetric = 3 */
    HYPRE_StructPFMGSetRelaxType(instance_xtra->hypre_pfmg_data, smoother);

    /* Galerkin=0; non-Galkerkin=1 */
    HYPRE_StructPFMGSetRAPType(instance_xtra->hypre_pfmg_data, raptype);

    HYPRE_StructPFMGSetSkipRelax(instance_xtra->hypre_pfmg_data, 1);

    HYPRE_StructPFMGSetDxyz(instance_xtra->hypre_pfmg_data,
                            instance_xtra->dxyz);

    HYPRE_StructPFMGSetup(instance_xtra->hypre_pfmg_data,
                          instance_xtra->hypre_mat,
                          instance_xtra->hypre_b, instance_xtra->hypre_x);
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
#else
  return NULL;
#endif
}


/*--------------------------------------------------------------------------
 * PFMGFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  PFMGFreeInstanceXtra()
{
#ifdef HAVE_HYPRE
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    if (instance_xtra->hypre_pfmg_data)
      HYPRE_StructPFMGDestroy(instance_xtra->hypre_pfmg_data);
    if (instance_xtra->hypre_mat)
      HYPRE_StructMatrixDestroy(instance_xtra->hypre_mat);
    if (instance_xtra->hypre_b)
      HYPRE_StructVectorDestroy(instance_xtra->hypre_b);
    if (instance_xtra->hypre_x)
      HYPRE_StructVectorDestroy(instance_xtra->hypre_x);
    if (instance_xtra->hypre_stencil)
      HYPRE_StructStencilDestroy(instance_xtra->hypre_stencil);
    if (instance_xtra->hypre_grid)
      HYPRE_StructGridDestroy(instance_xtra->hypre_grid);

    tfree(instance_xtra);
  }
#endif
}

/*--------------------------------------------------------------------------
 * PFMGNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *PFMGNewPublicXtra(char *name)
{
#ifdef HAVE_HYPRE
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;
  char key[IDB_MAX_KEY_LEN];
  char          *smoother_name;
  NameArray smoother_switch_na;
  char          *raptype_name;
  NameArray raptype_switch_na;

  public_xtra = ctalloc(PublicXtra, 1);

  sprintf(key, "%s.MaxIter", name);
  public_xtra->max_iter = GetIntDefault(key, 1);

  sprintf(key, "%s.NumPreRelax", name);
  public_xtra->num_pre_relax = GetIntDefault(key, 1);

  sprintf(key, "%s.NumPostRelax", name);
  public_xtra->num_post_relax = GetIntDefault(key, 1);

  /* Use a dummy place holder so that cardinalities match
   * with what HYPRE expects */
  smoother_switch_na = NA_NewNameArray("Jacobi WJacobi RBGaussSeidelSymmetric RBGaussSeidelNonSymmetric");
  sprintf(key, "%s.Smoother", name);
  smoother_name = GetStringDefault(key, "RBGaussSeidelNonSymmetric");
  public_xtra->smoother = NA_NameToIndexExitOnError(smoother_switch_na, smoother_name, key);
  NA_FreeNameArray(smoother_switch_na);

  raptype_switch_na = NA_NewNameArray("Galerkin NonGalerkin");
  sprintf(key, "%s.RAPType", name);
  raptype_name = GetStringDefault(key, "NonGalerkin");
  public_xtra->raptype = NA_NameToIndexExitOnError(raptype_switch_na, raptype_name, key);
  NA_FreeNameArray(raptype_switch_na);

  if (public_xtra->raptype == 0 && public_xtra->smoother  > 1)
  {
    InputError("Error: Galerkin RAPType is not compatible with Smoother <%s>.\n",
               smoother_name, key);
  }

  public_xtra->time_index_pfmg = RegisterTiming("PFMG");
  public_xtra->time_index_copy_hypre = RegisterTiming("HYPRE_Copies");

  PFModulePublicXtra(this_module) = public_xtra;

  return this_module;
#else
  amps_Printf("Error: Parflow not compiled with hypre, can't use pfmg\n");
  return NULL;
#endif
}

/*-------------------------------------------------------------------------
 * PFMGFreePublicXtra
 *-------------------------------------------------------------------------*/

void  PFMGFreePublicXtra()
{
#ifdef HAVE_HYPRE
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
#endif
}

/*--------------------------------------------------------------------------
 * PFMGSizeOfTempData
 *--------------------------------------------------------------------------*/

int  PFMGSizeOfTempData()
{
  return 0;
}

