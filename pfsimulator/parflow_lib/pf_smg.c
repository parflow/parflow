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

#ifdef HAVE_HYPRE
#include "pf_hypre.h"
#include "hypre_dependences.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int max_iter;
  int num_pre_relax;
  int num_post_relax;

  int time_index_smg;
  int time_index_copy_hypre;
} PublicXtra;

typedef struct {
  HYPRE_StructGrid hypre_grid;
  HYPRE_StructMatrix hypre_mat;
  HYPRE_StructVector hypre_b, hypre_x;
  HYPRE_StructStencil hypre_stencil;

  HYPRE_StructSolver hypre_smg_data;
} InstanceXtra;

#endif

/*--------------------------------------------------------------------------
 * SMG
 *--------------------------------------------------------------------------*/

void         SMG(
                 Vector *soln,
                 Vector *rhs,
                 double  tol,
                 int     zero)
{
#ifdef HAVE_HYPRE
  PFModule           *this_module = ThisPFModule;
  InstanceXtra       *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra         *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  HYPRE_StructMatrix hypre_mat = instance_xtra->hypre_mat;
  HYPRE_StructVector hypre_b = instance_xtra->hypre_b;
  HYPRE_StructVector hypre_x = instance_xtra->hypre_x;

  HYPRE_StructSolver hypre_smg_data = instance_xtra->hypre_smg_data;

  int num_iterations;
  double rel_norm;

  (void)zero;

  /* Copy rhs to hypre_b vector. */
  BeginTiming(public_xtra->time_index_copy_hypre);

  CopyParFlowVectorToHypreVector(rhs, &hypre_b);

  EndTiming(public_xtra->time_index_copy_hypre);

  if (tol > 0.0)
  {
    IfLogging(1)
    {
      HYPRE_StructSMGSetLogging(instance_xtra->hypre_smg_data, 1);
    }
  }

  /* Invoke the preconditioner using a zero initial guess */
  HYPRE_StructSMGSetZeroGuess(hypre_smg_data);
  HYPRE_StructSMGSetTol(hypre_smg_data, tol);

  BeginTiming(public_xtra->time_index_smg);

  HYPRE_StructSMGSolve(hypre_smg_data, hypre_mat, hypre_b, hypre_x);

  EndTiming(public_xtra->time_index_smg);

  if (tol > 0.0)
  {
    IfLogging(1)
    {
      FILE  *log_file;

      HYPRE_StructSMGGetNumIterations(hypre_smg_data, &num_iterations);
      HYPRE_StructSMGGetFinalRelativeResidualNorm(hypre_smg_data,
                                                  &rel_norm);

      log_file = OpenLogFile("SMG");
      fprintf(log_file, "SMG num. its: %i  SMG Final norm: %12.4e\n",
              num_iterations, rel_norm);
      CloseLogFile(log_file);
    }
  }

  /* Copy solution from hypre_x vector to the soln vector. */
  BeginTiming(public_xtra->time_index_copy_hypre);

  CopyHypreVectorToParflowVector(&hypre_x, soln);

  EndTiming(public_xtra->time_index_copy_hypre);
#endif
}

/*--------------------------------------------------------------------------
 * SMGInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *SMGInitInstanceXtra(
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

  double dummy[3];

  (void)problem;
  (void)problem_data;
  (void)temp_data;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  HypreAssembleGrid(grid, &(instance_xtra->hypre_grid), dummy);

  /* Reset the HYPRE solver for each recompute of the PC matrix.
   * This reset will require a matrix copy from PF format to HYPRE format. */
  if (pf_Bmat != NULL)
  {
    /* Free old solver data because HYPRE requires a new solver if
     * matrix values change */
    if (instance_xtra->hypre_smg_data)
    {
      HYPRE_StructSMGDestroy(instance_xtra->hypre_smg_data);
      instance_xtra->hypre_smg_data = NULL;
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

    /* Set up the SMG preconditioner */
    HYPRE_StructSMGCreate(amps_CommWorld,
                          &(instance_xtra->hypre_smg_data));

    /* Set SMG to recompute rather than save data */
    HYPRE_StructSMGSetMemoryUse(instance_xtra->hypre_smg_data, 0);

    HYPRE_StructSMGSetTol(instance_xtra->hypre_smg_data, 1.0e-40);
    /* Set user parameters for SMG */
    HYPRE_StructSMGSetMaxIter(instance_xtra->hypre_smg_data, max_iter);
    HYPRE_StructSMGSetNumPreRelax(instance_xtra->hypre_smg_data,
                                  num_pre_relax);
    HYPRE_StructSMGSetNumPostRelax(instance_xtra->hypre_smg_data,
                                   num_post_relax);

    HYPRE_StructSMGSetup(instance_xtra->hypre_smg_data,
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
 * SMGFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  SMGFreeInstanceXtra()
{
#ifdef HAVE_HYPRE
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    if (instance_xtra->hypre_smg_data)
      HYPRE_StructSMGDestroy(instance_xtra->hypre_smg_data);
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
 * SMGNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *SMGNewPublicXtra(char *name)
{
#ifdef HAVE_HYPRE
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  char key[IDB_MAX_KEY_LEN];

  public_xtra = ctalloc(PublicXtra, 1);

  sprintf(key, "%s.MaxIter", name);
  public_xtra->max_iter = GetIntDefault(key, 1);

  sprintf(key, "%s.NumPreRelax", name);
  public_xtra->num_pre_relax = GetIntDefault(key, 1);

  sprintf(key, "%s.NumPostRelax", name);
  public_xtra->num_post_relax = GetIntDefault(key, 0);

  public_xtra->time_index_smg = RegisterTiming("SMG");
  public_xtra->time_index_copy_hypre = RegisterTiming("HYPRE_Copies");

  PFModulePublicXtra(this_module) = public_xtra;

  return this_module;
#else
  return NULL;
#endif
}

/*-------------------------------------------------------------------------
 * SMGFreePublicXtra
 *-------------------------------------------------------------------------*/

void  SMGFreePublicXtra()
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
 * SMGSizeOfTempData
 *--------------------------------------------------------------------------*/

int  SMGSizeOfTempData()
{
  return 0;
}

