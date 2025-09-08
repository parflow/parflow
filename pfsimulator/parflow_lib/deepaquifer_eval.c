/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2025, Lawrence Livermore National Security,
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

typedef void PublicXtra;

typedef void InstanceXtra;


/*--------------------------------------------------------------------------
 * DeepAquiferEval
 *--------------------------------------------------------------------------*/
/**
 * @name DeepAquifer BC Module
 * @brief Evaluation of the DeepAquifer boundary
 *
 * This module can evaluate both the non-linear function
 * and its jacobian. When invoked, the `fcn` flag tracks
 * which output to compute.
 *
 * @param fcn flag = {CALCFCN , CALCDER}
 */
void DeepAquiferEval(ProblemData *problem_data,
                     int          isubgrid,
                     int          fcn)
{
  Vector *permeability = ProblemDataDeepAquiferPermeability(problem_data);
  Vector *specific_yield = ProblemDataDeepAquiferSpecificYield(problem_data);
  Vector *aquifer_depth = ProblemDataDeepAquiferAquiferDepth(problem_data);
  Vector *elevation = ProblemDataDeepAquiferElevation(problem_data);

  Subvector *K_sub = VectorSubvector(permeability, isubgrid);
  Subvector *Sy_sub = VectorSubvector(specific_yield, isubgrid);
  Subvector *Ad_sub = VectorSubvector(aquifer_depth, isubgrid);
  Subvector *El_sub = VectorSubvector(elevation, isubgrid);

  double *K = SubvectorData(K_sub);
  double *Sy = SubvectorData(Sy_sub);
  double *Ad = SubvectorData(Ad_sub);
  double *El = SubvectorData(El_sub);

  return;
}

/*--------------------------------------------------------------------------
 * DeepAquiferEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule* DeepAquiferEvalInitInstanceXtra()
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = ctalloc(InstanceXtra, 1);

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * DeepAquiferEvalFreeInstanceXtra
 *--------------------------------------------------------------------------*/
void DeepAquiferEvalFreeInstanceXtra()
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra =
    (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * DeepAquiferEvalNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule* DeepAquiferEvalNewPublicXtra()
{
  PFModule   *this_module = ThisPFModule;
  PublicXtra *public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * DeepAquiferEvalFreePublicXtra
 *-------------------------------------------------------------------------*/

void DeepAquiferEvalFreePublicXtra()
{
  PFModule   *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * DeepAquiferEvalSizeOfTempData
 *--------------------------------------------------------------------------*/

int DeepAquiferEvalSizeOfTempData()
{
  return 0;
}

/*--------------------------------------------------------------------------
 * SetDeepAquiferPermeability
 *--------------------------------------------------------------------------*/

void SetDeepAquiferPermeability(ProblemData *problem_data)
{
  Vector *permeability = ProblemDataDeepAquiferPermeability(problem_data);
  NameArray switch_na = NA_NewNameArray("Constant PFBFile SameAsBottomLayer");
  char key[IDB_MAX_KEY_LEN];

  sprintf(key, "Patch.BCPressure.DeepAquifer.Permeability.Type");

  char *switch_name = GetString(key);
  int switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0: // Constant
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.Permeability.Value");
      double value = GetDouble(key);
      InitVectorAll(permeability, value);
      break;
    }

    case 1: // PFBFile
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.Permeability.FileName");
      char *filename = GetString(key);
      ReadPFBinary(filename, permeability);

      VectorUpdateCommHandle *handle = InitVectorUpdate(permeability, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      break;
    }

    case 2: // SameAsBottomLayer
    {
      // Not implemented yet
      InputError("Swicth value <%s> not yet implemented for key <%s>", switch_name, key);
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(switch_na);

  return;
}

/*--------------------------------------------------------------------------
 * SetDeepAquiferSpecificYield
 *--------------------------------------------------------------------------*/

void SetDeepAquiferSpecificYield(ProblemData *problem_data)
{
  Vector *specific_yield = ProblemDataDeepAquiferSpecificYield(problem_data);
  NameArray switch_na = NA_NewNameArray("Constant PFBFile");
  char key[IDB_MAX_KEY_LEN];

  sprintf(key, "Patch.BCPressure.DeepAquifer.SpecificYield.Type");

  char *switch_name = GetString(key);
  int switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0: // Constant
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.SpecificYield.Value");
      double value = GetDouble(key);
      InitVectorAll(specific_yield, value);
      break;
    }

    case 1: // PFBFile
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.SpecificYield.FileName");
      char *filename = GetString(key);
      ReadPFBinary(filename, specific_yield);

      VectorUpdateCommHandle *handle = InitVectorUpdate(specific_yield, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(switch_na);

  return;
}

/*--------------------------------------------------------------------------
 * SetDeepAquiferAquiferDepth
 *--------------------------------------------------------------------------*/

void SetDeepAquiferAquiferDepth(ProblemData *problem_data)
{
  Vector *aquifer_depth = ProblemDataDeepAquiferAquiferDepth(problem_data);
  NameArray switch_na = NA_NewNameArray("Constant");
  char key[IDB_MAX_KEY_LEN];

  sprintf(key, "Patch.BCPressure.DeepAquifer.AquiferDepth.Type");

  char *switch_name = GetString(key);
  int switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0: // Constant
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.AquiferDepth.Value");
      double value = GetDouble(key);
      InitVectorAll(aquifer_depth, value);
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(switch_na);

  return;
}

/*--------------------------------------------------------------------------
 * SetDeepAquiferElevation
 *--------------------------------------------------------------------------*/

void SetDeepAquiferElevation(ProblemData *problem_data)
{
  Vector *elevation = ProblemDataDeepAquiferElevation(problem_data);
  NameArray switch_na = NA_NewNameArray("Constant PFBFile");
  char key[IDB_MAX_KEY_LEN];

  sprintf(key, "Patch.BCPressure.DeepAquifer.Elevation.Type");

  char *switch_name = GetString(key);
  int switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0: // Constant
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.Elevation.Value");
      double value = GetDouble(key);
      InitVectorAll(elevation, value);
      break;
    }

    case 1: // PFBFile
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.Elevation.FileName");
      char *filename = GetString(key);
      ReadPFBinary(filename, elevation);

      VectorUpdateCommHandle *handle = InitVectorUpdate(elevation, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(switch_na);

  return;
}