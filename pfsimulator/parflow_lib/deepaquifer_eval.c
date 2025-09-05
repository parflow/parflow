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
void DeepAquiferEval(int fcn)
{
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