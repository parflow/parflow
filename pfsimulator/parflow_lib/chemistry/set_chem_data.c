/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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
* Module for initializing the problem structure.
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "parflow.h"
#include "pf_alquimia.h"

#ifdef HAVE_ALQUIMIA


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void PublicXtra;

typedef struct {
  PFModule  *geochemcond;
  PFModule  *bc_concentration;
  Problem   *problem;
  Grid      *grid;
  double    *temp_data;
} InstanceXtra;


/** @brief Populate the chemistry-related problem-data fields (e.g. the
 * geochemical-condition index field) by invoking their sub-modules. The
 * SetChemData PFModule analogue of SetProblemData for chemistry.
 *
 * @param problem_data the problem data to populate
 */
#ifdef HAVE_ALQUIMIA
void          SetChemData(ProblemData *problem_data)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PFModule      *geochemcond = (instance_xtra->geochemcond);

  PFModuleInvokeType(GeochemCondInvoke, geochemcond,            //JJB
                     (problem_data,
                      ProblemDataGeochemCond(problem_data)));
}


/*--------------------------------------------------------------------------
 * SetChemDataInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *SetChemDataInitInstanceXtra(Problem *problem,
                                       Grid *   grid)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with `problem'
   *-----------------------------------------------------------------------*/

  if (problem != NULL)
    (instance_xtra->problem) = problem;

  /*-----------------------------------------------------------------------
   * Initialize data associated with `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
    (instance_xtra->grid) = grid;

  /*-----------------------------------------------------------------------
   * Initialize module instances
   *-----------------------------------------------------------------------*/

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->geochemcond) =
      PFModuleNewInstance(ProblemGeochemCond(problem), ());
  }
  else
  {
    PFModuleReNewInstance((instance_xtra->geochemcond), ());
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;






  return this_module;
}


/*--------------------------------------------------------------------------
 * SetChemDataFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  SetChemDataFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    PFModuleFreeInstance(instance_xtra->geochemcond);
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * SetChemDataNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *SetChemDataNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;


#if 0
  public_xtra = ctalloc(PublicXtra, 1);
#endif
  public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * SetChemDataFreePublicXtra
 *--------------------------------------------------------------------------*/

void  SetChemDataFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (public_xtra)
  {
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * SetChemDataSizeOfTempData
 *--------------------------------------------------------------------------*/

int       SetChemDataSizeOfTempData()
{
  return 0;
}
#endif

#endif /* HAVE_ALQUIMIA */
