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
* Module for initializing the problem structure.
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void PublicXtra;

typedef struct {
  PFModule  *geometries;
  PFModule  *domain;
  PFModule  *permeability;
  PFModule  *porosity;
  PFModule  *wells;
  PFModule  *bc_pressure;
  PFModule  *specific_storage;  //sk
  PFModule  *x_slope;  //sk
  PFModule  *y_slope;  //sk
  PFModule  *mann;  //sk
  PFModule  *dz_mult;   //RMM
  PFModule  *FBx;   //RMM
  PFModule  *FBy;   //RMM
  PFModule  *FBz;   //RMM

  PFModule  *real_space_z;
  int site_data_not_formed;

  /* InitInstanceXtra arguments */
  Problem   *problem;
  Grid      *grid;
  Grid      *grid2d;
  double    *temp_data;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * SetProblemData
 *--------------------------------------------------------------------------*/

void          SetProblemData(
                             ProblemData *problem_data)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  PFModule      *geometries = (instance_xtra->geometries);
  PFModule      *domain = (instance_xtra->domain);
  PFModule      *permeability = (instance_xtra->permeability);
  PFModule      *porosity = (instance_xtra->porosity);
  PFModule      *wells = (instance_xtra->wells);
  PFModule      *bc_pressure = (instance_xtra->bc_pressure);
  PFModule      *specific_storage = (instance_xtra->specific_storage);    //sk
  PFModule      *x_slope = (instance_xtra->x_slope);         //sk
  PFModule      *y_slope = (instance_xtra->y_slope);         //sk
  PFModule      *mann = (instance_xtra->mann);            //sk
  PFModule      *dz_mult = (instance_xtra->dz_mult);                //rmm
  PFModule      *FBx = (instance_xtra->FBx);                //rmm
  PFModule      *FBy = (instance_xtra->FBy);                //rmm
  PFModule      *FBz = (instance_xtra->FBz);                //rmm

  PFModule      *real_space_z = (instance_xtra->real_space_z);

  /* Note: the order in which these modules are called is important */
  PFModuleInvokeType(WellPackageInvoke, wells, (problem_data));
  if ((instance_xtra->site_data_not_formed))
  {
    PFModuleInvokeType(GeometriesInvoke, geometries, (problem_data));
    PFModuleInvokeType(DomainInvoke, domain, (problem_data));

    PFModuleInvokeType(SubsrfSimInvoke, permeability,
                       (problem_data,
                        ProblemDataPermeabilityX(problem_data),
                        ProblemDataPermeabilityY(problem_data),
                        ProblemDataPermeabilityZ(problem_data),
                        ProblemDataNumSolids(problem_data),
                        ProblemDataSolids(problem_data),
                        ProblemDataGrSolids(problem_data)));
    PFModuleInvokeType(PorosityInvoke, porosity,
                       (problem_data,
                        ProblemDataPorosity(problem_data),
                        ProblemDataNumSolids(problem_data),
                        ProblemDataSolids(problem_data),
                        ProblemDataGrSolids(problem_data)));
    PFModuleInvokeType(SpecStorageInvoke, specific_storage,                   //sk
                       (problem_data,
                        ProblemDataSpecificStorage(problem_data)));
    PFModuleInvokeType(SlopeInvoke, x_slope,                   //sk
                       (problem_data,
                        ProblemDataTSlopeX(problem_data),
                        ProblemDataPorosity(problem_data)));
    PFModuleInvokeType(SlopeInvoke, y_slope,                   //sk
                       (problem_data,
                        ProblemDataTSlopeY(problem_data),
                        ProblemDataPorosity(problem_data)));
    PFModuleInvokeType(ManningsInvoke, mann,                   //sk
                       (problem_data,
                        ProblemDataMannings(problem_data),
                        ProblemDataPorosity(problem_data)));
    PFModuleInvokeType(dzScaleInvoke, dz_mult,                   //RMM
                       (problem_data,
                        ProblemDataZmult(problem_data)));


    PFModuleInvokeType(realSpaceZInvoke, real_space_z,
                       (problem_data,
                        ProblemDataRealSpaceZ(problem_data)));

    PFModuleInvokeType(FBxInvoke, FBx,                   //RMM
                       (problem_data,
                        ProblemDataFBx(problem_data)));
    PFModuleInvokeType(FByInvoke, FBy,                   //RMM
                       (problem_data,
                        ProblemDataFBy(problem_data)));
    PFModuleInvokeType(FBzInvoke, FBz,                   //RMM
                       (problem_data,
                        ProblemDataFBz(problem_data)));


    (instance_xtra->site_data_not_formed) = 0;
  }

  PFModuleInvokeType(BCPressurePackageInvoke, bc_pressure, (problem_data));
}


/*--------------------------------------------------------------------------
 * SetProblemDataInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *SetProblemDataInitInstanceXtra(
                                          Problem *problem,
                                          Grid *   grid,
                                          Grid *   grid2d,
                                          double * temp_data)
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
   * Initialize data associated with `grid2d'
   *-----------------------------------------------------------------------*/

  if (grid2d != NULL)
    (instance_xtra->grid2d) = grid2d;

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `temp_data'
   *-----------------------------------------------------------------------*/

  if (temp_data != NULL)
    (instance_xtra->temp_data) = temp_data;

  /*-----------------------------------------------------------------------
   * Initialize module instances
   *-----------------------------------------------------------------------*/

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->geometries) =
      PFModuleNewInstanceType(GeometriesInitInstanceXtraInvoke, ProblemGeometries(problem), (grid));
    (instance_xtra->domain) =
      PFModuleNewInstanceType(DomainInitInstanceXtraInvoke, ProblemDomain(problem), (grid));
    (instance_xtra->permeability) =
      PFModuleNewInstanceType(SubsrfSimInitInstanceXtraInvoke, ProblemPermeability(problem), (grid, temp_data));
    (instance_xtra->porosity) =
      PFModuleNewInstanceType(SubsrfSimInitInstanceXtraInvoke,
                              ProblemPorosity(problem), (grid, temp_data));
    (instance_xtra->specific_storage) =                                       //sk
                                        PFModuleNewInstance(ProblemSpecStorage(problem), ());
    (instance_xtra->x_slope) =                                       //sk
                               PFModuleNewInstanceType(SlopeInitInstanceXtraInvoke,
                                                       ProblemXSlope(problem), (grid, grid2d));
    (instance_xtra->y_slope) =                                       //sk
                               PFModuleNewInstanceType(SlopeInitInstanceXtraInvoke,
                                                       ProblemYSlope(problem), (grid, grid2d));
    (instance_xtra->mann) =                                       //sk
                            PFModuleNewInstanceType(ManningsInitInstanceXtraInvoke,
                                                    ProblemMannings(problem), (grid, grid2d));
    (instance_xtra->dz_mult) =                                      //RMM
                               PFModuleNewInstance(ProblemdzScale(problem), ());

    (instance_xtra->FBx) =                                      //RMM
                           PFModuleNewInstance(ProblemFBx(problem), ());
    (instance_xtra->FBy) =                                      //RMM
                           PFModuleNewInstance(ProblemFBy(problem), ());
    (instance_xtra->FBz) =                                      //RMM
                           PFModuleNewInstance(ProblemFBz(problem), ());

    (instance_xtra->real_space_z) =
      PFModuleNewInstance(ProblemRealSpaceZ(problem), ());

    (instance_xtra->site_data_not_formed) = 1;

    (instance_xtra->wells) =
      PFModuleNewInstance(ProblemWellPackage(problem), ());

    (instance_xtra->bc_pressure) =
      PFModuleNewInstanceType(BCPressurePackageInitInstanceXtraInvoke,
                              ProblemBCPressurePackage(problem), (problem));
  }
  else
  {
    PFModuleReNewInstanceType(GeometriesInitInstanceXtraInvoke,
                              (instance_xtra->geometries), (grid));
    PFModuleReNewInstanceType(DomainInitInstanceXtraInvoke,
                              (instance_xtra->domain), (grid));
    PFModuleReNewInstanceType(SubsrfSimInitInstanceXtraInvoke,
                              (instance_xtra->permeability),
                              (grid, temp_data));
    PFModuleReNewInstanceType(SubsrfSimInitInstanceXtraInvoke,
                              (instance_xtra->porosity),
                              (grid, temp_data));
    PFModuleReNewInstance((instance_xtra->specific_storage), ());        //sk
    PFModuleReNewInstanceType(SlopeInitInstanceXtraInvoke,
                              (instance_xtra->x_slope), (grid, grid2d));        //sk
    PFModuleReNewInstanceType(SlopeInitInstanceXtraInvoke,
                              (instance_xtra->y_slope), (grid, grid2d));        //sk
    PFModuleReNewInstanceType(ManningsInitInstanceXtraInvoke,
                              (instance_xtra->mann), (grid, grid2d));        //sk
    PFModuleReNewInstance((instance_xtra->dz_mult), ());        //RMM
    PFModuleReNewInstance((instance_xtra->FBx), ());        //RMM
    PFModuleReNewInstance((instance_xtra->FBy), ());        //RMM
    PFModuleReNewInstance((instance_xtra->FBz), ());        //RMM

    PFModuleReNewInstance((instance_xtra->real_space_z), ());
    PFModuleReNewInstance((instance_xtra->wells), ());
    PFModuleReNewInstanceType(BCPressurePackageInitInstanceXtraInvoke,
                              (instance_xtra->bc_pressure), (problem));
  }


  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * SetProblemDataFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  SetProblemDataFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    PFModuleFreeInstance(instance_xtra->bc_pressure);
    PFModuleFreeInstance(instance_xtra->wells);

    PFModuleFreeInstance(instance_xtra->geometries);
    PFModuleFreeInstance(instance_xtra->domain);
    PFModuleFreeInstance(instance_xtra->porosity);
    PFModuleFreeInstance(instance_xtra->permeability);
    PFModuleFreeInstance(instance_xtra->specific_storage);       //sk
    PFModuleFreeInstance(instance_xtra->x_slope);       //sk
    PFModuleFreeInstance(instance_xtra->y_slope);       //sk
    PFModuleFreeInstance(instance_xtra->mann);       //sk
    PFModuleFreeInstance(instance_xtra->dz_mult);       // RMM
    PFModuleFreeInstance(instance_xtra->FBx);       // RMM
    PFModuleFreeInstance(instance_xtra->FBy);       // RMM
    PFModuleFreeInstance(instance_xtra->FBz);       // RMM

    PFModuleFreeInstance(instance_xtra->real_space_z);
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * SetProblemDataNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *SetProblemDataNewPublicXtra()
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
 * SetProblemDataFreePublicXtra
 *--------------------------------------------------------------------------*/

void  SetProblemDataFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (public_xtra)
  {
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * SetProblemDataSizeOfTempData
 *--------------------------------------------------------------------------*/

int       SetProblemDataSizeOfTempData()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int sz = 0;


  /* set `sz' to max of each of the called modules */
  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->geometries));
  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->porosity));
  sz = pfmax(sz, PFModuleSizeOfTempData(instance_xtra->permeability));

  return sz;
}
