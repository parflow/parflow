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
*****************************************************************************/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int num_phases;
  double satconstitutive;
} PublicXtra;

typedef struct {
  Grid    *grid;
} InstanceXtra;

/*--------------------------------------------------------------------------
 * SaturationConstitutive
 *--------------------------------------------------------------------------*/

void     SaturationConstitutive(
                                Vector **phase_saturations)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int num_phases = (public_xtra->num_phases);
  double satconstitutive = (public_xtra->satconstitutive);

  Grid          *grid = (instance_xtra->grid);

  SubgridArray  *subgrids;

  Subvector     *subvector_ps, *subvector_psi;

  double        *ps, *psi;

  int i, j, k;
  int ix, iy, iz;
  int nx, ny, nz;

  int nx_ps, ny_ps, nz_ps;
  int nx_psi, ny_psi, nz_psi;

  int sg, ips, ipsi;


  subgrids = GridSubgrids(grid);


  ForSubgridI(sg, subgrids)
  {
    subvector_ps = VectorSubvector(phase_saturations[num_phases - 1], sg);

    ix = SubvectorIX(subvector_ps);
    iy = SubvectorIY(subvector_ps);
    iz = SubvectorIZ(subvector_ps);

    nx = SubvectorNX(subvector_ps);
    ny = SubvectorNY(subvector_ps);
    nz = SubvectorNZ(subvector_ps);

    nx_ps = SubvectorNX(subvector_ps);
    ny_ps = SubvectorNY(subvector_ps);
    nz_ps = SubvectorNZ(subvector_ps);

    ps = SubvectorElt(subvector_ps, ix, iy, iz);

    ips = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              ips, nx_ps, ny_ps, nz_ps, 1, 1, 1,
    {
      ps[ips] = satconstitutive;
    });
  }


  for (i = 0; i < num_phases - 1; i++)
  {
    ForSubgridI(sg, subgrids)
    {
      subvector_ps = VectorSubvector(phase_saturations[num_phases - 1], sg);
      subvector_psi = VectorSubvector(phase_saturations[i], sg);

      ix = SubvectorIX(subvector_ps);
      iy = SubvectorIY(subvector_ps);
      iz = SubvectorIZ(subvector_ps);

      nx = SubvectorNX(subvector_ps);
      ny = SubvectorNY(subvector_ps);
      nz = SubvectorNZ(subvector_ps);

      nx_ps = SubvectorNX(subvector_ps);
      ny_ps = SubvectorNY(subvector_ps);
      nz_ps = SubvectorNZ(subvector_ps);

      nx_psi = SubvectorNX(subvector_psi);
      ny_psi = SubvectorNY(subvector_psi);
      nz_psi = SubvectorNZ(subvector_psi);

      ps = SubvectorElt(subvector_ps, ix, iy, iz);
      psi = SubvectorElt(subvector_psi, ix, iy, iz);

      ips = 0;
      ipsi = 0;
      BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                ips, nx_ps, ny_ps, nz_ps, 1, 1, 1,
                ipsi, nx_psi, ny_psi, nz_psi, 1, 1, 1,
      {
        ps[ips] -= psi[ipsi];
      });
    }
  }
}

/*--------------------------------------------------------------------------
 * SaturationConstitutiveInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *SaturationConstitutiveInitInstanceXtra(
                                                  Grid *grid)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (grid != NULL)
  {
    (instance_xtra->grid) = grid;
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;

  return this_module;
}


/*--------------------------------------------------------------------------
 * SaturationConstitutiveFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  SaturationConstitutiveFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * SaturationConstitutiveNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *SaturationConstitutiveNewPublicXtra(
                                               int num_phases)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  /*-----------------------------------------------------------------------
   * Setup the PublicXtra structure
   *-----------------------------------------------------------------------*/

  public_xtra = ctalloc(PublicXtra, 1);

  /*-------------------------------------------------------------*/
  /*                     setup parameters                        */

  (public_xtra->num_phases) = num_phases;

  (public_xtra->satconstitutive) = 1.0;

  /*-------------------------------------------------------------*/

  PFModulePublicXtra(this_module) = public_xtra;

  return this_module;
}

/*-------------------------------------------------------------------------
 * SaturationConstitutiveFreePublicXtra
 *-------------------------------------------------------------------------*/

void  SaturationConstitutiveFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * SaturationConstitutiveSizeOfTempData
 *--------------------------------------------------------------------------*/

int  SaturationConstitutiveSizeOfTempData()
{
  return 0;
}
