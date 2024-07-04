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

  int    *type;  /* array of size num_phases of input types */
  void  **data;  /* array of size num_phases of pointers to Type structures */
} PublicXtra;

typedef void InstanceXtra;

typedef struct {
  double constant;
} Type0;

typedef struct {
  double irreducible_saturation;
  double exponent;
} Type1;                      /* (S_i - S_i0)^{a_i} */


/*--------------------------------------------------------------------------
 * PhaseMobility
 *--------------------------------------------------------------------------*/

void    PhaseMobility(
                      Vector *phase_mobility_x,
                      Vector *phase_mobility_y,
                      Vector *phase_mobility_z,
                      Vector *perm_x,
                      Vector *perm_y,
                      Vector *perm_z,
                      int     phase,
                      Vector *phase_saturation,
                      double  phase_viscosity)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0         *dummy0;
  Type1         *dummy1;

  double pv = phase_viscosity;

  Grid          *grid = VectorGrid(phase_mobility_x);

  Subvector     *kx_sub;
  Subvector     *ky_sub;
  Subvector     *kz_sub;
  Subvector     *pmx_sub;
  Subvector     *pmz_sub;
  Subvector     *pmy_sub;
  Subvector     *ps_sub;

  double        *kxp, *kyp, *kzp;
  double        *pmxp, *pmyp, *pmzp;
  double        *psp;

  Subgrid       *subgrid;

  int sg;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_k, ny_k, nz_k;
  int nx_pm, ny_pm, nz_pm;
  int nx_ps, ny_ps, nz_ps;

  int i, j, k, ik, ipm, ips;

  switch ((public_xtra->type[phase]))
  {
    case 0:
    {
      double constant;

      dummy0 = (Type0*)(public_xtra->data[phase]);

      constant = (dummy0->constant);

      ForSubgridI(sg, GridSubgrids(grid))
      {
        subgrid = GridSubgrid(grid, sg);

        kx_sub = VectorSubvector(perm_x, sg);
        ky_sub = VectorSubvector(perm_y, sg);
        kz_sub = VectorSubvector(perm_z, sg);
        pmx_sub = VectorSubvector(phase_mobility_x, sg);
        pmy_sub = VectorSubvector(phase_mobility_y, sg);
        pmz_sub = VectorSubvector(phase_mobility_z, sg);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        nx_k = SubvectorNX(kx_sub);
        ny_k = SubvectorNY(kx_sub);
        nz_k = SubvectorNZ(kx_sub);

        nx_pm = SubvectorNX(pmx_sub);
        ny_pm = SubvectorNY(pmx_sub);
        nz_pm = SubvectorNZ(pmx_sub);

        kxp = SubvectorElt(kx_sub, ix, iy, iz);
        kyp = SubvectorElt(ky_sub, ix, iy, iz);
        kzp = SubvectorElt(kz_sub, ix, iy, iz);
        pmxp = SubvectorElt(pmx_sub, ix, iy, iz);
        pmyp = SubvectorElt(pmy_sub, ix, iy, iz);
        pmzp = SubvectorElt(pmz_sub, ix, iy, iz);

        ik = 0;
        ipm = 0;
        BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                  ik, nx_k, ny_k, nz_k, 1, 1, 1,
                  ipm, nx_pm, ny_pm, nz_pm, 1, 1, 1,
        {
          pmxp[ipm] = kxp[ik] * constant / pv;
          pmyp[ipm] = kyp[ik] * constant / pv;
          pmzp[ipm] = kzp[ik] * constant / pv;
        });
      }

      break;
    }

    case 1:
    {
      double ps0, e;

      dummy1 = (Type1*)(public_xtra->data[phase]);

      ps0 = (dummy1->irreducible_saturation);
      e = (dummy1->exponent);

      ForSubgridI(sg, GridSubgrids(grid))
      {
        subgrid = GridSubgrid(grid, sg);

        kx_sub = VectorSubvector(perm_x, sg);
        ky_sub = VectorSubvector(perm_y, sg);
        kz_sub = VectorSubvector(perm_z, sg);

        pmx_sub = VectorSubvector(phase_mobility_x, sg);
        pmy_sub = VectorSubvector(phase_mobility_y, sg);
        pmz_sub = VectorSubvector(phase_mobility_z, sg);

        ps_sub = VectorSubvector(phase_saturation, sg);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        nx_k = SubvectorNX(kx_sub);
        ny_k = SubvectorNY(kx_sub);
        nz_k = SubvectorNZ(kx_sub);

        nx_pm = SubvectorNX(pmx_sub);
        ny_pm = SubvectorNY(pmx_sub);
        nz_pm = SubvectorNZ(pmx_sub);

        nx_ps = SubvectorNX(ps_sub);
        ny_ps = SubvectorNY(ps_sub);
        nz_ps = SubvectorNZ(ps_sub);

        kxp = SubvectorElt(kx_sub, ix, iy, iz);
        kyp = SubvectorElt(ky_sub, ix, iy, iz);
        kzp = SubvectorElt(kz_sub, ix, iy, iz);

        pmxp = SubvectorElt(pmx_sub, ix, iy, iz);
        pmyp = SubvectorElt(pmy_sub, ix, iy, iz);
        pmzp = SubvectorElt(pmz_sub, ix, iy, iz);

        psp = SubvectorElt(ps_sub, ix, iy, iz);

        ik = 0;
        ipm = 0;
        ips = 0;
        BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
                  ik, nx_k, ny_k, nz_k, 1, 1, 1,
                  ipm, nx_pm, ny_pm, nz_pm, 1, 1, 1,
                  ips, nx_ps, ny_ps, nz_ps, 1, 1, 1,
        {
          pmxp[ipm] = kxp[ik] * pow((psp[ips] - ps0), e) / pv;
          pmyp[ipm] = kyp[ik] * pow((psp[ips] - ps0), e) / pv;
          pmzp[ipm] = kzp[ik] * pow((psp[ips] - ps0), e) / pv;
        });
      }

      break;
    }
  }
}

/*--------------------------------------------------------------------------
 * PhaseMobilityInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *PhaseMobilityInitInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

#if 0
  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
#endif
  instance_xtra = NULL;

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * PhaseMobilityFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  PhaseMobilityFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * PhaseMobilityNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *PhaseMobilityNewPublicXtra(
                                      int num_phases)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0            *dummy0;
  Type1            *dummy1;

  char          *switch_name;
  NameArray switch_na;
  char key[IDB_MAX_KEY_LEN];

  int i;

  /*----------------------------------------------------------
   * The name array to map names to switch values
   *----------------------------------------------------------*/
  switch_na = NA_NewNameArray("Constant Polynomial");

  public_xtra = ctalloc(PublicXtra, 1);

  (public_xtra->num_phases) = num_phases;

  (public_xtra->type) = ctalloc(int, num_phases);
  (public_xtra->data) = ctalloc(void *, num_phases);

  for (i = 0; i < num_phases; i++)
  {
    sprintf(key, "Phase.%s.Density.Type",
            NA_IndexToName(GlobalsPhaseNames, i));

    switch_name = GetString(key);

    public_xtra->type[i] = NA_NameToIndexExitOnError(switch_na, switch_name, key);

    switch ((public_xtra->type[i]))
    {
      case 0:
      {
        dummy0 = ctalloc(Type0, 1);

        sprintf(key, "Phase.%s.Mobility.Value",
                NA_IndexToName(GlobalsPhaseNames, i));
        dummy0->constant = GetDouble(key);

        (public_xtra->data[i]) = (void*)dummy0;

        break;
      }

      case 1:
      {
        dummy1 = ctalloc(Type1, 1);
        sprintf(key, "Phase.%s.Mobility.Exponent",
                NA_IndexToName(GlobalsPhaseNames, i));
        dummy1->exponent = GetDoubleDefault(key, 2.0);
        sprintf(key, "Phase.%s.Mobility.IrreducibleSaturation",
                NA_IndexToName(GlobalsPhaseNames, i));
        dummy1->irreducible_saturation = GetDoubleDefault(key, 0.0);

        (public_xtra->data[i]) = (void*)dummy1;

        break;
      }

      default:
      {
	InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
      }
    }
  }

  NA_FreeNameArray(switch_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * PhaseMobilityFreePublicXtra
 *-------------------------------------------------------------------------*/

void  PhaseMobilityFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0        *dummy0;
  Type1        *dummy1;

  int i;

  if (public_xtra)
  {
    for (i = 0; i < (public_xtra->num_phases); i++)
    {
      switch ((public_xtra->type[i]))
      {
        case 0:
          dummy0 = (Type0*)(public_xtra->data[i]);
          tfree(dummy0);
          break;

        case 1:
          dummy1 = (Type1*)(public_xtra->data[i]);
          tfree(dummy1);
          break;
      }
    }

    tfree(public_xtra->data);
    tfree(public_xtra->type);

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * PhaseMobilitySizeOfTempData
 *--------------------------------------------------------------------------*/

int  PhaseMobilitySizeOfTempData()
{
  return 0;
}
