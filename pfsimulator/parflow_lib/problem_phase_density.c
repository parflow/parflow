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
*  This module computes phase densities.  Currently, two types of densities
*  are supported, constant (Type0) or a basic equation of state where density
*  depends on pressure (Type1).
*
*  The equation of state used is:
*  rho(p) = rho_ref exp(c p)
*  where rho_ref is the density at atmoshperic pressure and c is the
*  phase compressibility constant.
*
*  The density module can be invoked either expecting only a
*  double array of densities back - where NULL Vectors are
*  sent in for the phase pressure and the density return Vector - or a
*  Vector of densities at each grid block.  Note that code using the
*  Vector density option can also have a constant density.
*  This "overloading" was provided so that the density module written
*  for the Richards' solver modules would be backward compatible with
*  the Impes modules and so that densities can be evaluated for pressures
*  not necessarily associated with a grid (as in boundary patches).
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
  double reference_density;
  double compressibility_constant;
} Type1;                      /* rho_ref exp(compressibility*pressure) */

/*-------------------------------------------------------------------------
 * PhaseDensityConstants
 * Used to get the constant values for when density vector is always NULL
 *-------------------------------------------------------------------------*/

void    PhaseDensityConstants(int     phase,
                              int     fcn,
                              int *   phase_type,
                              double *constant,
                              double *ref_den,
                              double *comp_const)
{
  PFModule *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  Type0 *dummy0;
  Type1 *dummy1;

  (*phase_type) = public_xtra->type[phase];
  switch (*phase_type)
  {
    case 0:
      if (fcn == CALCFCN)
      {
        dummy0 = (Type0*)(public_xtra->data[phase]);
        (*constant) = dummy0->constant;
      }
      else
      {
        (*constant) = 0.0;
      }
      break;

    case 1:
      dummy1 = (Type1*)(public_xtra->data[phase]);
      (*ref_den) = dummy1->reference_density;
      (*comp_const) = dummy1->compressibility_constant;
      break;
  }
}

/*-------------------------------------------------------------------------
 * PhaseDensity
 *-------------------------------------------------------------------------*/

void    PhaseDensity(
                     int     phase, /* Phase */
                     Vector *phase_pressure, /* Vector of phase pressures at each block */
                     Vector *density_v, /* Vector of return densities at each block */
                     double *pressure_d, /* Double array of pressures */
                     double *density_d, /* Double array return density */
                     int     fcn) /* Flag determining what to calculate
                                   * fcn = CALCFCN => calculate the function value
                                   * fcn = CALCDER => calculate the function
                                   *                  derivative */

/*  Module returns either a double array or Vector of densities.
 *  If density_v is NULL, then a double array is returned.
 *  This "overloading" was provided so that the density module written
 *  for the Richards' solver modules would be backward compatible with
 *  the Impes modules.
 */
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0         *dummy0;
  Type1         *dummy1;

  Grid          *grid;

  Subvector     *p_sub;
  Subvector     *d_sub;

  double        *pp;
  double        *dp;

  Subgrid       *subgrid;

  int sg;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_p, ny_p, nz_p;
  int nx_d, ny_d, nz_d;

  int i, j, k, ip, id;


  switch ((public_xtra->type[phase]))
  {
    case 0:
    {
      double constant;
      dummy0 = (Type0*)(public_xtra->data[phase]);
      constant = (dummy0->constant);

      if (density_v != NULL)
      {
        grid = VectorGrid(density_v);
        ForSubgridI(sg, GridSubgrids(grid))
        {
          subgrid = GridSubgrid(grid, sg);

          d_sub = VectorSubvector(density_v, sg);

          ix = SubgridIX(subgrid) - 1;
          iy = SubgridIY(subgrid) - 1;
          iz = SubgridIZ(subgrid) - 1;

          nx = SubgridNX(subgrid) + 2;
          ny = SubgridNY(subgrid) + 2;
          nz = SubgridNZ(subgrid) + 2;

          nx_d = SubvectorNX(d_sub);
          ny_d = SubvectorNY(d_sub);
          nz_d = SubvectorNZ(d_sub);

          dp = SubvectorElt(d_sub, ix, iy, iz);

          id = 0;
          if (fcn == CALCFCN)
          {
            BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                      id, nx_d, ny_d, nz_d, 1, 1, 1,
            {
              dp[id] = constant;
            });
          }
          else   /* fcn = CALCDER */
          {
            BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                      id, nx_d, ny_d, nz_d, 1, 1, 1,
            {
              dp[id] = 0.0;
            });
          }   /* End if fcn */
        }    /* End subgrid loop */
      }      /* End if density_v is not NULL */
      else
      {
        if (fcn == CALCFCN)
        {
          (*density_d) = constant;
        }
        else  /* fcn = CALCDER */
        {
          (*density_d) = 0.0;
        }
      }      /* End else */
      break;
    }        /* End case 0 */

    case 1:
    {
      double ref, comp;
      dummy1 = (Type1*)(public_xtra->data[phase]);
      ref = (dummy1->reference_density);
      comp = (dummy1->compressibility_constant);

      if (density_v != NULL)
      {
        grid = VectorGrid(density_v);
        ForSubgridI(sg, GridSubgrids(grid))
        {
          subgrid = GridSubgrid(grid, sg);

          p_sub = VectorSubvector(phase_pressure, sg);
          d_sub = VectorSubvector(density_v, sg);

          ix = SubgridIX(subgrid) - 1;
          iy = SubgridIY(subgrid) - 1;
          iz = SubgridIZ(subgrid) - 1;

          nx = SubgridNX(subgrid) + 2;
          ny = SubgridNY(subgrid) + 2;
          nz = SubgridNZ(subgrid) + 2;

          nx_p = SubvectorNX(p_sub);
          ny_p = SubvectorNY(p_sub);
          nz_p = SubvectorNZ(p_sub);

          nx_d = SubvectorNX(d_sub);
          ny_d = SubvectorNY(d_sub);
          nz_d = SubvectorNZ(d_sub);

          pp = SubvectorElt(p_sub, ix, iy, iz);
          dp = SubvectorElt(d_sub, ix, iy, iz);

          ip = 0;
          id = 0;

          if (fcn == CALCFCN)
          {
            BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                      ip, nx_p, ny_p, nz_p, 1, 1, 1,
                      id, nx_d, ny_d, nz_d, 1, 1, 1,
            {
              dp[id] = ref * exp(pp[ip] * comp);
            });
          }
          else          /* fcn = CALCDER */
          {
            BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                      ip, nx_p, ny_p, nz_p, 1, 1, 1,
                      id, nx_d, ny_d, nz_d, 1, 1, 1,
            {
              dp[id] = comp * ref * exp(pp[ip] * comp);
            });
          }
        }
      }
      else
      {
        if (fcn == CALCFCN)
        {
          (*density_d) = ref * exp((*pressure_d) * comp);
        }
        else
        {
          (*density_d) = comp * ref * exp((*pressure_d) * comp);
        }
      }

      break;
    }        /* End case 1 */
  }          /* End switch */
}

/*--------------------------------------------------------------------------
 * PhaseDensityInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *PhaseDensityInitInstanceXtra()
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
 * PhaseDensityFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  PhaseDensityFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * PhaseDensityNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *PhaseDensityNewPublicXtra(
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
  switch_na = NA_NewNameArray("Constant EquationOfState");

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

        sprintf(key, "Phase.%s.Density.Value",
                NA_IndexToName(GlobalsPhaseNames, i));
        dummy0->constant = GetDouble(key);

        (public_xtra->data[i]) = (void*)dummy0;

        break;
      }

      case 1:
      {
        dummy1 = ctalloc(Type1, 1);

        sprintf(key, "Phase.%s.Density.ReferenceDensity",
                NA_IndexToName(GlobalsPhaseNames, i));
        dummy1->reference_density = GetDouble(key);

        sprintf(key, "Phase.%s.Density.CompressibiltyConstant",
                NA_IndexToName(GlobalsPhaseNames, i));
        dummy1->compressibility_constant = GetDouble(key);

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
 * PhaseDensityFreePublicXtra
 *-------------------------------------------------------------------------*/

void  PhaseDensityFreePublicXtra()
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
 * PhaseDensitySizeOfTempData
 *--------------------------------------------------------------------------*/

int  PhaseDensitySizeOfTempData()
{
  return 0;
}
