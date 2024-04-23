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

  int    *type_density;  /* array of size num_phases of input types */
  void  **data_density;  /* array of size num_phases of pointers to Type structures */
  int    *type_energy;  /* array of size num_phases of input types */
  void  **data_energy;  /* array of size num_phases of pointers to Type structures */
} PublicXtra;

typedef void InstanceXtra;

typedef struct {
  double constant_density;
} TypeDensity0;

typedef struct {
  double constant_energy;
} TypeEnergy0;

/*-------------------------------------------------------------------------
 * PhaseDensity
 *-------------------------------------------------------------------------*/

void    InternalEnergyDensity(
                              int     phase, /* Phase */
                              Vector *pressure, /* Vector of phase pressures at each block */
                              Vector *temperature, /* Vector of phase temperature at each block */
                              Vector *energy, /* Vector of return densities at each block */
                              Vector *density, /* Double array return density */
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

  TypeDensity0         *dummy_density0;
  TypeEnergy0         *dummy_energy0;

  Grid          *grid;

  Subvector     *u_sub, *d_sub;

  // SGS
  // This code can't possibly be correct, pt is never set to point to any vector!
  double        *pt = NULL, *pp = NULL, *pu = NULL, *pd = NULL;

  Subgrid       *subgrid;

  int sg;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_d, ny_d, nz_d;

  int i, j, k, ip;

  const double a1 = 6.824687741E3;
  const double a2 = -5.422063673E2;
//   const double  a3=-2.096666205E4;
  const double a4 = 3.941286787E4;
  const double a5 = -13.466555478E4;
  const double a6 = 29.707143084E4;
  const double a7 = -4.375647096E5;
  const double a8 = 42.954208335E4;
  const double a9 = -27.067012452E4;
  const double a10 = 9.926972482E4;
  const double a11 = -16.138168904E3;
  const double a12 = 7.982692717E0;

  const double a13 = -2.616571843E-2;
  const double a14 = 1.522411790E-3;
  const double a15 = 2.284279054E-2;
  const double a16 = 2.421647003E2;
  const double a17 = 1.269716088E-10;
  const double a18 = 2.074838328E-7;
  const double a19 = 2.174020350E-8;
  const double a20 = 1.105710498E-9;
  const double a21 = 1.293441934E1;
  const double a22 = 1.308119072E-5;
  const double a23 = 6.047626338E-14;

  const double sa1 = 8.438375405E-1;
  const double sa2 = 5.362162162E-4;
  const double sa3 = 1.720000000E0;
  const double sa4 = 7.342278489E-2;
  const double sa5 = 4.975858870E-2;
  const double sa6 = 6.537154300E-1;
  const double sa7 = 1.150E-6;
  const double sa8 = 1.51080E-5;
  const double sa9 = 1.41880E-1;
  const double sa10 = 7.002753165E0;
  const double sa11 = 2.995284926E-4;
  const double sa12 = 2.040E-1;

  double tkr, tkr2, tkr3, tkr4, tkr6, tkr7, tkr8, tkr10, tkr11, tkr19, tkr18, tkr20;
  double pnmr, pnmr2, pnmr3, pnmr4;
  double y, zp, z, cz, aa1;
  double par1, par2, par3, par4, par5;
  double cc1, cc2, cc4, cc8, cc10;
  double dd1, dd2, dd4;
  double vmkr, v, yd, snum;
  double prt1, prt2, prt3, prt4, prt5;
  double bb1, bb2, ee1, ee3, entr, h;

  (void)pressure;

  switch ((public_xtra->type_density[phase]))
  {
    case 0:
    {
      double constant_energy, constant_density;
      dummy_density0 = (TypeDensity0*)(public_xtra->data_density[phase]);
      constant_density = (dummy_density0->constant_density);
      dummy_energy0 = (TypeEnergy0*)(public_xtra->data_energy[phase]);
      constant_energy = (dummy_energy0->constant_energy);

      grid = VectorGrid(energy);
      ForSubgridI(sg, GridSubgrids(grid))
      {
        subgrid = GridSubgrid(grid, sg);

        u_sub = VectorSubvector(energy, sg);
        d_sub = VectorSubvector(density, sg);

        ix = SubgridIX(subgrid) - 1;
        iy = SubgridIY(subgrid) - 1;
        iz = SubgridIZ(subgrid) - 1;

        nx = SubgridNX(subgrid) + 2;
        ny = SubgridNY(subgrid) + 2;
        nz = SubgridNZ(subgrid) + 2;

        nx_d = SubvectorNX(u_sub);
        ny_d = SubvectorNY(u_sub);
        nz_d = SubvectorNZ(u_sub);

        pu = SubvectorData(u_sub);
        pd = SubvectorData(d_sub);

        ip = 0;
        if (fcn == CALCFCN)
        {
          BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                    ip, nx_d, ny_d, nz_d, 1, 1, 1,
          {
            pu[ip] = constant_energy;
            pd[ip] = constant_density;
          });
        }
        else     /* fcn = CALCDER */
        {
          BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                    ip, nx_d, ny_d, nz_d, 1, 1, 1,
          {
            pu[ip] = 0.0;
            pd[ip] = 0.0;
          });
        }     /* End if fcn */
      }      /* End subgrid loop */

      break;
    }        /* End case 0 */

    case 1:
    {
      grid = VectorGrid(energy);
      ForSubgridI(sg, GridSubgrids(grid))
      {
        subgrid = GridSubgrid(grid, sg);

        u_sub = VectorSubvector(energy, sg);

        ix = SubgridIX(subgrid) - 1;
        iy = SubgridIY(subgrid) - 1;
        iz = SubgridIZ(subgrid) - 1;

        nx = SubgridNX(subgrid) + 2;
        ny = SubgridNY(subgrid) + 2;
        nz = SubgridNZ(subgrid) + 2;

        nx_d = SubvectorNX(u_sub);
        ny_d = SubvectorNY(u_sub);
        nz_d = SubvectorNZ(u_sub);

        pu = SubvectorData(u_sub);

        ip = 0;
        if (fcn == CALCFCN)
        {
          BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                    ip, nx_d, ny_d, nz_d, 1, 1, 1,
          {
            tkr = (pt[ip] + 273.15) / 647.3;
            tkr2 = tkr * tkr;
            tkr3 = tkr * tkr2;
            tkr4 = tkr2 * tkr2;
            tkr6 = tkr4 * tkr2;
            tkr7 = tkr4 * tkr3;
            tkr8 = tkr4 * tkr4;
            tkr10 = tkr4 * tkr6;
            tkr11 = tkr * tkr10;
            tkr19 = tkr8 * tkr11;
            tkr18 = tkr8 * tkr10;
            tkr20 = tkr10 * tkr10;
            pnmr = pp[ip] / 2.212e+7;
            pnmr2 = pnmr * pnmr;
            pnmr3 = pnmr * pnmr2;
            pnmr4 = pnmr * pnmr3;
            y = 1. - sa1 * tkr2 - sa2 / tkr6;
            zp = sa3 * y * y - 2. * sa4 * tkr + 2. * sa5 * pnmr;

            if (zp >= 0.)
            {
              z = y + sqrt(zp);
              cz = pow(z, (5. / 17.));
              par1 = a12 * sa5 / cz;
              cc1 = sa6 - tkr;
              cc2 = cc1 * cc1;
              cc4 = cc2 * cc2;
              cc8 = cc4 * cc4;
              cc10 = cc2 * cc8;
              aa1 = sa7 + tkr19;
              par2 = a13 + a14 * tkr + a15 * tkr2 + a16 * cc10 + a17 / aa1;
              par3 = (a18 + 2. * a19 * pnmr + 3. * a20 * pnmr2) / (sa8 + tkr11);
              dd1 = sa10 + pnmr;
              dd2 = dd1 * dd1;
              dd4 = dd2 * dd2;
              par4 = a21 * tkr18 * (sa9 + tkr2) * (-3. / dd4 + sa11);
              par5 = 3. * a22 * (sa12 - tkr) * pnmr2 + 4. * a23 / tkr20 * pnmr3;
              vmkr = par1 + par2 - par3 - par4 + par5;
              v = vmkr * 3.17e-3;
              pd[ip] = 1. / v;
              yd = -2. * sa1 * tkr + 6. * sa2 / tkr7;
              snum = a10 + a11 * tkr;
              snum = snum * tkr + a9;
              snum = snum * tkr + a8;
              snum = snum * tkr + a7;
              snum = snum * tkr + a6;
              snum = snum * tkr + a5;
              snum = snum * tkr + a4;
              snum = snum * tkr2 - a2;
              prt1 = a12 * (z * (17. * (z / 29. - y / 12.) + 5. * tkr * yd / 12.) + sa4 * tkr - (sa3 - 1.) * tkr * y * yd) / cz;
              prt2 = pnmr * (a13 - a15 * tkr2 + a16 * (9. * tkr + sa6) * cc8 * cc1 + a17 * (19. * tkr19 - aa1) / (aa1 * aa1));
              bb1 = sa8 + tkr11;
              bb2 = bb1 * bb1;
              prt3 = (11. * tkr11 + bb1) / bb2 * (a18 * pnmr + a19 * pnmr2 + a20 * pnmr3);
              ee1 = sa10 + pnmr;
              ee3 = ee1 * ee1 * ee1;
              prt4 = a21 * tkr18 * (17. * sa9 + 19. * tkr2) * (1. / ee3 + sa11 * pnmr);
              prt5 = a22 * sa12 * pnmr3 + 21. * a23 / tkr20 * pnmr4;
              entr = a1 * tkr - snum + prt1 + prt2 - prt3 + prt4 + prt5;
              h = entr * 70120.4;
              pu[ip] = h - pp[ip] * v;
            }
          });
        }
        else /* fcn = CALCDER */
        {
          ip = 0;
          BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                    ip, nx_d, ny_d, nz_d, 1, 1, 1,
          {
            pu[ip] = 0.0;
            pd[ip] = 0.0;
          });
        } /* End if fcn */
      }  /* End subgrid loop */

      break;
    } /*End case 1 */
  }          /* End switch */
}

/*--------------------------------------------------------------------------
 * InternalEnergyDensityInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *InternalEnergyDensityInitInstanceXtra()
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
 * InternalEnergyDensityFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  InternalEnergyDensityFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * InternalEnergyDensityNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *InternalEnergyDensityNewPublicXtra(
                                              int num_phases)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  TypeDensity0            *dummy_density0;
  TypeEnergy0             *dummy_energy0;

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

  (public_xtra->type_energy) = ctalloc(int, num_phases);
  (public_xtra->type_density) = ctalloc(int, num_phases);
  (public_xtra->data_energy) = ctalloc(void *, num_phases);
  (public_xtra->data_density) = ctalloc(void *, num_phases);

  for (i = 0; i < num_phases; i++)
  {
    sprintf(key, "Phase.%s.Density.Type",
            NA_IndexToName(GlobalsPhaseNames, i));

    switch_name = GetString(key);

    public_xtra->type_density[i] = NA_NameToIndexExitOnError(switch_na, switch_name, key);

    switch ((public_xtra->type_density[i]))
    {
      case 0:
      {
        dummy_density0 = ctalloc(TypeDensity0, 1);

        sprintf(key, "Phase.%s.Density.Value",
                NA_IndexToName(GlobalsPhaseNames, i));
        dummy_density0->constant_density = GetDouble(key);

        (public_xtra->data_density[i]) = (void*)dummy_density0;

        break;
      }

      case 1:
      {
        //No input parameters needed
        break;
      }

      default:
      {
	InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
      }
    }
  }

  for (i = 0; i < num_phases; i++)
  {
    sprintf(key, "Phase.%s.InternalEnergy.Type",
            NA_IndexToName(GlobalsPhaseNames, i));

    switch_name = GetString(key);

    public_xtra->type_energy[i] = NA_NameToIndexExitOnError(switch_na, switch_name, key);

    switch ((public_xtra->type_energy[i]))
    {
      case 0:
      {
        dummy_energy0 = ctalloc(TypeEnergy0, 1);

        sprintf(key, "Phase.%s.InternalEnergy.Value",
                NA_IndexToName(GlobalsPhaseNames, i));
        dummy_energy0->constant_energy = GetDouble(key);

        (public_xtra->data_energy[i]) = (void*)dummy_energy0;

        break;
      }

      case 1:
      {
        //No input parameters needed
        break;
      }

      default:
      {
	InputError("Invalid switch value <%s> for key <%s>", switch_name, key);      }
    }
  }

  NA_FreeNameArray(switch_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * InternalEnergyDensityFreePublicXtra
 *-------------------------------------------------------------------------*/

void  InternalEnergyDensityFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  TypeEnergy0        *dummy_energy0;
  TypeDensity0       *dummy_density0;

  int i;

  if (public_xtra)
  {
    for (i = 0; i < (public_xtra->num_phases); i++)
    {
      switch ((public_xtra->type_energy[i]))
      {
        case 0:
          dummy_energy0 = (TypeEnergy0*)(public_xtra->data_energy[i]);
          tfree(dummy_energy0);
          break;
      }

      switch ((public_xtra->type_density[i]))
      {
        case 0:
          dummy_density0 = (TypeDensity0*)(public_xtra->data_density[i]);
          tfree(dummy_density0);
          break;
      }
    }

    tfree(public_xtra->data_energy);
    tfree(public_xtra->data_density);
    tfree(public_xtra->type_energy);
    tfree(public_xtra->type_density);

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * InternalEnergyDensitySizeOfTempData
 *--------------------------------------------------------------------------*/

int  InternalEnergyDensitySizeOfTempData()
{
  return 0;
}
