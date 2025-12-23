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

typedef struct {
  int solution_type;
  void *data;
} PublicXtra;

typedef void InstanceXtra;

typedef struct {
  double value;
} Type1;            /* Constant */

/*--------------------------------------------------------------------------
 * L2ErrorNorm
 *--------------------------------------------------------------------------*/

void         L2ErrorNorm(
                         double       time,
                         Vector *     pressure,
                         ProblemData *problem_data,
                         double *     l2_error_norm)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Grid             *grid = VectorGrid(pressure);

  Type1          *dummy1;

  SubgridArray     *subgrids = GridSubgrids(grid);

  Subgrid          *subgrid;
  Subvector        *p_sub;

  GrGeomSolid      *gr_domain;

  amps_Invoice result_invoice;

  double           *data;
  double err, soln;
  double x, y, z;
  double dx, dy, dz, vol = 0.0;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;
  int is, i, j, k, ips;

  /*-----------------------------------------------------------------------
   * Calculate l2-norm of the error based on exact solution known
   *-----------------------------------------------------------------------*/

  err = 0.0;

  gr_domain = ProblemDataGrDomain(problem_data);

  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);
    p_sub = VectorSubvector(pressure, is);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    vol = dx * dy * dz;

    /* RDF: assume resolution is the same in all 3 directions */
    r = SubgridRX(subgrid);

    data = SubvectorData(p_sub);

    switch ((public_xtra->solution_type))
    {
      case 0:    /* No known exact solution */
      {
        (*l2_error_norm) = -1.0;
        break;
      }

      case 1:     /* p = constant */
      {
        dummy1 = (Type1*)(public_xtra->data);
        soln = (dummy1->value);

        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ips = SubvectorEltIndex(p_sub, i, j, k);

          err += (data[ips] - soln) * (data[ips] - soln);
        });
        break;
      }      /* End case p = constant */

      case 2:     /* p = x */
      {
        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ips = SubvectorEltIndex(p_sub, i, j, k);

          x = RealSpaceX(i, SubgridRX(subgrid));

          soln = x;
          err += (data[ips] - soln) * (data[ips] - soln);
        });
        break;
      }      /* End case p = x */

      case 3:     /* p = x + y +z */
      {
        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ips = SubvectorEltIndex(p_sub, i, j, k);

          x = RealSpaceX(i, SubgridRX(subgrid));
          y = RealSpaceY(j, SubgridRY(subgrid));
          z = RealSpaceZ(k, SubgridRZ(subgrid));

          soln = x + y + z;
          err += (data[ips] - soln) * (data[ips] - soln);
        });
        break;
      }      /* End case p = x + y + z */

      case 4:     /* p = x^3y^2 + sinxy + 1 */
      {
        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ips = SubvectorEltIndex(p_sub, i, j, k);

          x = RealSpaceX(i, SubgridRX(subgrid));
          y = RealSpaceY(j, SubgridRY(subgrid));

          soln = x * x * x * y * y + sin(x * y) + 1;
          err += (data[ips] - soln) * (data[ips] - soln);
        });
        break;
      }     /* p = x^3y^2 + sinxy + 1 */

      case 5:     /* p = x^3y^4 + x^2 + sinxy cosy +1 */
      {
        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ips = SubvectorEltIndex(p_sub, i, j, k);

          x = RealSpaceX(i, SubgridRX(subgrid));
          y = RealSpaceY(j, SubgridRY(subgrid));
          z = RealSpaceZ(k, SubgridRZ(subgrid));

          soln = pow(x, 3) * pow(y, 4) + x * x + sin(x * y) * cos(y) + 1;
          err += (data[ips] - soln) * (data[ips] - soln);
        });
        break;
      }      /* End case  p = x^3y^4 + x^2 + sinxy cosy +1 */

      case 6:     /* p = x*y*z*t + 1*/
      {
        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ips = SubvectorEltIndex(p_sub, i, j, k);

          x = RealSpaceX(i, SubgridRX(subgrid));
          y = RealSpaceY(j, SubgridRY(subgrid));
          z = RealSpaceZ(k, SubgridRZ(subgrid));

          soln = x * y * z * time + 1;
          err += (data[ips] - soln) * (data[ips] - soln);
        });
        break;
      }      /* End case  p = x*y*z*t + 1*/

      case 7:     /* p = x*y*z*t + 1*/
      {
        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ips = SubvectorEltIndex(p_sub, i, j, k);

          x = RealSpaceX(i, SubgridRX(subgrid));
          y = RealSpaceY(j, SubgridRY(subgrid));
          z = RealSpaceZ(k, SubgridRZ(subgrid));

          soln = x * y * z * time + 1;
          err += (data[ips] - soln) * (data[ips] - soln);
        });
        break;
      }      /* End case  p = x*y*z*t + 1*/
    }     /* End switch statement for solution types */
  }       /* End subgrid loop */

  if ((public_xtra->solution_type) > 0)
  {
    result_invoice = amps_NewInvoice("%d", &err);
    amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
    amps_FreeInvoice(result_invoice);

    (*l2_error_norm) = sqrt((err * vol));
  }
}

/*--------------------------------------------------------------------------
 * L2ErrorNormInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *L2ErrorNormInitInstanceXtra()
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
 * L2ErrorNormFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  L2ErrorNormFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * L2ErrorNormNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *L2ErrorNormNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type1         *dummy1;

  char          *switch_name;
  char key[IDB_MAX_KEY_LEN];

  NameArray switch_na;

  /*----------------------------------------------------------
   * The name array to map names to switch values
   *----------------------------------------------------------*/
  switch_na = NA_NewNameArray("NoKnownSolution Constant X XPlusYPlusZ \
                             X3Y2PlusSinXYPlus1 X3Y4PlusX2PlusSinXYCosYPlus1 \
                             XYZTPlus1 XYZTPlus1PermTensor");

  public_xtra = ctalloc(PublicXtra, 1);

  sprintf(key, "KnownSolution");

  switch_name = GetString(key);
  public_xtra->solution_type = NA_NameToIndexExitOnError(switch_na, switch_name, key);

  switch ((public_xtra->solution_type))
  {
    case 0:
    {
      break;
    }

    case 1:
    {
      dummy1 = ctalloc(Type1, 1);

      dummy1->value = GetDouble("KnownSolution.Value");

      (public_xtra->data) = (void*)dummy1;

      break;
    }     /* End case 1 */

    case 2:
    {
      break;
    }

    case 3:
    {
      break;
    }

    case 4:
    {
      break;
    }

    case 5:
    {
      break;
    }

    case 6:
    {
      break;
    }

    case 7:
    {
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
      break;
    }
  }    /* End switch statement */

  PFModulePublicXtra(this_module) = public_xtra;

  NA_FreeNameArray(switch_na);

  return this_module;
}

/*-------------------------------------------------------------------------
 * L2ErrorNormFreePublicXtra
 *-------------------------------------------------------------------------*/

void  L2ErrorNormFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type1       *dummy1;

  if (public_xtra)
  {
    switch ((public_xtra->solution_type))
    {
      case 1:
      {
        dummy1 = (Type1*)(public_xtra->data);

        tfree(dummy1);
        break;
      }
    }

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * L2ErrorNormSizeOfTempData
 *--------------------------------------------------------------------------*/

int  L2ErrorNormSizeOfTempData()
{
  return 0;
}
