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
#include "water_balance.h"

/*-----------------------------------------------------------------------
 * ComputeSurfaceStorage:
 *
 * Computes the subsurface storage at the top of the domain defined by the
 * top databox index values.
 *
 * Returns a Databox surface_storage with subsurface storage values at
 * each (i,j) location.
 *
 *-----------------------------------------------------------------------*/

#include <math.h>

void ComputeSurfaceStorage(
                           Databox *top,
                           Databox *pressure,
                           Databox *surface_storage)
{
  int i, j;
  int nx, ny, nz;
  double dx, dy;

  nx = DataboxNx(pressure);
  ny = DataboxNy(pressure);
  nz = DataboxNz(pressure);

  dx = DataboxDx(pressure);
  dy = DataboxDy(pressure);

  for (j = 0; j < ny; j++)
  {
    for (i = 0; i < nx; i++)
    {
      int k = *(DataboxCoeff(top, i, j, 0));
      if (k < 0)
      {
        *(DataboxCoeff(surface_storage, i, j, 0)) = 0.0;
      }
      else if (k < nz)
      {
        if (*(DataboxCoeff(pressure, i, j, k)) > 0)
        {
          *(DataboxCoeff(surface_storage, i, j, 0)) = *(DataboxCoeff(pressure, i, j, k)) * dx * dy;
        }
      }
      else
      {
        printf("Error: Index in top (k=%d) is outside of domain (nz=%d)\n", k, nz);
      }
    }
  }
}

void ComputeSubsurfaceStorage(Databox *mask,
                              Databox *porosity,
                              Databox *pressure,
                              Databox *saturation,
                              Databox *specific_storage,
                              Databox *subsurface_storage)
{
  int m;

  double *mask_coeff = DataboxCoeffs(mask);
  double *porosity_coeff = DataboxCoeffs(porosity);
  double *pressure_coeff = DataboxCoeffs(pressure);
  double *saturation_coeff = DataboxCoeffs(saturation);
  double *specific_storage_coeff = DataboxCoeffs(specific_storage);
  double *subsurface_storage_coeff = DataboxCoeffs(subsurface_storage);

  int nx = DataboxNx(pressure);
  int ny = DataboxNy(pressure);
  int nz = DataboxNz(pressure);

  double dx = DataboxDx(pressure);
  double dy = DataboxDy(pressure);
  double dz = DataboxDz(pressure);

  for (m = 0; m < (nx * ny * nz); m++)
  {
    if (mask_coeff[m] > 0)
    {
      subsurface_storage_coeff[m] = saturation_coeff[m] * porosity_coeff[m] * dx * dy * dz;
      /* @RMM mod to remove porosity from subsurface storage */
      /* OLD way
       * subsurface_storage_coeff[m] += pressure_coeff[m] * specific_storage_coeff[m] * saturation_coeff[m] * porosity_coeff[m] * dx * dy * dz;
       */
      subsurface_storage_coeff[m] += pressure_coeff[m] * specific_storage_coeff[m] * saturation_coeff[m] * dx * dy * dz;
    }
  }
}


void ComputeGWStorage(Databox *mask,
                      Databox *porosity,
                      Databox *pressure,
                      Databox *saturation,
                      Databox *specific_storage,
                      Databox *gw_storage)
{
  int m;

  double *mask_coeff = DataboxCoeffs(mask);
  double *porosity_coeff = DataboxCoeffs(porosity);
  double *pressure_coeff = DataboxCoeffs(pressure);
  double *saturation_coeff = DataboxCoeffs(saturation);
  double *specific_storage_coeff = DataboxCoeffs(specific_storage);
  double *gw_storage_coeff = DataboxCoeffs(gw_storage);

  int nx = DataboxNx(pressure);
  int ny = DataboxNy(pressure);
  int nz = DataboxNz(pressure);

  double dx = DataboxDx(pressure);
  double dy = DataboxDy(pressure);
  double dz = DataboxDz(pressure);

  for (m = 0; m < (nx * ny * nz); m++)
  {
    if (mask_coeff[m] > 0 && saturation_coeff[m] == 1.0)
    {
      /* IMF -- Same as above, except only sums over saturated cells */
      gw_storage_coeff[m] = saturation_coeff[m] * porosity_coeff[m] * dx * dy * dz;
      gw_storage_coeff[m] += pressure_coeff[m] * specific_storage_coeff[m] * saturation_coeff[m] * dx * dy * dz;
    }
  }
}


/*
 * This is from nl_function_eval for comparison.
 *                         qx_[io] = dir_x * (RPowerR(fabs(x_sl_dat[io]),0.5) / mann_dat[io]) * RPowerR(max((pp[ip]),0.0),(5.0/3.0));
 */

void ComputeSurfaceRunoff(Databox *top,
                          Databox *slope_x,
                          Databox *slope_y,
                          Databox *mannings,
                          Databox *pressure,
                          Databox *surface_runoff)
{
  int i, j;
  int nx, ny;
  double dx, dy;

  nx = DataboxNx(pressure);
  ny = DataboxNy(pressure);

  dx = DataboxDx(pressure);
  dy = DataboxDy(pressure);

  for (i = 0; i < nx; i++)
  {
    for (j = 0; j < ny; j++)
    {
      int k = *(DataboxCoeff(top, i, j, 0));

      /*
       * If top is non negative then moving into active region
       */
      if (!(k < 0))
      {
        /*
         * Compute runnoff if slope is running off of active region
         */
        if (*DataboxCoeff(slope_y, i, j, 0) > 0)
        {
          if (*DataboxCoeff(pressure, i, j, k) > 0)
          {
            *DataboxCoeff(surface_runoff, i, j, 0) =
              (sqrt(fabs(*DataboxCoeff(slope_y, i, j, 0))) / *DataboxCoeff(mannings, i, j, 0)) *
              pow(*DataboxCoeff(pressure, i, j, k), 5.0 / 3.0) * dx;
          }
        }

        /*
         * Loop until going back outside of active area
         */
        while ((j + 1 < ny) && !(*DataboxCoeff(top, i, j + 1, 0) < 0))
        {
          j++;
        }

        /*
         * Found either domain boundary or outside of active area.
         * Compute runnoff if slope is running off of active region.
         */

        k = *(DataboxCoeff(top, i, j, 0));
        if (*DataboxCoeff(slope_y, i, j, 0) < 0)
        {
          if (*DataboxCoeff(pressure, i, j, k) > 0)
          {
            *DataboxCoeff(surface_runoff, i, j, 0) =
              (sqrt(fabs(*DataboxCoeff(slope_y, i, j, 0))) / *DataboxCoeff(mannings, i, j, 0)) *
              pow(*DataboxCoeff(pressure, i, j, k), 5.0 / 3.0) * dx;
          }
        }
      }
    }
  }


  for (j = 0; j < ny; j++)
  {
    for (i = 0; i < nx; i++)
    {
      int k = *(DataboxCoeff(top, i, j, 0));

      /*
       * If top is non negative then moving into active region
       */
      if (!(k < 0))
      {
        /*
         * Compute runnoff if slope is running off of active region
         */
        if (*DataboxCoeff(slope_x, i, j, 0) > 0)
        {
          if (*DataboxCoeff(pressure, i, j, k) > 0)
          {
            *DataboxCoeff(surface_runoff, i, j, 0) =
              (sqrt(fabs(*DataboxCoeff(slope_x, i, j, 0))) / *DataboxCoeff(mannings, i, j, 0)) *
              pow(*DataboxCoeff(pressure, i, j, k), 5.0 / 3.0) * dy;
          }
        }

        /*
         * Loop until going back outside of active area
         */
        while ((i + 1 < nx) && !(*DataboxCoeff(top, i + 1, j, 0) < 0))
        {
          i++;
        }

        /*
         * Found either domain boundary or outside of active area.
         * Compute runnoff if slope is running off of active region.
         */
        k = *(DataboxCoeff(top, i, j, 0));
        if (*DataboxCoeff(slope_x, i, j, 0) < 0)
        {
          if (*DataboxCoeff(pressure, i, j, k) > 0)
          {
            *DataboxCoeff(surface_runoff, i, j, 0) =
              (sqrt(fabs(*DataboxCoeff(slope_x, i, j, 0))) / *DataboxCoeff(mannings, i, j, 0)) *
              pow(*DataboxCoeff(pressure, i, j, k), 5.0 / 3.0) * dy;
          }
        }
      }
    }
  }
}

