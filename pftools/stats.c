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
* Print various statistics
*
* (C) 1995 Regents of the University of California.
*
* $Revision: 1.4 $
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "stats.h"

/*-----------------------------------------------------------------------
 * Print various statistics
 *-----------------------------------------------------------------------*/

void        Stats(
                  Databox *databox,
                  double * min,
                  double * max,
                  double * mean,
                  double * sum,
                  double * variance,
                  double * stdev)
{
  double  *dp;

  double dtmp;

  int i, j, k;
  int nx, ny, nz, n;


  nx = DataboxNx(databox);
  ny = DataboxNy(databox);
  nz = DataboxNz(databox);
  n = nx * ny * nz;

  /*---------------------------------------------------
   * Compute min, max, mean
   *---------------------------------------------------*/

  dp = DataboxCoeffs(databox);

  *min = *dp;
  *max = *dp;
  *mean = 0.0;
  *sum = 0.0;

  for (k = 0; k < nz; k++)
  {
    for (j = 0; j < ny; j++)
    {
      for (i = 0; i < nx; i++)
      {
        if (*dp < *min)
          *min = *dp;

        if (*dp > *max)
          *max = *dp;

        *sum += *dp;

        dp++;
      }
    }
  }

  *mean = *sum / n;

  /*---------------------------------------------------
   * Compute variance, standard deviation
   *---------------------------------------------------*/

  dp = DataboxCoeffs(databox);

  *variance = 0.0;

  for (k = 0; k < nz; k++)
  {
    for (j = 0; j < ny; j++)
    {
      for (i = 0; i < nx; i++)
      {
        dtmp = (*dp - *mean);
        *variance += dtmp * dtmp;

        dp++;
      }
    }
  }

  *variance /= n;
  *stdev = sqrt(*variance);
}
