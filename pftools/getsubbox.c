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
*  CompSub_Box
*
* (C) 1993 Regents of the University of California.
*
*-----------------------------------------------------------------------------
* $Revision: 1.1 $
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include <stdlib.h>

#include "getsubbox.h"


/*-----------------------------------------------------------------------
 * Compute cell-centered velocities from conductivity and pressure head
 *-----------------------------------------------------------------------*/

Databox       *CompSubBox(
                          Databox *fun,
                          int      il,
                          int      jl,
                          int      kl,
                          int      iu,
                          int      ju,
                          int      ku)
{
  Databox       *sub_fun;

  int nx, ny;
  double x, y, z;
  double dx, dy, dz;

  int nx_sub, ny_sub, nz_sub;
  double x_sub, y_sub, z_sub;

  double         *funp;
  double         *sub_fun_pt;

  int m, m_old;
  int ii, jj, kk;
  double x_max, y_max, z_max;


  nx = DataboxNx(fun);
  ny = DataboxNy(fun);

  x = DataboxX(fun);
  y = DataboxY(fun);
  z = DataboxZ(fun);

  dx = DataboxDx(fun);
  dy = DataboxDy(fun);
  dz = DataboxDz(fun);

  nx_sub = iu - il;
  ny_sub = ju - jl;
  nz_sub = ku - kl;

  x_sub = x + dx * il;
  y_sub = y + dy * jl;
  z_sub = z + dz * kl;

  x_max = x_sub + (iu - il) * dx;
  y_max = y_sub + (ju - jl) * dy;
  z_max = z_sub + (ku - kl) * dz;

  if (il >= iu)
  {
    printf(" Error: il must be less than iu\n");
  }

  if (jl >= ju)
  {
    printf(" Error: jl must be less than ju\n");
  }

  if (kl >= ku)
  {
    printf(" Error: kl must be less than ku\n");
  }


  if ((sub_fun = NewDatabox(nx_sub, ny_sub, nz_sub,
                            x_sub, y_sub, z_sub, dx, dy, dz)) == NULL)
    return((Databox*)NULL);

  funp = DataboxCoeffs(fun);
  sub_fun_pt = DataboxCoeffs(sub_fun);

  m = 0;

  for (kk = kl; kk < ku; kk++)
  {
    for (jj = jl; jj < ju; jj++)
    {
      for (ii = il; ii < iu; ii++)
      {
        m_old = ii + nx * jj + nx * ny * kk;
        sub_fun_pt[m] = funp[m_old];
        m++;
      }
    }
  }

  printf("  Sub Box Boundaries\n");
  printf("     Xmin = %f", x_sub);
  printf("   Xmax = %f\n", x_max);
  printf("     Ymin = %f", y_sub);
  printf("   Ymax = %f\n", y_max);
  printf("     Zmin = %f", z_sub);
  printf("   Zmax = %f\n", z_max);

  return sub_fun;
}
