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
* CompVel, CompVMag
*
* (C) 1993 Regents of the University of California.
*
*-----------------------------------------------------------------------------
* $Revision: 1.13 $
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include <stdlib.h>

#include "velocity.h"


#if 0
#define Mean(a, b) (0.5*((a) + (b)))
#define Mean(a, b) (sqrt((a) * (b)))
#endif
#define Mean(a, b) (2*((a) * (b)) / ((a) + (b)))


/*-----------------------------------------------------------------------
 * Compute cell-centered velocities from conductivity and pressure head
 *-----------------------------------------------------------------------*/

Databox       **CompCellVel(
                            Databox *k,
                            Databox *h)
{
  Databox       **v;

  int nx, ny, nz;
  double x, y, z;
  double dx, dy, dz;

  double         *kp, *hp;
  double         *vxp, *vyp, *vzp;

  int m1, m2, m3, m4, m5, m6, m7, m8;
  int ii, jj, kk;


  nx = DataboxNx(k);
  ny = DataboxNy(k);
  nz = DataboxNz(k);

  x = DataboxX(k);
  y = DataboxY(k);
  z = DataboxZ(k);

  dx = DataboxDx(k);
  dy = DataboxDy(k);
  dz = DataboxDz(k);

#if 0      /* ADD LATER */
  if ((dx != DataboxDx(h)) ||
      (dy != DataboxDy(h)) ||
      (dz != DataboxDz(h)))
  {
    Error("Spacings are not compatible\n");
    return NULL;
  }
#endif


  if ((v = (Databox**)calloc(3, sizeof(Databox *))) == NULL)
    return((Databox**)NULL);

  if ((v[0] = NewDatabox((nx - 1), (ny - 1), (nz - 1), x, y, z, dx, dy, dz)) == NULL)
  {
    free(v);
    return((Databox**)NULL);
  }

  if ((v[1] = NewDatabox((nx - 1), (ny - 1), (nz - 1), x, y, z, dx, dy, dz)) == NULL)
  {
    FreeDatabox(v[0]);
    free(v);
    return((Databox**)NULL);
  }

  if ((v[2] = NewDatabox((nx - 1), (ny - 1), (nz - 1), x, y, z, dx, dy, dz)) == NULL)
  {
    FreeDatabox(v[0]);
    FreeDatabox(v[1]);
    free(v);
    return((Databox**)NULL);
  }

  kp = DataboxCoeffs(k);
  hp = DataboxCoeffs(h);
  vxp = DataboxCoeffs(v[0]);
  vyp = DataboxCoeffs(v[1]);
  vzp = DataboxCoeffs(v[2]);

  m1 = 0;
  m2 = m1 + 1;
  m3 = m1 + nx;
  m4 = m3 + 1;
  m5 = m1 + ny * nx;
  m6 = m5 + 1;
  m7 = m5 + nx;
  m8 = m7 + 1;

  for (kk = 0; kk < (nz - 1); kk++)
  {
    for (jj = 0; jj < (ny - 1); jj++)
    {
      for (ii = 0; ii < (nx - 1); ii++)
      {
        *vxp = -(Mean(kp[m1], kp[m2]) * (hp[m2] - hp[m1]) +
                 Mean(kp[m3], kp[m4]) * (hp[m4] - hp[m3]) +
                 Mean(kp[m5], kp[m6]) * (hp[m6] - hp[m5]) +
                 Mean(kp[m7], kp[m8]) * (hp[m8] - hp[m7])) / (4.0 * dx);
        *vyp = -(Mean(kp[m1], kp[m3]) * (hp[m3] - hp[m1]) +
                 Mean(kp[m2], kp[m4]) * (hp[m4] - hp[m2]) +
                 Mean(kp[m5], kp[m7]) * (hp[m7] - hp[m5]) +
                 Mean(kp[m6], kp[m8]) * (hp[m8] - hp[m6])) / (4.0 * dy);
        *vzp = -(Mean(kp[m1], kp[m5]) * (hp[m5] - hp[m1] + dz) +
                 Mean(kp[m3], kp[m7]) * (hp[m7] - hp[m3] + dz) +
                 Mean(kp[m2], kp[m6]) * (hp[m6] - hp[m2] + dz) +
                 Mean(kp[m4], kp[m8]) * (hp[m8] - hp[m4] + dz)) / (4.0 * dz);

        vxp++;
        vyp++;
        vzp++;

        kp++;
        hp++;
      }
      kp++;
      hp++;
    }
    kp += nx;
    hp += nx;
  }

  return (Databox**)v;
}


/*-----------------------------------------------------------------------
 * Compute vertex-centered velocities from conductivity and pressure head
 *-----------------------------------------------------------------------*/

Databox       **CompVertVel(
                            Databox *k,
                            Databox *h)
{
  Databox       **v;

  int nx, ny, nz;
  double x, y, z;
  double dx, dy, dz;

  double         *kp, *hp;
  double         *vxp, *vyp, *vzp;

  int m, sx, sy, sz;
  int ii, jj, kk;


  nx = DataboxNx(k);
  ny = DataboxNy(k);
  nz = DataboxNz(k);

  x = DataboxX(k);
  y = DataboxY(k);
  z = DataboxZ(k);

  dx = DataboxDx(k);
  dy = DataboxDy(k);
  dz = DataboxDz(k);

#if 0      /* ADD LATER */
  if ((dx != DataboxDx(h)) ||
      (dy != DataboxDy(h)) ||
      (dz != DataboxDz(h)))
  {
    Error("Spacings are not compatible\n");
    return NULL;
  }
#endif

  if ((v = (Databox**)calloc(3, sizeof(Databox *))) == NULL)
    return((Databox**)NULL);

  if ((v[0] = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)) == NULL)
  {
    free(v);
    return((Databox**)NULL);
  }

  if ((v[1] = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)) == NULL)
  {
    FreeDatabox(v[0]);
    free(v);
    return((Databox**)NULL);
  }

  if ((v[2] = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)) == NULL)
  {
    FreeDatabox(v[0]);
    FreeDatabox(v[1]);
    free(v);
    return((Databox**)NULL);
  }


  kp = DataboxCoeffs(k);
  hp = DataboxCoeffs(h);
  vxp = DataboxCoeffs(v[0]);
  vyp = DataboxCoeffs(v[1]);
  vzp = DataboxCoeffs(v[2]);

  m = 0;
  sx = 1;
  sy = nx;
  sz = ny * nx;

  for (kk = 1; kk < (nz - 1); kk++)
  {
    for (jj = 1; jj < (ny - 1); jj++)
    {
      m = kk * sz + jj * sy + sx;

      for (ii = 1; ii < (nx - 1); ii++)
      {
        vxp[m] = -(Mean(kp[m], kp[m + sx]) * (hp[m + sx] - hp[m])) / dx;
        vyp[m] = -(Mean(kp[m], kp[m + sy]) * (hp[m + sy] - hp[m])) / dy;
        vzp[m] = -(Mean(kp[m], kp[m + sz]) * (hp[m + sz] - hp[m] + dz)) / dz;

        m++;
      }
    }
  }

  return (Databox**)v;
}

/*-----------------------------------------------------------------------
 * Compute block face-centered velocities from conductivity and pressure head
 *-----------------------------------------------------------------------*/

Databox       **CompBFCVel(
                           Databox *k,
                           Databox *h)
{
  Databox       **v;

  int nx, ny, nz;
  int nx1, ny1, nz1;
  double x, y, z;
  double dx, dy, dz;

  double         *kp, *hp;
  double         *vxp, *vyp, *vzp;

  int m, m_bfc, sx, sy, sz;
  int ii, jj, kk;


  nx = DataboxNx(k);
  ny = DataboxNy(k);
  nz = DataboxNz(k);

  x = DataboxX(k);
  y = DataboxY(k);
  z = DataboxZ(k);

  dx = DataboxDx(k);
  dy = DataboxDy(k);
  dz = DataboxDz(k);

  nx1 = nx - 1;
  ny1 = ny - 1;
  nz1 = nz - 1;

#if 0      /* ADD LATER */
  if ((dx != DataboxDx(h)) ||
      (dy != DataboxDy(h)) ||
      (dz != DataboxDz(h)))
  {
    Error("Spacings are not compatible\n");
    return NULL;
  }
#endif

  if ((v = (Databox**)calloc(3, sizeof(Databox *))) == NULL)
    return((Databox**)NULL);

  if ((v[0] = NewDatabox(nx1, ny, nz, x + 0.5 * dx, y, z, dx, dy, dz)) == NULL)
  {
    free(v);
    return((Databox**)NULL);
  }

  if ((v[1] = NewDatabox(nx, ny1, nz, x, y + 0.5 * dy, z, dx, dy, dz)) == NULL)
  {
    FreeDatabox(v[0]);
    free(v);
    return((Databox**)NULL);
  }

  if ((v[2] = NewDatabox(nx, ny, nz1, x, y, z + 0.5 * dz, dx, dy, dz)) == NULL)
  {
    FreeDatabox(v[0]);
    FreeDatabox(v[1]);
    free(v);
    return((Databox**)NULL);
  }


  kp = DataboxCoeffs(k);
  hp = DataboxCoeffs(h);
  vxp = DataboxCoeffs(v[0]);
  vyp = DataboxCoeffs(v[1]);
  vzp = DataboxCoeffs(v[2]);

  sx = 1;
  sy = nx;
  sz = ny * nx;

  for (kk = 0; kk < (nz - 1); kk++)
  {
    for (jj = 0; jj < (ny - 1); jj++)
    {
      for (ii = 0; ii < (nx - 2); ii++)
      {
        m = ii + nx * jj + nx * ny * kk;
        m_bfc = ii + nx1 * jj + nx1 * ny * kk;
        vxp[m_bfc] = -(Mean(kp[m], kp[m + sx]) * (hp[m + sx] - hp[m])) / dx;
      }
    }
  }

  for (kk = 0; kk < (nz - 1); kk++)
  {
    for (jj = 0; jj < (ny - 2); jj++)
    {
      for (ii = 0; ii < (nx - 1); ii++)
      {
        m = ii + nx * jj + nx * ny * kk;
        m_bfc = ii + nx * jj + nx * ny1 * kk;
        vyp[m_bfc] = -(Mean(kp[m], kp[m + sy]) * (hp[m + sy] - hp[m])) / dy;
      }
    }
  }
  for (kk = 0; kk < (nz - 2); kk++)
  {
    for (jj = 0; jj < (ny - 1); jj++)
    {
      for (ii = 0; ii < (nx - 1); ii++)
      {
        m = ii + nx * jj + nx * ny * kk;
        m_bfc = ii + nx * jj + nx * ny * kk;
        vzp[m_bfc] = -(Mean(kp[m], kp[m + sz]) * (hp[m + sz] - hp[m] + dz)) / dz;
      }
    }
  }

  return (Databox**)v;
}


/*-----------------------------------------------------------------------
 * Compute velocity magnitude
 *-----------------------------------------------------------------------*/

Databox        *CompVMag(
                         Databox *vx,
                         Databox *vy,
                         Databox *vz)
{
  Databox        *v;

  int nx, ny, nz;
  double x, y, z;
  double dx, dy, dz;

  double         *vxp, *vyp, *vzp, *vp;

  int m, ii;


  nx = DataboxNx(vx);
  ny = DataboxNy(vx);
  nz = DataboxNz(vx);

  x = DataboxX(vx);
  y = DataboxY(vx);
  z = DataboxZ(vx);

  dx = DataboxDx(vx);
  dy = DataboxDy(vx);
  dz = DataboxDz(vx);

  if ((v = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)) == NULL)
    return((Databox*)NULL);

  vxp = DataboxCoeffs(vx);
  vyp = DataboxCoeffs(vy);
  vzp = DataboxCoeffs(vz);
  vp = DataboxCoeffs(v);

  m = 0;
  for (ii = 0; ii < (nx * ny * nz); ii++)
  {
    vp[m] = sqrt(vxp[m] * vxp[m] + vyp[m] * vyp[m] + vzp[m] * vzp[m]);

    m++;
  }

  return v;
}
