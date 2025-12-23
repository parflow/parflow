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
* HHead, PHead
*
* (C) 1993 Regents of the University of California.
*
* see info_header.h for complete information
*
*                               History
*-----------------------------------------------------------------------------
* $Log: head.c,v $
* Revision 1.5  1996/08/10 06:35:12  mccombjr
*** empty log message ***
***
* Revision 1.4  1996/04/25  01:05:50  falgout
* Added general BC capability.
*
* Revision 1.3  1995/12/21  00:56:38  steve
* Added copyright
*
* Revision 1.2  1995/06/27  21:36:13  falgout
* Added (X, Y, Z) coordinates to databox structure.
*
* Revision 1.1  1993/08/20  21:22:32  falgout
* Initial revision
*
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "head.h"


/*-----------------------------------------------------------------------
 * compute the hydraulic head from the pressure head
 *-----------------------------------------------------------------------*/

Databox        *HHead(
                      Databox *h,
                      GridType grid_type)
{
  Databox        *v;

  int nx, ny, nz;
  double x, y, z;
  double dx, dy, dz;

  double         *hp, *vp;

  int ji, k;

  double zz, dz2 = 0.0;


  nx = DataboxNx(h);
  ny = DataboxNy(h);
  nz = DataboxNz(h);

  x = DataboxX(h);
  y = DataboxY(h);
  z = DataboxZ(h);

  dx = DataboxDx(h);
  dy = DataboxDy(h);
  dz = DataboxDz(h);

  switch (grid_type)
  {
    case vertex: /* vertex centered */
      dz2 = 0.0;
      break;

    case cell:  /* cell centered */
      dz2 = dz / 2;
      break;
  }

  if ((v = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)) == NULL)
    return((Databox*)NULL);

  hp = DataboxCoeffs(h);
  vp = DataboxCoeffs(v);

  for (k = 0; k < nz; k++)
  {
    zz = z + ((double)k) * dz + dz2;
    for (ji = 0; ji < ny * nx; ji++)
      *(vp++) = *(hp++) + zz;
  }

  return v;
}


/*-----------------------------------------------------------------------
 * compute the pressure from the hydraulic head
 *-----------------------------------------------------------------------*/

Databox        *PHead(
                      Databox *h,
                      GridType grid_type)
{
  Databox        *v;

  int nx, ny, nz;
  double x, y, z;
  double dx, dy, dz;

  double         *hp, *vp;

  int ji, k;

  double zz, dz2 = 0.0;


  nx = DataboxNx(h);
  ny = DataboxNy(h);
  nz = DataboxNz(h);

  x = DataboxX(h);
  y = DataboxY(h);
  z = DataboxZ(h);

  dx = DataboxDx(h);
  dy = DataboxDy(h);
  dz = DataboxDz(h);

  switch (grid_type)
  {
    case vertex: /* vertex centered */
      dz2 = 0.0;
      break;

    case cell:  /* cell centered */
      dz2 = dz / 2;
      break;
  }

  if ((v = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)) == NULL)
    return((Databox*)NULL);

  hp = DataboxCoeffs(h);
  vp = DataboxCoeffs(v);

  for (k = 0; k < nz; k++)
  {
    zz = z + ((double)k) * dz + dz2;
    for (ji = 0; ji < ny * nx; ji++)
      *(vp++) = *(hp++) - zz;
  }

  return v;
}


