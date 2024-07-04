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
#include <stdlib.h>
#include "enlargebox.h"

Databox       *EnlargeBox(Databox *inbox,
                          int new_nx, int new_ny, int new_nz)
{
  Databox       *newbox;

  int nx, ny, nz;
  double x, y, z;
  double dx, dy, dz;

  int i, j, k;

  nx = DataboxNx(inbox);
  ny = DataboxNy(inbox);
  nz = DataboxNz(inbox);

  x = DataboxX(inbox);
  y = DataboxY(inbox);
  z = DataboxZ(inbox);

  dx = DataboxDx(inbox);
  dy = DataboxDy(inbox);
  dz = DataboxDz(inbox);

  if (new_nx < nx)
  {
    printf(" Error: new_nx must be greater than or equal to old size\n");
  }

  if (new_ny < ny)
  {
    printf(" Error: new_ny must be greater than or equal to old size\n");
  }

  if (new_nz < nz)
  {
    printf(" Error: new_nz must be greater than or equal to old size\n");
  }

  if ((newbox = NewDatabox(new_nx, new_ny, new_nz,
                           x, y, z,
                           dx, dy, dz)) == NULL)
    return((Databox*)NULL);

  /* First just copy the old values into the new box */
  for (k = 0; k < nz; k++)
  {
    for (j = 0; j < ny; j++)
    {
      for (i = 0; i < nx; i++)
      {
        *DataboxCoeff(newbox, i, j, k) = *DataboxCoeff(inbox, i, j, k);
      }
    }
  }

  /* Copy the z plane  from the existing nz'th plane */
  for (k = nz; k < new_nz; k++)
    for (j = 0; j < ny; j++)
      for (i = 0; i < nx; i++)
        *DataboxCoeff(newbox, i, j, k) = *DataboxCoeff(newbox, i, j, nz - 1);

  /* Copy the y plane  from the existing ny'th plane */
  for (j = ny; j < new_ny; j++)
    for (k = 0; k < new_nz; k++)
      for (i = 0; i < nx; i++)
        *DataboxCoeff(newbox, i, j, k) = *DataboxCoeff(newbox, i, ny - 1, k);

  /* Copy the i planes from the existing nx'th plane */
  for (i = nx; i < new_nx; i++)
    for (j = 0; j < new_ny; j++)
      for (k = 0; k < new_nz; k++)
        *DataboxCoeff(newbox, i, j, k) = *DataboxCoeff(newbox, nx - 1, i, k);

  return newbox;
}
