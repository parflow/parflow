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
* Routines for generating line processes for the turning bands method.
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * LineProc
 *--------------------------------------------------------------------------*/

void  LineProc(
               double *Z,
               double  phi,
               double  theta,
               double  dzeta,
               int     izeta,
               int     nzeta,
               double  Kmax,
               double  dK)
{
  double pi = acos(-1.0);

  int M;
  double dk;
  double phij, kj, kj_fudge, S1, dW, zeta;

  int i, j;

  (void)phi;
  (void)theta;

  /* compute M and dk */
  M = (int)(Kmax / dK);
  dk = dK;

  /* initialize Z */
  for (i = 0; i < nzeta; i++)
    Z[i] = 0.0;

  /*-----------------------------------------------------------------------
   * compute Z
   *-----------------------------------------------------------------------*/

  for (j = 0; j < M; j++)
  {
    phij = 2.0 * pi * Rand();
    kj = dk * (j + 0.5);

    kj_fudge = kj + (dk / 20.0) * (2.0 * (Rand() - 0.5));

    S1 = (2.0 * kj * kj) / (pi * (kj * kj + 1.0) * (kj * kj + 1.0));

    dW = 2.0 * sqrt(S1 * dk);

    for (i = 0; i < nzeta; i++)
    {
      zeta = (izeta + i) * dzeta;
      Z[i] += dW * cos(kj_fudge * zeta + phij);
    }
  }
}
