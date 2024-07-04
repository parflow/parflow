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
* FILE: nodiag_scale.c
*
* WRITTEN BY:   Bill Bosl
*               phone: (510) 423-2873
*               e-mail: wjbosl@llnl.gov
*
* FUNCTIONS IN THIS FILE:
* NoDiagScale, InitNoDiagScale, FreeNoDiagScale, NewNoDiagScale
*
* DESCRIPTION:
* This module does nothing to the input matrix. It's primary
* purpose is to give a "do nothing" alternative for the
* diagonal scaling options.
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * NoDiagScale
 *--------------------------------------------------------------------------*/

void NoDiagScale(Vector *x, Matrix *A, Vector *b, int flag)
{
  (void)x;
  (void)A;
  (void)b;
  (void)flag;
  return;
}

/*--------------------------------------------------------------------------
 * NoDiagScaleInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *NoDiagScaleInitInstanceXtra(Grid *grid)
{
  (void)grid;
  return ThisPFModule;
}


/*--------------------------------------------------------------------------
 * NoDiagScaleFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  NoDiagScaleFreeInstanceXtra()
{
  return;
}


/*--------------------------------------------------------------------------
 * NoDiagScaleNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *NoDiagScaleNewPublicXtra(char *name)
{
  (void)name;
  return ThisPFModule;
}


/*--------------------------------------------------------------------------
 * NoDiagScaleFreePublicXtra
 *--------------------------------------------------------------------------*/

void  NoDiagScaleFreePublicXtra()
{
  return;
}

/*--------------------------------------------------------------------------
 * NoDiagScaleSizeOfTempData
 *--------------------------------------------------------------------------*/

int  NoDiagScaleSizeOfTempData()
{
  return 0;
}
