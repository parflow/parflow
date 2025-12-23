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
* SeedRand, Rand
*
* Routines for generating random numbers.
*
*****************************************************************************/

#include "amps.h"
#include <math.h>


/*--------------------------------------------------------------------------
 * Static variables
 *--------------------------------------------------------------------------*/

amps_ThreadLocalDcl(static int, Seed);

#define M 1048576
#define L 1027

/*--------------------------------------------------------------------------
 * SeedRand:
 *   The seed must always be positive.
 *
 *   Note: the internal seed must be positive and odd, so it is set
 *   to (2*input_seed - 1);
 *--------------------------------------------------------------------------*/

void  SeedRand(
               int seed)
{
  amps_ThreadLocal(Seed) = (2 * seed - 1) % M;
}

/*--------------------------------------------------------------------------
 * Rand
 *--------------------------------------------------------------------------*/

double  Rand()
{
  amps_ThreadLocal(Seed) = (L * amps_ThreadLocal(Seed)) % M;

  return(((double)amps_ThreadLocal(Seed)) / ((double)M));
}
