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
* Header info for the Background structure
*
*****************************************************************************/

#ifndef _BACKGROUND_HEADER
#define _BACKGROUND_HEADER

/*--------------------------------------------------------------------------
 * Background:
 *   Structure describing a uniform grid in real-space.
 *--------------------------------------------------------------------------*/

typedef struct {
  double X, Y, Z;         /* Anchor point in real-space */
  double DX, DY, DZ;      /* Spacing in each coordinate direction */

  /* Bounding information (essentially a level 0 subgrid) */
  int IX, IY, IZ;
  int NX, NY, NZ;
} Background;

/*--------------------------------------------------------------------------
 * Accessor macros: Background
 *--------------------------------------------------------------------------*/

#define BackgroundX(bg)   ((bg)->X)
#define BackgroundY(bg)   ((bg)->Y)
#define BackgroundZ(bg)   ((bg)->Z)

#define BackgroundDX(bg)  ((bg)->DX)
#define BackgroundDY(bg)  ((bg)->DY)
#define BackgroundDZ(bg)  ((bg)->DZ)

#define BackgroundIX(bg)  ((bg)->IX)
#define BackgroundIY(bg)  ((bg)->IY)
#define BackgroundIZ(bg)  ((bg)->IZ)

#define BackgroundNX(bg)  ((bg)->NX)
#define BackgroundNY(bg)  ((bg)->NY)
#define BackgroundNZ(bg)  ((bg)->NZ)

#define BackgroundXLower(bg) \
        (BackgroundX(bg) + BackgroundIX(bg) * BackgroundDX(bg))
#define BackgroundYLower(bg) \
        (BackgroundY(bg) + BackgroundIY(bg) * BackgroundDY(bg))
#define BackgroundZLower(bg) \
        (BackgroundZ(bg) + BackgroundIZ(bg) * BackgroundDZ(bg))

#define BackgroundXUpper(bg) \
        (BackgroundX(bg) + (BackgroundIX(bg) + BackgroundNX(bg)) * BackgroundDX(bg))
#define BackgroundYUpper(bg) \
        (BackgroundY(bg) + (BackgroundIY(bg) + BackgroundNY(bg)) * BackgroundDY(bg))
#define BackgroundZUpper(bg) \
        (BackgroundZ(bg) + (BackgroundIZ(bg) + BackgroundNZ(bg)) * BackgroundDZ(bg))


#endif
