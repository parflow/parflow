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
* Header file for `velocity.c'
*
* (C) 1993 Regents of the University of California.
*
* see info_header.h for complete information
*
*                               History
*-----------------------------------------------------------------------------
* $Log: velocity.h,v $
* Revision 1.6  1997/03/07 00:31:01  shumaker
* Added two new pftools, pfgetsubbox and pfbfcvel
*
* Revision 1.5  1995/12/21  00:56:38  steve
* Added copyright
*
* Revision 1.4  1995/08/17  21:48:59  falgout
* Added a vertex velocity capability.
*
* Revision 1.3  1993/09/01  01:29:54  falgout
* Added ability to compute velocities as well as velocity magnitudes.
*
* Revision 1.2  1993/08/20  21:22:32  falgout
* Added PrintParflowB, modified velocity computation, added HHead and PHead.
*
* Revision 1.1  1993/04/13  18:00:21  falgout
* Initial revision
*
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#ifndef VELOCITY_HEADER
#define VELOCITY_HEADER

#include "databox.h"

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

/* velocity.c */
Databox * *CompBFCVel(Databox *k, Databox *h);
Databox **CompCellVel(Databox *k, Databox *h);
Databox **CompVertVel(Databox *k, Databox *h);
Databox *CompVMag(Databox *vx, Databox *vy, Databox *vz);

#ifdef __cplusplus
}
#endif

#endif

