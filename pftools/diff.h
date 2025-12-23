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
/****************************************************************************
 * Header file for `diff.c'
 *
 * (C) 1993 Regents of the University of California.
 *
 *----------------------------------------------------------------------------
 * $Revision: 1.7 $
 *
 *----------------------------------------------------------------------------
 *
 *****************************************************************************/

#ifndef DIFF_HEADER
#define DIFF_HEADER

#include "databox.h"

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* diff.c */
void SigDiff(Databox *v1, Databox *v2, int m, double absolute_zero, FILE *fp);
double DiffElt(Databox *v1, Databox *v2, int i, int j, int k, int m, double absolute_zero);
void MSigDiff(Tcl_Interp *interp, Databox *v1, Databox *v2, int m, double absolute_zero, Tcl_Obj *result);

#ifdef __cplusplus
}
#endif

#endif

