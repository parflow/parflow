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
#ifndef TOP_WATER_BALANCE
#define TOP_WATER_BALANCE

#include "databox.h"

#ifdef __cplusplus
extern "C" {
#endif

/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

void ComputeSurfaceStorage(
                           Databox *top,
                           Databox *pressure,
                           Databox *surface_storage);

void ComputeSubsurfaceStorage(
                              Databox *mask,
                              Databox *porosity,
                              Databox *pressure,
                              Databox *saturation,
                              Databox *specific_storage,
                              Databox *subsurface_storage);

void ComputeGWStorage(
                      Databox *mask,
                      Databox *porosity,
                      Databox *pressure,
                      Databox *saturation,
                      Databox *specific_storage,
                      Databox *gw_storage);

void ComputeSurfaceRunoff(
                          Databox *top,
                          Databox *slope_x,
                          Databox *slope_y,
                          Databox *mannings,
                          Databox *pressure,
                          Databox *surface_runoff);

#ifdef __cplusplus
}
#endif

#endif

