/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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

#ifndef TRANSPORT_UTILITIES_H
#define TRANSPORT_UTILITIES_H

/* transport_utilities.c -- transport helpers for the react_trans coupling;
 * compiled unconditionally (no Alquimia dependency). */

double InterpolateTimeCycle(double total_cycle_length, double subcycle_dt);

void TransportSaturation(Vector *sat_transport_start, Vector *delta_sat, Vector *old_sat, Vector *new_sat);

void SelectReactTransTimeStep(double max_velocity, double CFL,
                              double PF_dt, double *advect_react_dt,
                              int *num_rt_iterations);

void BCConcenPatchExtent(Subgrid *subgrid, int *ix, int *iy, int *iz, int *nx, int *ny, int *nz, int ipatch);

void BCConcenCopyPatch(Problem *problem, Grid *grid, Vector **concentrations, int ipatch);

void BCConcenCopyAdjacent(Problem *problem, Grid *grid, Vector **concentrations);

int BoundaryCell(int ipatch, int i, int j, int k);

#endif
