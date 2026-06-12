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

#ifndef CHEM_UTILITIES_H
#define CHEM_UTILITIES_H

/* chem_utilities.c  */
double InterpolateTimeCycle(double total_cycle_length, double subcycle_dt);

void TransportSaturation(Vector *sat_transport_start, Vector *delta_sat, Vector *old_sat, Vector *new_sat);

void SelectReactTransTimeStep(double max_velocity, double CFL,
                              double PF_dt, double *advect_react_dt,
                              int *num_rt_iterations);

int  SubgridNumCells(Grid *grid);

#ifdef HAVE_ALQUIMIA
void CutTimeStepandSolveSingleCell(AlquimiaInterface chem, AlquimiaState *chem_state, AlquimiaProperties *chem_properties, void *chem_engine, AlquimiaAuxiliaryData *chem_aux_data, AlquimiaEngineStatus *chem_status, double original_dt);

//void CutTimeStepandSolveRecursively(AlquimiaDataPF * alquimia_data, double original_dt, int level, int chem_index);

void WriteChemChkpt(Grid *grid, AlquimiaSizes *chem_sizes, AlquimiaState *chem_state, AlquimiaAuxiliaryData *chem_aux_data, AlquimiaProperties *chem_properties, char *file_prefix, char *file_suffix);

void ReadChemChkpt(Grid *grid, AlquimiaSizes *chem_sizes, AlquimiaState *chem_state, AlquimiaAuxiliaryData *chem_aux_data, AlquimiaProperties *chem_properties, char *filename);
#endif

#endif

