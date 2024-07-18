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

#ifndef _PARFLOW_GROUNDWATERFLOW_EVAL
#define _PARFLOW_GROUNDWATERFLOW_EVAL

#include "parflow.h"

typedef void GroundwaterFlowPublicXtra;

typedef struct {

  double*** SpecificYield;   // [ipatch][isubgrid][icell] - Updated every step
  double*** AquiferDepth;    // [ipatch][isubgrid][icell] - Updated every step
  double*** AquiferRecharge; // [ipatch][isubgrid][icell] - Updated every step
  int num_patches;  // ipatch varies between [0, num_patches[
  int subgrid_size; // isubgrid varies between [0, subgrid_size[

} GroundwaterFlowInstanceXtra;

void NewGroundwaterFlowParameters(void** Sy, void** Ad, void** Ar, 
    char* patch_name, int global_cycle, int interval_division);

void FreeGroundwaterFlowParameters(
    void** Sy, void** Ad, void** Ar, int interval_division);

void FreeGroundwaterFlowParameter(void* parameter);

void SetGroundwaterFlowParameter(
    double* values, void* parameter, BCStruct* bc_struct, int ipatch, int is);



typedef void (*GroundwaterFlowEvalInvoke)(
    double* q_groundwater, BCStruct* bc_struct, Subgrid* subgrid, 
    Subvector* p_sub, double* new_pressure, double* old_pressure, 
    double dt, double* perm_x, double* perm_y, double* perm_z, 
    double* z_mult, int ipatch, int isubgrid);

void GroundwaterFlowEval(
    double* q_groundwater, BCStruct* bc_struct, Subgrid* subgrid, 
    Subvector* p_sub, double* new_pressure, double* old_pressure, 
    double dt, double* perm_x, double* perm_y, double* perm_z, 
    double* z_mult, int ipatch, int isubgrid);

PFModule* GroundwaterFlowEvalInitInstanceXtra(
    int num_patches, int subgrid_size);

void GroundwaterFlowEvalFreeInstanceXtra();

PFModule* GroundwaterFlowEvalNewPublicXtra();

void GroundwaterFlowEvalFreePublicXtra();

int GroundwaterFlowEvalSizeOfTempData();

#endif // _PARFLOW_GROUNDWATERFLOW_EVAL