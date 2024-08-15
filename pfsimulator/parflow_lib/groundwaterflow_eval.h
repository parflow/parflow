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

double ParameterAt(void *parameter, int ival);

void* NewGroundwaterFlowParameter(char *patch_name, char *parameter_name);

void FreeGroundwaterFlowParameter(void *parameter);


typedef void (*GroundwaterFlowEvalInvoke)(
    double *q_groundwater, BCStruct *bc_struct, Subgrid *subgrid, 
    Subvector *p_sub, double *old_pressure, double dt, 
    double *perm_x, double *perm_y, double *z_mult, 
    int ipatch, int isubgrid, ProblemData *problem_data);

void GroundwaterFlowEval(
    double *q_groundwater, BCStruct *bc_struct, Subgrid *subgrid, 
    Subvector *p_sub, double *old_pressure, double dt, 
    double *perm_x, double *perm_y, double *z_mult, 
    int ipatch, int isubgrid, ProblemData *problem_data);

PFModule* GroundwaterFlowEvalInitInstanceXtra();

void GroundwaterFlowEvalFreeInstanceXtra();

PFModule* GroundwaterFlowEvalNewPublicXtra();

void GroundwaterFlowEvalFreePublicXtra();

int GroundwaterFlowEvalSizeOfTempData();

#endif // _PARFLOW_GROUNDWATERFLOW_EVAL