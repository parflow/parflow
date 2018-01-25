/*BHEADER**********************************************************************
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

#ifndef _FLOWVR_HEADER
#define _FLOWVR_HEADER

#include "parflow.h"

/**
 * If FLOWVR_ACTIVE is not 0 parflow will act as FlowVR module.
 */
extern int FLOWVR_ACTIVE;

/**
 * Sets FLOWVR_ACTIVE depending on compile options in the input database.
 * If FLOWVR_ACTIVE inits the FlowVR module
 */
void NewFlowVR(void);

#ifdef HAVE_FLOWVR

#include <fca.h>
#include <messages.h>

#ifdef __DEBUG
#define D(x ...) printf("=======%d:", amps_Rank(amps_CommWorld)); printf(x); \
  printf("\n"); printf(" %s:%d\n", __FILE__, __LINE__)
#else
#define D(...)
#endif

typedef struct {
  char * filename;
  double * time;
  Vector * pressure_out;
  Vector * porosity_out;
  Vector * saturation_out;
  Grid * grid;
  ProblemData * problem_data;
} SimulationSnapshot;

/**
 * GetSimulationSnapshot is used to comfortably fill a SimulationSnapshot data structure
 * in the AdvanceRichards() or the SetupRichards() context in richards_solver.c
 */
#define GetSimulationSnapshot \
  (SimulationSnapshot){ \
    filename, \
    &t, \
    instance_xtra->pressure, \
    NULL, \
    instance_xtra->saturation, \
    instance_xtra->grid, \
    problem_data \
  }


// PFModule that is used here: solver_richards.
// TODO: documentation, also in other c files

void FlowVRinitTranslation(SimulationSnapshot *snapshot);

// do all the in message handlinge
int FlowVRInteract(SimulationSnapshot *sshot);


int FlowVRFulFillContracts(int timestep, SimulationSnapshot const * const sshot);

void FlowVRServeFinalState(SimulationSnapshot *snapshot);

void FreeFlowVR();

extern inline void FlowVRAbort();


#endif
#endif
