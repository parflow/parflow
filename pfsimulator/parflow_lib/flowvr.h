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
#include <messages.h>


extern int FLOWVR_ACTIVE;

void NewFlowVR(void);

#ifdef HAVE_FLOWVR

#include <fca/fca.h>

#ifdef __DEBUG  //TODO: add filename and line number
#define D(x ...) printf("=======%d:", amps_Rank(amps_CommWorld)); printf(x); printf("\n") //printf(" %s:%d\n",  __FILE__, __LINE__)
#else
#define D(...)
#endif

typedef struct {
  char * filename;
  double * time;
  Vector * pressure_out;
  Vector * porosity_out;
  Vector * saturation_out;
} SimulationSnapshot;

#define GetSimulationSnapshot \
  (SimulationSnapshot){ \
    filename, \
    &t, \
    instance_xtra->pressure, \
    NULL, \
    instance_xtra->saturation, \
  }

extern fca_module moduleParflow;

// PFModule that is used here: solver_richards.
// TODO: documentation, also in other c files

// do all the in message handlinge
int FlowVRInteract(SimulationSnapshot *sshot);


// needed for writer:
void DumpRichardsToFlowVR(SimulationSnapshot const * const snapshot);

void FlowVRServeFinalState(SimulationSnapshot *snapshot);

void FreeFlowVR();


#endif
#endif
