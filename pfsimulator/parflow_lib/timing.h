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
*
* Header file for parflow timing.
*
*****************************************************************************/

#ifndef _TIMING_HEADER
#define _TIMING_HEADER


/*--------------------------------------------------------------------------
 * With timing on
 * These values need to be in sync with the register statements in timing.c
 *--------------------------------------------------------------------------*/
#define SolverSetupTimingIndex 0
#define SolverTimingIndex 1
#define RichardsExclude1stTimeStepIndex 2
#define SolverCleanupTimingIndex 3
#define MatvecTimingIndex  4
#define PFSBTimingIndex  5
#define PFBTimingIndex  6
#define CLMTimingIndex  7
#define PFSOLReadTimingIndex  8
#define ClusteringTimingIndex 9
#define NetcdfTimingIndex 10
#ifdef VECTOR_UPDATE_TIMING
#define VectorUpdateTimingIndex  11
#endif


#if defined(PF_TIMING)
/*--------------------------------------------------------------------------
 * Global timing structure
 *--------------------------------------------------------------------------*/

typedef double FLOPType;

typedef struct {
  amps_Clock_t     *time;
  amps_CPUClock_t  *cpu_time;
  FLOPType         *flops;
  char            **name;

  int size;

  amps_Clock_t time_count;
  amps_CPUClock_t CPU_count;
  FLOPType FLOP_count;
} TimingType;

#ifdef PARFLOW_GLOBALS
amps_ThreadLocalDcl(TimingType *, timing_ptr);
#else
amps_ThreadLocalDcl(extern TimingType *, timing_ptr);
#endif

#define timing amps_ThreadLocal(timing_ptr)

/*--------------------------------------------------------------------------
 * Accessor functions
 *--------------------------------------------------------------------------*/

#define TimingTime(i)    (timing->time[(i)])
#define TimingCPUTime(i) (timing->cpu_time[(i)])
#define TimingFLOPS(i)   (timing->flops[(i)])
#define TimingName(i)    (timing->name[(i)])

#define TimingSize       (timing->size)

#define TimingTimeCount  (timing->time_count)
#define TimingCPUCount   (timing->CPU_count)
#define TimingFLOPCount  (timing->FLOP_count)

/*--------------------------------------------------------------------------
 * Timing macros
 *--------------------------------------------------------------------------*/

#define IncFLOPCount(inc) TimingFLOPCount += (FLOPType)inc
#define StartTiming()     TimingTimeCount -= amps_Clock(); \
  TimingCPUCount -= amps_CPUClock()
#define StopTiming()      TimingTimeCount += amps_Clock(); \
  TimingCPUCount += amps_CPUClock()

#ifdef TIMING_WITH_SYNC
#define BeginTiming(i)                  \
  {                                     \
    StopTiming();                       \
    TimingTime(i) -= TimingTimeCount;   \
    TimingCPUTime(i) -= TimingCPUCount; \
    TimingFLOPS(i) -= TimingFLOPCount;  \
    amps_Sync(amps_CommWorld);          \
    StartTiming();                      \
  }
#else
#define BeginTiming(i)                  \
  {                                     \
    StopTiming();                       \
    TimingTime(i) -= TimingTimeCount;   \
    TimingCPUTime(i) -= TimingCPUCount; \
    TimingFLOPS(i) -= TimingFLOPCount;  \
    StartTiming();                      \
  }
#endif

#define EndTiming(i)                    \
  {                                     \
    StopTiming();                       \
    TimingTime(i) += TimingTimeCount;   \
    TimingCPUTime(i) += TimingCPUCount; \
    TimingFLOPS(i) += TimingFLOPCount;  \
    StartTiming();                      \
  }

#ifdef VECTOR_UPDATE_TIMING

/* Global structure to hold some events */
#define MatvecStart 0
#define MatvecEnd 1
#define InitStart 2
#define InitEnd 3
#define FinalizeStart 4
#define FinalizeEnd 5

#ifdef PARFLOW_GLOBALS
int NumEvents = 0;
long EventTiming[10000][6];
#else
extern int NumEvents;
extern EventTiming[][6];
#endif

#endif

#else

/*--------------------------------------------------------------------------
 * With timing off
 *--------------------------------------------------------------------------*/

#define IncFLOPCount(inc)
#define StartTiming()
#define StopTiming()
#define BeginTiming(i) if (i == 0)
#define EndTiming(i)
#define NewTiming()
#define RegisterTiming(name) 0
#define PrintTiming()
#define FreeTiming()

#endif


#endif
