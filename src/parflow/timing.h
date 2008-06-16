/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
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
#define SolverCleanupTimingIndex 2
#define MatvecTimingIndex  3
#define PFSBTimingIndex  4
#define PFBTimingIndex  5
#ifdef VECTOR_UPDATE_TIMING
#define VectorUpdateTimingIndex  6
#endif


#if defined(PF_TIMING)
/*--------------------------------------------------------------------------
 * Global timing structure
 *--------------------------------------------------------------------------*/

typedef double FLOPType;

typedef struct
{
   amps_Clock_t     *time;
   amps_CPUClock_t  *cpu_time;
   FLOPType         *flops;
   char            **name;

   int               size;

   amps_Clock_t      time_count;
   amps_CPUClock_t   CPU_count;
   FLOPType          FLOP_count;

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

#define TimingTime(i)    (timing -> time[(i)])
#define TimingCPUTime(i) (timing -> cpu_time[(i)])
#define TimingFLOPS(i)   (timing -> flops[(i)])
#define TimingName(i)    (timing -> name[(i)])

#define TimingSize       (timing -> size)

#define TimingTimeCount  (timing -> time_count)
#define TimingCPUCount   (timing -> CPU_count)
#define TimingFLOPCount  (timing -> FLOP_count)

/*--------------------------------------------------------------------------
 * Timing macros
 *--------------------------------------------------------------------------*/

#define IncFLOPCount(inc) TimingFLOPCount += (FLOPType) inc
#define StartTiming()     TimingTimeCount -= amps_Clock(); \
                          TimingCPUCount -= amps_CPUClock()
#define StopTiming()      TimingTimeCount += amps_Clock(); \
                          TimingCPUCount += amps_CPUClock()

#ifdef TIMING_WITH_SYNC
#define BeginTiming(i) \
   { \
      StopTiming(); \
      TimingTime(i)    -= TimingTimeCount; \
      TimingCPUTime(i) -= TimingCPUCount; \
      TimingFLOPS(i)   -= TimingFLOPCount; \
      amps_Sync(amps_CommWorld); \
      StartTiming(); \
   }
#else
#define BeginTiming(i) \
   { \
      StopTiming(); \
      TimingTime(i)    -= TimingTimeCount; \
      TimingCPUTime(i) -= TimingCPUCount; \
      TimingFLOPS(i)   -= TimingFLOPCount; \
      StartTiming(); \
   }
#endif

#define EndTiming(i) \
   { \
      StopTiming(); \
      TimingTime(i)    += TimingTimeCount; \
      TimingCPUTime(i) += TimingCPUCount; \
      TimingFLOPS(i)   += TimingFLOPCount; \
      StartTiming(); \
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
#define BeginTiming(i) if(i == 0)
#define EndTiming(i)
#define NewTiming()
#define RegisterTiming(name) 0
#define PrintTiming()
#define FreeTiming()

#endif


#endif
