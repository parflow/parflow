/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#ifndef _TIME_CYCLE_HEADER
#define _TIME_CYCLE_HEADER

/*----------------------------------------------------------------
 * Time_Cycle structure
 *----------------------------------------------------------------*/

typedef struct
{
   int   number_of_cycles;
   int  *interval_divisions;
   int **intervals;
   int  *repeat_counts;
   int  *cycle_lengths;
} TimeCycleData;

/*--------------------------------------------------------------------------
 * Accessor macros: TimeCycleData
 *--------------------------------------------------------------------------*/

#define TimeCycleDataNumberOfCycles(time_cycle_data) ((time_cycle_data) -> number_of_cycles)

#define TimeCycleDataIntervalDivisions(time_cycle_data) ((time_cycle_data) -> interval_divisions)

#define TimeCycleDataIntervalDivision(time_cycle_data, cycle_number) ((time_cycle_data) -> interval_divisions[cycle_number])

#define TimeCycleDataIntervalsPtr(time_cycle_data) ((time_cycle_data) -> intervals)

#define TimeCycleDataIntervals(time_cycle_data, cycle_number) ((time_cycle_data) -> intervals[cycle_number])

#define TimeCycleDataInterval(time_cycle_data, cycle_number,interval_number) (((time_cycle_data) -> intervals[cycle_number])[interval_number])

#define TimeCycleDataRepeatCounts(time_cycle_data) ((time_cycle_data) -> repeat_counts)

#define TimeCycleDataRepeatCount(time_cycle_data, cycle_number) ((time_cycle_data) -> repeat_counts[cycle_number])

#define TimeCycleDataCycleLengths(time_cycle_data) ((time_cycle_data) -> cycle_lengths)

#define TimeCycleDataCycleLength(time_cycle_data, cycle_number) ((time_cycle_data) -> cycle_lengths[cycle_number])

#endif
