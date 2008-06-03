/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "parflow.h"

/******************************************************************************
 *
 * The functions in this file are for manipulating the WellData structure
 *   in ProblemData and work in conjuction with the WellPackage module.
 *
 *
 *****************************************************************************/


/*--------------------------------------------------------------------------
 * NewTimeCycleData
 *--------------------------------------------------------------------------*/

TimeCycleData *NewTimeCycleData(number_of_cycles, number_of_intervals)
int  number_of_cycles;
int *number_of_intervals;
{
   TimeCycleData *time_cycle_data;
   int            cycle_number;

   time_cycle_data = ctalloc(TimeCycleData, 1);

   TimeCycleDataNumberOfCycles(time_cycle_data) = number_of_cycles;

   TimeCycleDataIntervalDivisions(time_cycle_data) = ctalloc(int, number_of_cycles);

   TimeCycleDataIntervalsPtr(time_cycle_data) = ctalloc(int *, number_of_cycles);

   for(cycle_number = 0; cycle_number < number_of_cycles; cycle_number++)
   {
      TimeCycleDataIntervalDivision(time_cycle_data, cycle_number) = number_of_intervals[cycle_number];
      TimeCycleDataIntervals(time_cycle_data, cycle_number) = ctalloc(int, TimeCycleDataIntervalDivision(time_cycle_data, cycle_number));
   }

   TimeCycleDataRepeatCounts(time_cycle_data) = ctalloc(int, number_of_cycles);

   TimeCycleDataCycleLengths(time_cycle_data) = ctalloc(int, number_of_cycles);

   return time_cycle_data;
}


/*--------------------------------------------------------------------------
 * FreeTimeCycleData
 *--------------------------------------------------------------------------*/

void FreeTimeCycleData(time_cycle_data)
TimeCycleData *time_cycle_data;
{
   int               cycle_number;

   if ( time_cycle_data )
   {
      tfree( TimeCycleDataCycleLengths(time_cycle_data) );

      tfree( TimeCycleDataRepeatCounts(time_cycle_data) );

      if ( TimeCycleDataIntervalsPtr(time_cycle_data) )
      {
         for(cycle_number = 0; cycle_number < TimeCycleDataNumberOfCycles(time_cycle_data); cycle_number++)
         {
            tfree( TimeCycleDataIntervals(time_cycle_data, cycle_number) );
         }
         tfree( TimeCycleDataIntervalsPtr(time_cycle_data) );
      }

      tfree( TimeCycleDataIntervalDivisions(time_cycle_data) );

      tfree(time_cycle_data);
   }
}


/*--------------------------------------------------------------------------
 * PrintTimeCycleData
 *--------------------------------------------------------------------------*/

void PrintTimeCycleData(time_cycle_data)
TimeCycleData *time_cycle_data;
{
   int  cycle_number, interval_number;

   amps_Printf("Time Cycle Information");

   amps_Printf(" Number Of Cycles = %d\n", TimeCycleDataNumberOfCycles(time_cycle_data));

   amps_Printf(" Interval Divisions :\n");
   for(cycle_number = 0; cycle_number < TimeCycleDataNumberOfCycles(time_cycle_data); cycle_number++)
   {
      amps_Printf("  id[%02d] = %d\n", cycle_number, TimeCycleDataIntervalDivision(time_cycle_data, cycle_number));
   }

   amps_Printf(" Interval Data\n");
   for(cycle_number = 0; cycle_number < TimeCycleDataNumberOfCycles(time_cycle_data); cycle_number++)
   {
      for(interval_number = 0; interval_number < TimeCycleDataIntervalDivision(time_cycle_data, cycle_number); interval_number++)
      {
         amps_Printf("  d[%02d][%03d] = %d\n", cycle_number, interval_number, TimeCycleDataInterval(time_cycle_data, cycle_number, interval_number));
      }
   }

   amps_Printf(" Repeat Counts :\n");
   for(cycle_number = 0; cycle_number < TimeCycleDataNumberOfCycles(time_cycle_data); cycle_number++)
   {
      amps_Printf("  r[%02d] = %d\n", cycle_number, TimeCycleDataRepeatCount(time_cycle_data, cycle_number));
   }

   amps_Printf(" Cycle Lengths :\n");
   for(cycle_number = 0; cycle_number < TimeCycleDataNumberOfCycles(time_cycle_data); cycle_number++)
   {
      amps_Printf("  l[%02d] = %d\n", cycle_number, TimeCycleDataCycleLength(time_cycle_data, cycle_number));
   }
}


/*--------------------------------------------------------------------------
 * TimeCycleDataComputeIntervalNumber
 *--------------------------------------------------------------------------*/

int TimeCycleDataComputeIntervalNumber(problem, time, time_cycle_data, cycle_number)
Problem       *problem;
double         time;
TimeCycleData *time_cycle_data;
int            cycle_number;
{
   double   base_time_unit = ProblemBaseTimeUnit(problem);
   double   start_time     = ProblemStartTime(problem);

   int      repeat_count, cycle_length;
   int      n, intervals_completed, interval_number, total;
   double   current_cycle_time;

   interval_number = -1;
   if ( time_cycle_data != NULL )
   {
      repeat_count =  TimeCycleDataRepeatCount(time_cycle_data, cycle_number);
      cycle_length = TimeCycleDataCycleLength(time_cycle_data, cycle_number);
      if ( (repeat_count < 0) || (time < ((double) (repeat_count * cycle_length * base_time_unit))) )
      {
         n = (int) floor((time - start_time) / (((double) cycle_length) * base_time_unit));
         current_cycle_time = time - (start_time + ((double) (n * cycle_length)) * base_time_unit);
         intervals_completed = (int) floor(current_cycle_time / base_time_unit);
         interval_number = 0;
         total = TimeCycleDataInterval(time_cycle_data, cycle_number, interval_number);
         while(intervals_completed >= total)
         {
            interval_number++;
            total += TimeCycleDataInterval(time_cycle_data, cycle_number, interval_number);
         }
      }
      else
      {
         interval_number = TimeCycleDataIntervalDivision(time_cycle_data, cycle_number) - 1;
      }
   }
   return interval_number;
}


/*--------------------------------------------------------------------------
 * TimeCycleDataComputeNextTransition
 *--------------------------------------------------------------------------*/

double TimeCycleDataComputeNextTransition(problem, time, time_cycle_data)
Problem       *problem;
double         time;
TimeCycleData *time_cycle_data;
{
   double   base_time_unit = ProblemBaseTimeUnit(problem);
   double   start_time     = ProblemStartTime(problem);
   double   stop_time      = ProblemStopTime(problem);

   int      repeat_count, cycle_length, interval_division;
   int      n, cycle_number, intervals_completed, interval_number, next_interval_number, total;
   int      deltat_assigned;
   double   deltat, time_defined, current_cycle_time, transition_time;

   deltat = -1.0;

   if ( time_cycle_data != NULL )
   {
      deltat_assigned = FALSE;
      for(cycle_number = 0; cycle_number < TimeCycleDataNumberOfCycles(time_cycle_data); cycle_number++)
      {
         repeat_count      = TimeCycleDataRepeatCount(time_cycle_data, cycle_number);
         cycle_length      = TimeCycleDataCycleLength(time_cycle_data, cycle_number);
         interval_division = TimeCycleDataIntervalDivision(time_cycle_data, cycle_number);
         if ( repeat_count > 0 )
         {
            time_defined = ((double) (repeat_count * cycle_length * base_time_unit));
         }
         else
         {
            time_defined = stop_time;
         }

         if ( (time < time_defined) && (interval_division > 1) )
         {
            n = (int) floor((time - start_time) / (((double) cycle_length) * base_time_unit));
            current_cycle_time = time - (start_time + ((double) (n * cycle_length)) * base_time_unit);
            intervals_completed = (int) floor(current_cycle_time / base_time_unit);
            interval_number = 0;
            total = TimeCycleDataInterval(time_cycle_data, cycle_number, interval_number);
            while(intervals_completed >= total)
            {
               interval_number++;
               total += TimeCycleDataInterval(time_cycle_data, cycle_number, interval_number);
            }
            next_interval_number = (interval_number + 1) % interval_division;
            if (next_interval_number == 0)
            {
               n++;
               transition_time = start_time + ((double) (n * cycle_length)) * base_time_unit;
            }
            else
            {
               transition_time = start_time + ((double) (n * cycle_length + total)) * base_time_unit;
            }
            if ( transition_time < time_defined )
            {
               if ( deltat_assigned )
               {
                  deltat = min(deltat, (transition_time - time));
               }
               else
               {
                  deltat = transition_time - time;
                  deltat_assigned = TRUE;
               }
            }
         }
      }
   }

   return deltat;
}


void ReadGlobalTimeCycleData()
{

   int i;
   char *cycle_names;
   char *cycle_name;
   
   char key[IDB_MAX_KEY_LEN];
   char *id_names;
   
   NameArray id_na;

   int id;
   
   char *interval_name;

   /* Get the time cycling information */
   cycle_names = GetString("Cycle.Names");
   GlobalsCycleNames = NA_NewNameArray(cycle_names);

   GlobalsNumCycles = NA_Sizeof(GlobalsCycleNames);

   GlobalsIntervalDivisions  = ctalloc(int,   GlobalsNumCycles);
   GlobalsIntervals          = ctalloc(int *, GlobalsNumCycles);
   GlobalsIntervalNames      = ctalloc(NameArray, GlobalsNumCycles);
   GlobalsRepeatCounts       = ctalloc(int,   GlobalsNumCycles);

   for(i = 0; i < GlobalsNumCycles; i++)
   {
      cycle_name = NA_IndexToName(GlobalsCycleNames, i);

      sprintf(key, "Cycle.%s.Names", cycle_name); 
      id_names = GetString(key);

      id_na = NA_NewNameArray(id_names);
      GlobalsIntervalNames[i] = id_na;

      GlobalsIntervalDivisions[i] = NA_Sizeof(id_na);

      GlobalsIntervals[i] = ctalloc(int, GlobalsIntervalDivisions[i]);

      sprintf(key, "Cycle.%s.Repeat", cycle_name); 
      GlobalsRepeatCounts[i] = GetInt(key);

      for(id = 0; id < GlobalsIntervalDivisions[i]; id++)
      {
	 interval_name = NA_IndexToName(id_na, id);

	 sprintf(key, "Cycle.%s.%s.Length", cycle_name, interval_name);
	 GlobalsIntervals[i][id] = GetInt(key);
      }
   }
}

void FreeGlobalTimeCycleData()
{
   int i;

   for(i = 0; i < GlobalsNumCycles; i++)
   {
      NA_FreeNameArray(GlobalsIntervalNames[i]);
      tfree(GlobalsIntervals[i]);
   }
   NA_FreeNameArray(GlobalsCycleNames);
   tfree(GlobalsIntervalDivisions);
   tfree(GlobalsIntervals);
   tfree(GlobalsIntervalNames);
   tfree(GlobalsRepeatCounts);
   
}

