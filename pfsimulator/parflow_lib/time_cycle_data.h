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

#ifndef _TIME_CYCLE_HEADER
#define _TIME_CYCLE_HEADER

/*----------------------------------------------------------------
 * Time_Cycle structure
 *----------------------------------------------------------------*/

typedef struct {
  int number_of_cycles;
  int  *interval_divisions;
  int **intervals;
  int  *repeat_counts;
  int  *cycle_lengths;
} TimeCycleData;

/*--------------------------------------------------------------------------
 * Accessor macros: TimeCycleData
 *--------------------------------------------------------------------------*/

#define TimeCycleDataNumberOfCycles(time_cycle_data) ((time_cycle_data)->number_of_cycles)

#define TimeCycleDataIntervalDivisions(time_cycle_data) ((time_cycle_data)->interval_divisions)

#define TimeCycleDataIntervalDivision(time_cycle_data, cycle_number) ((time_cycle_data)->interval_divisions[cycle_number])

#define TimeCycleDataIntervalsPtr(time_cycle_data) ((time_cycle_data)->intervals)

#define TimeCycleDataIntervals(time_cycle_data, cycle_number) ((time_cycle_data)->intervals[cycle_number])

#define TimeCycleDataInterval(time_cycle_data, cycle_number, interval_number) (((time_cycle_data)->intervals[cycle_number])[interval_number])

#define TimeCycleDataRepeatCounts(time_cycle_data) ((time_cycle_data)->repeat_counts)

#define TimeCycleDataRepeatCount(time_cycle_data, cycle_number) ((time_cycle_data)->repeat_counts[cycle_number])

#define TimeCycleDataCycleLengths(time_cycle_data) ((time_cycle_data)->cycle_lengths)

#define TimeCycleDataCycleLength(time_cycle_data, cycle_number) ((time_cycle_data)->cycle_lengths[cycle_number])

#endif
