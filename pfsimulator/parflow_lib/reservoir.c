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

#include "parflow.h"

#include <string.h>


/*****************************************************************************
*
* The functions in this file are for manipulating the ReservoirData structure
*   in ProblemData and work in conjuction with the ReservoirPackage module.
*
* Because of the times things are called, the New function is twinky
* (it was basically put in to be symmetric with the New/Free paradigm
* used through out the code) and is invoked by SetProblemData.  The Alloc
* function actually allocates the data within the sub-structures and is
* invoked by the ReservoirPackage (which has the data needed to compute the array
* sizes and such).  The Free is smart enough to clean up after both the New
* and Alloc functions and is called by SetProblemData.
*
*****************************************************************************/


/*--------------------------------------------------------------------------
 * NewReservoirData
 *--------------------------------------------------------------------------*/

ReservoirData *NewReservoirData()
{
  ReservoirData    *reservoir_data;

  reservoir_data = ctalloc(ReservoirData, 1);

  ReservoirDataNumPhases(reservoir_data) = 0;
  ReservoirDataNumContaminants(reservoir_data) = 0;

  ReservoirDataNumReservoirs(reservoir_data) = -1;

  ReservoirDataTimeCycleData(reservoir_data) = NULL;

  ReservoirDataNumPressReservoirs(reservoir_data) = -1;
  ReservoirDataPressReservoirPhysicals(reservoir_data) = NULL;

  ReservoirDataNumFluxReservoirs(reservoir_data) = -1;
  ReservoirDataFluxReservoirPhysicals(reservoir_data) = NULL;


  return reservoir_data;
}


/*--------------------------------------------------------------------------
 * FreeReservoirData
 *--------------------------------------------------------------------------*/

void FreeReservoirData(
    ReservoirData *reservoir_data)
{
  ReservoirDataPhysical *reservoir_data_physical;
  int i, cycle_number, interval_division, interval_number;

  TimeCycleData   *time_cycle_data;

  if (reservoir_data)
  {
    if (ReservoirDataNumReservoirs(reservoir_data) > 0)
    {
      time_cycle_data = ReservoirDataTimeCycleData(reservoir_data);

      if (ReservoirDataNumFluxReservoirs(reservoir_data) > 0) {
        for (i = 0; i < ReservoirDataNumFluxReservoirs(reservoir_data); i++) {
          for (i = 0; i < ReservoirDataNumFluxReservoirs(reservoir_data); i++) {
            reservoir_data_physical = ReservoirDataFluxReservoirPhysical(reservoir_data, i);
            cycle_number = ReservoirDataPhysicalCycleNumber(reservoir_data_physical);
            interval_division = TimeCycleDataIntervalDivision(time_cycle_data, cycle_number);
            for (i = 0; i < ReservoirDataNumFluxReservoirs(reservoir_data); i++) {
              reservoir_data_physical = ReservoirDataFluxReservoirPhysical(reservoir_data, i);
              if (ReservoirDataPhysicalName(reservoir_data_physical)) {
                tfree(ReservoirDataPhysicalName(reservoir_data_physical));
              }
              if (ReservoirDataPhysicalIntakeSubgrid(reservoir_data_physical)) {
                FreeSubgrid(ReservoirDataPhysicalIntakeSubgrid(reservoir_data_physical));
              }
              tfree(reservoir_data_physical);
            }
            if (ReservoirDataFluxReservoirPhysicals(reservoir_data)) {
              tfree(ReservoirDataFluxReservoirPhysicals(reservoir_data));
            }
          }
        }
      }
      if (ReservoirDataNumPressReservoirs(reservoir_data) > 0)
      {
        for (i = 0; i < ReservoirDataNumPressReservoirs(reservoir_data); i++) {

          for (i = 0; i < ReservoirDataNumPressReservoirs(reservoir_data); i++) {
            reservoir_data_physical = ReservoirDataPressReservoirPhysical(reservoir_data, i);
            cycle_number = ReservoirDataPhysicalCycleNumber(reservoir_data_physical);
            interval_division = TimeCycleDataIntervalDivision(time_cycle_data, cycle_number);


            for (i = 0; i < ReservoirDataNumPressReservoirs(reservoir_data); i++) {
              reservoir_data_physical = ReservoirDataPressReservoirPhysical(reservoir_data, i);
              if (ReservoirDataPhysicalName(reservoir_data_physical)) {
                tfree(ReservoirDataPhysicalName(reservoir_data_physical));
              }
              if (ReservoirDataPhysicalIntakeSubgrid(reservoir_data_physical)) {
                FreeSubgrid(ReservoirDataPhysicalIntakeSubgrid(reservoir_data_physical));
              }
              tfree(reservoir_data_physical);
            }
            if (ReservoirDataPressReservoirPhysicals(reservoir_data)) {
              tfree(ReservoirDataPressReservoirPhysicals(reservoir_data));
            }
          }
        }
      FreeTimeCycleData(time_cycle_data);
    }
    tfree(reservoir_data);
  }
      }
}
