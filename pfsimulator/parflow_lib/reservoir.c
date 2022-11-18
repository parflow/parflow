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
  ReservoirDataPressReservoirValues(reservoir_data) = NULL;
  ReservoirDataPressReservoirStats(reservoir_data) = NULL;

  ReservoirDataNumFluxReservoirs(reservoir_data) = -1;
  ReservoirDataFluxReservoirPhysicals(reservoir_data) = NULL;
  ReservoirDataFluxReservoirValues(reservoir_data) = NULL;
  ReservoirDataFluxReservoirStats(reservoir_data) = NULL;

  return reservoir_data;
}


/*--------------------------------------------------------------------------
 * FreeReservoirData
 *--------------------------------------------------------------------------*/

void FreeReservoirData(
    ReservoirData *reservoir_data)
{
  ReservoirDataPhysical *reservoir_data_physical;
  ReservoirDataValue    *reservoir_data_value;
  ReservoirDataStat     *reservoir_data_stat;
  int i, cycle_number, interval_division, interval_number;

  TimeCycleData   *time_cycle_data;

  if (reservoir_data)
  {
    if (ReservoirDataNumReservoirs(reservoir_data) > 0)
    {
      time_cycle_data = ReservoirDataTimeCycleData(reservoir_data);

      if (ReservoirDataNumFluxReservoirs(reservoir_data) > 0)
      {
        for (i = 0; i < ReservoirDataNumFluxReservoirs(reservoir_data); i++)
        {
          reservoir_data_stat = ReservoirDataFluxReservoirStat(reservoir_data, i);
          if (ReservoirDataStatDeltaPhases(reservoir_data_stat))
          {
            tfree(ReservoirDataStatDeltaPhases(reservoir_data_stat));
          }
          if (ReservoirDataStatPhaseStats(reservoir_data_stat))
          {
            tfree(ReservoirDataStatPhaseStats(reservoir_data_stat));
          }
          if (ReservoirDataStatDeltaSaturations(reservoir_data_stat))
          {
            tfree(ReservoirDataStatDeltaSaturations(reservoir_data_stat));
          }
          if (ReservoirDataStatSaturationStats(reservoir_data_stat))
          {
            tfree(ReservoirDataStatSaturationStats(reservoir_data_stat));
          }
          if (ReservoirDataStatDeltaContaminants(reservoir_data_stat))
          {
            tfree(ReservoirDataStatDeltaContaminants(reservoir_data_stat));
          }
          if (ReservoirDataStatContaminantStats(reservoir_data_stat))
          {
            tfree(ReservoirDataStatContaminantStats(reservoir_data_stat));
          }
          tfree(reservoir_data_stat);
        }
        if (ReservoirDataFluxReservoirStats(reservoir_data))
        {
          tfree(ReservoirDataFluxReservoirStats(reservoir_data));
        }
        for (i = 0; i < ReservoirDataNumFluxReservoirs(reservoir_data); i++)
        {
          reservoir_data_physical = ReservoirDataFluxReservoirPhysical(reservoir_data, i);
          cycle_number = ReservoirDataPhysicalCycleNumber(reservoir_data_physical);
          interval_division = TimeCycleDataIntervalDivision(time_cycle_data, cycle_number);

          for (interval_number = 0; interval_number < interval_division; interval_number++)
          {
            reservoir_data_value = ReservoirDataFluxReservoirIntervalValue(reservoir_data, i, interval_number);
            if (ReservoirDataValuePhaseValues(reservoir_data_value))
            {
              tfree(ReservoirDataValuePhaseValues(reservoir_data_value));
            }
            if (ReservoirDataPhysicalAction(reservoir_data_physical) == INJECTION_RESERVOIR)
            {
              if (ReservoirDataValueSaturationValues(reservoir_data_value))
              {
                tfree(ReservoirDataValueSaturationValues(reservoir_data_value));
              }
              if (ReservoirDataValueContaminantValues(reservoir_data_value))
              {
                tfree(ReservoirDataValueContaminantValues(reservoir_data_value));
              }
            }
            if (ReservoirDataValueContaminantFractions(reservoir_data_value))
            {
              tfree(ReservoirDataValueContaminantFractions(reservoir_data_value));
            }
            tfree(reservoir_data_value);
          }
          if (ReservoirDataFluxReservoirIntervalValues(reservoir_data, i))
          {
            tfree(ReservoirDataFluxReservoirIntervalValues(reservoir_data, i));
          }
        }
        if (ReservoirDataFluxReservoirValues(reservoir_data))
        {
          tfree(ReservoirDataFluxReservoirValues(reservoir_data));
        }
        for (i = 0; i < ReservoirDataNumFluxReservoirs(reservoir_data); i++)
        {
          reservoir_data_physical = ReservoirDataFluxReservoirPhysical(reservoir_data, i);
          if (ReservoirDataPhysicalName(reservoir_data_physical))
          {
            tfree(ReservoirDataPhysicalName(reservoir_data_physical));
          }
          if (ReservoirDataPhysicalSubgrid(reservoir_data_physical))
          {
            FreeSubgrid(ReservoirDataPhysicalSubgrid(reservoir_data_physical));
          }
          tfree(reservoir_data_physical);
        }
        if (ReservoirDataFluxReservoirPhysicals(reservoir_data))
        {
          tfree(ReservoirDataFluxReservoirPhysicals(reservoir_data));
        }
      }

      if (ReservoirDataNumPressReservoirs(reservoir_data) > 0)
      {
        for (i = 0; i < ReservoirDataNumPressReservoirs(reservoir_data); i++)
        {
          reservoir_data_stat = ReservoirDataPressReservoirStat(reservoir_data, i);
          if (ReservoirDataStatDeltaPhases(reservoir_data_stat))
          {
            tfree(ReservoirDataStatDeltaPhases(reservoir_data_stat));
          }
          if (ReservoirDataStatPhaseStats(reservoir_data_stat))
          {
            tfree(ReservoirDataStatPhaseStats(reservoir_data_stat));
          }
          if (ReservoirDataStatDeltaSaturations(reservoir_data_stat))
          {
            tfree(ReservoirDataStatDeltaSaturations(reservoir_data_stat));
          }
          if (ReservoirDataStatSaturationStats(reservoir_data_stat))
          {
            tfree(ReservoirDataStatSaturationStats(reservoir_data_stat));
          }
          if (ReservoirDataStatDeltaContaminants(reservoir_data_stat))
          {
            tfree(ReservoirDataStatDeltaContaminants(reservoir_data_stat));
          }
          if (ReservoirDataStatContaminantStats(reservoir_data_stat))
          {
            tfree(ReservoirDataStatContaminantStats(reservoir_data_stat));
          }
          tfree(reservoir_data_stat);
        }
        if (ReservoirDataPressReservoirStats(reservoir_data))
        {
          tfree(ReservoirDataPressReservoirStats(reservoir_data));
        }
        for (i = 0; i < ReservoirDataNumPressReservoirs(reservoir_data); i++)
        {
          reservoir_data_physical = ReservoirDataPressReservoirPhysical(reservoir_data, i);
          cycle_number = ReservoirDataPhysicalCycleNumber(reservoir_data_physical);
          interval_division = TimeCycleDataIntervalDivision(time_cycle_data, cycle_number);

          for (interval_number = 0; interval_number < interval_division; interval_number++)
          {
            reservoir_data_value = ReservoirDataPressReservoirIntervalValue(reservoir_data, i, interval_number);
            if (ReservoirDataValuePhaseValues(reservoir_data_value))
            {
              tfree(ReservoirDataValuePhaseValues(reservoir_data_value));
            }
            if (ReservoirDataPhysicalAction(reservoir_data_physical) == INJECTION_RESERVOIR)
            {
              if (ReservoirDataValueSaturationValues(reservoir_data_value))
              {
                tfree(ReservoirDataValueSaturationValues(reservoir_data_value));
              }
              if (ReservoirDataValueContaminantValues(reservoir_data_value))
              {
                tfree(ReservoirDataValueContaminantValues(reservoir_data_value));
              }
            }
            if (ReservoirDataValueContaminantFractions(reservoir_data_value))
            {
              tfree(ReservoirDataValueContaminantFractions(reservoir_data_value));
            }
            tfree(reservoir_data_value);
          }
          if (ReservoirDataPressReservoirIntervalValues(reservoir_data, i))
          {
            tfree(ReservoirDataPressReservoirIntervalValues(reservoir_data, i));
          }
        }
        if (ReservoirDataPressReservoirValues(reservoir_data))
        {
          tfree(ReservoirDataPressReservoirValues(reservoir_data));
        }
        for (i = 0; i < ReservoirDataNumPressReservoirs(reservoir_data); i++)
        {
          reservoir_data_physical = ReservoirDataPressReservoirPhysical(reservoir_data, i);
          if (ReservoirDataPhysicalName(reservoir_data_physical))
          {
            tfree(ReservoirDataPhysicalName(reservoir_data_physical));
          }
          if (ReservoirDataPhysicalSubgrid(reservoir_data_physical))
          {
            FreeSubgrid(ReservoirDataPhysicalSubgrid(reservoir_data_physical));
          }
          tfree(reservoir_data_physical);
        }
        if (ReservoirDataPressReservoirPhysicals(reservoir_data))
        {
          tfree(ReservoirDataPressReservoirPhysicals(reservoir_data));
        }
      }
      FreeTimeCycleData(time_cycle_data);
    }
    tfree(reservoir_data);
  }
}


/*--------------------------------------------------------------------------
 * PrintReservoirData
 *--------------------------------------------------------------------------*/

void PrintReservoirData(
    ReservoirData *   reservoir_data,
    unsigned int print_mask)
{
  TimeCycleData    *time_cycle_data;

  ReservoirDataPhysical *reservoir_data_physical;
  ReservoirDataValue    *reservoir_data_value;
  ReservoirDataStat     *reservoir_data_stat;

  Subgrid          *subgrid;

  int cycle_number, interval_division, interval_number;
  int reservoir, phase, concentration, indx;
  double value, stat;

  amps_Printf("Reservoir Information\n");
  if (ReservoirDataNumReservoirs(reservoir_data) == -1)
  {
    amps_Printf("Reservoirs have not been setup.\n");
  }
  else if (ReservoirDataNumReservoirs(reservoir_data) == 0)
  {
    amps_Printf("No Reservoirs.\n");
  }
  else
  {
    time_cycle_data = ReservoirDataTimeCycleData(reservoir_data);

    PrintTimeCycleData(time_cycle_data);

    if (ReservoirDataNumFluxReservoirs(reservoir_data) > 0)
    {
      amps_Printf("Info on Flux Reservoirs :\n");
      for (reservoir = 0; reservoir < ReservoirDataNumFluxReservoirs(reservoir_data); reservoir++)
      {
        amps_Printf(" Flux Reservoir Number : %02d\n", reservoir);

        if ((print_mask & RESERVOIRDATA_PRINTPHYSICAL))
        {
          amps_Printf("  Reservoir Physical Data :\n");
          reservoir_data_physical = ReservoirDataFluxReservoirPhysical(reservoir_data, reservoir);

          amps_Printf("   sequence number = %2d\n",
                      ReservoirDataPhysicalNumber(reservoir_data_physical));
          amps_Printf("   name = %s\n",
                      ReservoirDataPhysicalName(reservoir_data_physical));
          amps_Printf("   x_lower, y_lower, z_lower = %f %f %f\n",
                      ReservoirDataPhysicalXLower(reservoir_data_physical),
                      ReservoirDataPhysicalYLower(reservoir_data_physical),
                      ReservoirDataPhysicalZLower(reservoir_data_physical));
          amps_Printf("   x_upper, y_upper, z_upper = %f %f %f\n",
                      ReservoirDataPhysicalXUpper(reservoir_data_physical),
                      ReservoirDataPhysicalYUpper(reservoir_data_physical),
                      ReservoirDataPhysicalZUpper(reservoir_data_physical));
          amps_Printf("   diameter = %f\n",
                      ReservoirDataPhysicalDiameter(reservoir_data_physical));

          subgrid = ReservoirDataPhysicalSubgrid(reservoir_data_physical);
          amps_Printf("   (ix, iy, iz) = (%d, %d, %d)\n",
                      SubgridIX(subgrid),
                      SubgridIY(subgrid),
                      SubgridIZ(subgrid));
          amps_Printf("   (nx, ny, nz) = (%d, %d, %d)\n",
                      SubgridNX(subgrid),
                      SubgridNY(subgrid),
                      SubgridNZ(subgrid));
          amps_Printf("   (rx, ry, rz) = (%d, %d, %d)\n",
                      SubgridRX(subgrid),
                      SubgridRY(subgrid),
                      SubgridRZ(subgrid));
          amps_Printf("   process = %d\n",
                      SubgridProcess(subgrid));
          amps_Printf("   size = %f\n",
                      ReservoirDataPhysicalSize(reservoir_data_physical));
          amps_Printf("   action = %d\n",
                      ReservoirDataPhysicalAction(reservoir_data_physical));
          amps_Printf("   method = %d\n",
                      ReservoirDataPhysicalMethod(reservoir_data_physical));
        }

        if ((print_mask & RESERVOIRDATA_PRINTVALUES))
        {
          amps_Printf("  Reservoir Values :\n");
          reservoir_data_physical = ReservoirDataFluxReservoirPhysical(reservoir_data, reservoir);
          cycle_number = ReservoirDataPhysicalCycleNumber(reservoir_data_physical);
          interval_division = TimeCycleDataIntervalDivision(time_cycle_data, cycle_number);

          for (interval_number = 0; interval_number < interval_division; interval_number++)
          {
            amps_Printf("  Value[%2d] :\n", interval_number);
            reservoir_data_value = ReservoirDataFluxReservoirIntervalValue(reservoir_data, reservoir, interval_number);

            if (ReservoirDataValuePhaseValues(reservoir_data_value))
            {
              for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
              {
                value = ReservoirDataValuePhaseValue(reservoir_data_value, phase);
                amps_Printf("   value for phase %01d = %f\n", phase, value);
              }
            }
            else
            {
              amps_Printf("   no phase values present.\n");
            }
            if (ReservoirDataPhysicalAction(reservoir_data_physical) == INJECTION_RESERVOIR)
            {
              if (ReservoirDataValueSaturationValues(reservoir_data_value))
              {
                for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
                {
                  value = ReservoirDataValueSaturationValue(reservoir_data_value, phase);
                  amps_Printf("   s_bar[%01d] = %f\n", phase, value);
                }
              }
              else
              {
                amps_Printf("   no saturation values present.\n");
              }
              if (ReservoirDataValueDeltaSaturationPtrs(reservoir_data_value))
              {
                for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
                {
                  value = ReservoirDataValueDeltaSaturationPtr(reservoir_data_value, phase);
                  amps_Printf("   delta_saturations[%01d] = %f\n", phase, value);
                }
              }
              else
              {
                amps_Printf("   no saturation delta values present.\n");
              }
              if (ReservoirDataValueContaminantValues(reservoir_data_value))
              {
                for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
                {
                  for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
                  {
                    indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                    value = ReservoirDataValueContaminantValue(reservoir_data_value, indx);
                    amps_Printf("   c_bar[%01d][%02d] = %f\n", phase, concentration, value);
                  }
                }
              }
              else
              {
                amps_Printf("   no contaminant values present.\n");
              }
              if (ReservoirDataValueDeltaContaminantPtrs(reservoir_data_value))
              {
                for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
                {
                  for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
                  {
                    indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                    value = ReservoirDataValueDeltaContaminantPtr(reservoir_data_value, indx);
                    amps_Printf("  delta_concentration[%01d][%02d] = %f\n", phase, concentration, value);
                  }
                }
              }
              else
              {
                amps_Printf("   no concentration delta values present.\n");
              }
            }
            if (ReservoirDataValueContaminantFractions(reservoir_data_value))
            {
              for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
              {
                for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
                {
                  indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                  value = ReservoirDataValueContaminantFraction(reservoir_data_value, indx);
                  amps_Printf("   relevant_fraction[%01d][%02d] = %f\n", phase, concentration, value);
                }
              }
            }
            else
            {
              amps_Printf("   no relevant component fractions present.\n");
            }
          }
        }

        if ((print_mask & RESERVOIRDATA_PRINTSTATS))
        {
          amps_Printf("  Reservoir Stats :\n");
          reservoir_data_stat = ReservoirDataFluxReservoirStat(reservoir_data, reservoir);
          if (ReservoirDataStatDeltaPhases(reservoir_data_stat))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              stat = ReservoirDataStatDeltaPhase(reservoir_data_stat, phase);
              amps_Printf("  delta_p[%01d] = %f\n", phase, stat);
            }
          }
          if (ReservoirDataStatPhaseStats(reservoir_data_stat))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              stat = ReservoirDataStatPhaseStat(reservoir_data_stat, phase);
              amps_Printf("  phase[%01d] = %f\n", phase, stat);
            }
          }
          if (ReservoirDataStatDeltaSaturations(reservoir_data_stat))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              stat = ReservoirDataStatDeltaSaturation(reservoir_data_stat, phase);
              amps_Printf("  delta_s[%01d] = %f\n", phase, stat);
            }
          }
          if (ReservoirDataStatSaturationStats(reservoir_data_stat))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              stat = ReservoirDataStatSaturationStat(reservoir_data_stat, phase);
              amps_Printf("  saturation[%01d] = %f\n", phase, stat);
            }
          }
          if (ReservoirDataStatDeltaContaminants(reservoir_data_stat))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
              {
                indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                stat = ReservoirDataStatDeltaContaminant(reservoir_data_stat, indx);
                amps_Printf("  delta_c[%01d][%02d] = %f\n", phase, concentration, stat);
              }
            }
          }
          if (ReservoirDataStatContaminantStats(reservoir_data_stat))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
              {
                indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                stat = ReservoirDataStatContaminantStat(reservoir_data_stat, indx);
                amps_Printf("  concentration[%01d][%02d] = %f\n", phase, concentration, stat);
              }
            }
          }
        }
      }
    }
    else
    {
      amps_Printf("No Flux Reservoirs.\n");
    }

    if (ReservoirDataNumPressReservoirs(reservoir_data) > 0)
    {
      amps_Printf("Info on Pressure Reservoirs :\n");
      for (reservoir = 0; reservoir < ReservoirDataNumPressReservoirs(reservoir_data); reservoir++)
      {
        amps_Printf(" Pressure Reservoir Number : %2d\n", reservoir);
        if ((print_mask & RESERVOIRDATA_PRINTPHYSICAL))
        {
          amps_Printf("  Reservoir Physical Data :\n");
          reservoir_data_physical = ReservoirDataPressReservoirPhysical(reservoir_data, reservoir);

          amps_Printf("   sequence number = %2d\n",
                      ReservoirDataPhysicalNumber(reservoir_data_physical));
          amps_Printf("   name = %s\n",
                      ReservoirDataPhysicalName(reservoir_data_physical));
          amps_Printf("   x_lower, y_lower, z_lower = %f %f %f\n",
                      ReservoirDataPhysicalXLower(reservoir_data_physical),
                      ReservoirDataPhysicalYLower(reservoir_data_physical),
                      ReservoirDataPhysicalZLower(reservoir_data_physical));
          amps_Printf("   x_upper, y_upper, z_upper = %f %f %f\n",
                      ReservoirDataPhysicalXUpper(reservoir_data_physical),
                      ReservoirDataPhysicalYUpper(reservoir_data_physical),
                      ReservoirDataPhysicalZUpper(reservoir_data_physical));
          amps_Printf("   diameter = %f\n",
                      ReservoirDataPhysicalDiameter(reservoir_data_physical));

          subgrid = ReservoirDataPhysicalSubgrid(reservoir_data_physical);
          amps_Printf("   (ix, iy, iz) = (%d, %d, %d)\n",
                      SubgridIX(subgrid),
                      SubgridIY(subgrid),
                      SubgridIZ(subgrid));
          amps_Printf("   (nx, ny, nz) = (%d, %d, %d)\n",
                      SubgridNX(subgrid),
                      SubgridNY(subgrid),
                      SubgridNZ(subgrid));
          amps_Printf("   (rx, ry, rz) = (%d, %d, %d)\n",
                      SubgridRX(subgrid),
                      SubgridRY(subgrid),
                      SubgridRZ(subgrid));
          amps_Printf("   process = %d\n",
                      SubgridProcess(subgrid));
          amps_Printf("   size = %f\n",
                      ReservoirDataPhysicalSize(reservoir_data_physical));
          amps_Printf("   action = %d\n",
                      ReservoirDataPhysicalAction(reservoir_data_physical));
          amps_Printf("   method = %d\n",
                      ReservoirDataPhysicalMethod(reservoir_data_physical));
        }

        if ((print_mask & RESERVOIRDATA_PRINTVALUES))
        {
          amps_Printf("  Reservoir Values :\n");
          reservoir_data_physical = ReservoirDataPressReservoirPhysical(reservoir_data, reservoir);
          cycle_number = ReservoirDataPhysicalCycleNumber(reservoir_data_physical);
          interval_division = TimeCycleDataIntervalDivision(time_cycle_data, cycle_number);

          for (interval_number = 0; interval_number < interval_division; interval_number++)
          {
            amps_Printf("  Value[%2d] :\n", interval_number);

            reservoir_data_value = ReservoirDataPressReservoirIntervalValue(reservoir_data, reservoir, interval_number);

            if (ReservoirDataValuePhaseValues(reservoir_data_value))
            {
              phase = 0;
              {
                value = ReservoirDataValuePhaseValue(reservoir_data_value, phase);
                amps_Printf("   value for phase %01d = %f\n", phase, value);
              }
            }
            else
            {
              amps_Printf("   no phase values present.\n");
            }
            if (ReservoirDataPhysicalAction(reservoir_data_physical) == INJECTION_RESERVOIR)
            {
              if (ReservoirDataValueSaturationValues(reservoir_data_value))
              {
                for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
                {
                  value = ReservoirDataValueSaturationValue(reservoir_data_value, phase);
                  amps_Printf("   s_bar[%01d] = %f\n", phase, value);
                }
              }
              else
              {
                amps_Printf("   no saturation values present.\n");
              }
              if (ReservoirDataValueDeltaSaturationPtrs(reservoir_data_value))
              {
                for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
                {
                  value = ReservoirDataValueDeltaSaturationPtr(reservoir_data_value, phase);
                  amps_Printf("   delta_saturations[%01d] = %f\n", phase, value);
                }
              }
              else
              {
                amps_Printf("   no saturation delta values present.\n");
              }
              if (ReservoirDataValueContaminantValues(reservoir_data_value))
              {
                for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
                {
                  for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
                  {
                    indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                    value = ReservoirDataValueContaminantValue(reservoir_data_value, indx);
                    amps_Printf("   c_bar[%01d][%02d] = %f\n", phase, concentration, value);
                  }
                }
              }
              else
              {
                amps_Printf("   no component values present.\n");
              }
              if (ReservoirDataValueDeltaContaminantPtrs(reservoir_data_value))
              {
                for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
                {
                  for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
                  {
                    indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                    value = ReservoirDataValueDeltaContaminantPtr(reservoir_data_value, indx);
                    amps_Printf("  delta_concentration[%01d][%02d] = %f\n", phase, concentration, value);
                  }
                }
              }
              else
              {
                amps_Printf("   no concentration delta values present.\n");
              }
            }
            if (ReservoirDataValueContaminantFractions(reservoir_data_value))
            {
              for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
              {
                for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
                {
                  indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                  value = ReservoirDataValueContaminantFraction(reservoir_data_value, indx);
                  amps_Printf("   relevant_fraction[%01d][%02d] = %f\n", phase, concentration, value);
                }
              }
            }
            else
            {
              amps_Printf("   no relevant component values present.\n");
            }
          }
        }

        if ((print_mask & RESERVOIRDATA_PRINTSTATS))
        {
          amps_Printf("  Reservoir Stats :\n");
          reservoir_data_stat = ReservoirDataPressReservoirStat(reservoir_data, reservoir);
          if (ReservoirDataStatDeltaPhases(reservoir_data_stat))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              stat = ReservoirDataStatDeltaPhase(reservoir_data_stat, phase);
              amps_Printf("  delta_p[%01d] = %f\n", phase, stat);
            }
          }
          if (ReservoirDataStatPhaseStats(reservoir_data_stat))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              stat = ReservoirDataStatPhaseStat(reservoir_data_stat, phase);
              amps_Printf("  phase[%01d] = %f\n", phase, stat);
            }
          }
          if (ReservoirDataStatDeltaSaturations(reservoir_data_stat))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              stat = ReservoirDataStatDeltaSaturation(reservoir_data_stat, phase);
              amps_Printf("  delta_s[%01d] = %f\n", phase, stat);
            }
          }
          if (ReservoirDataStatSaturationStats(reservoir_data_stat))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              stat = ReservoirDataStatSaturationStat(reservoir_data_stat, phase);
              amps_Printf("  saturation[%01d] = %f\n", phase, stat);
            }
          }
          if (ReservoirDataStatDeltaContaminants(reservoir_data_stat))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
              {
                indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                stat = ReservoirDataStatDeltaContaminant(reservoir_data_stat, indx);
                amps_Printf("  delta_c[%01d][%02d] = %f\n", phase, concentration, stat);
              }
            }
          }
          if (ReservoirDataStatContaminantStats(reservoir_data_stat))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
              {
                indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                stat = ReservoirDataStatContaminantStat(reservoir_data_stat, indx);
                amps_Printf("  concentration[%01d][%02d] = %f\n", phase, concentration, stat);
              }
            }
          }
        }
      }
    }
    else
    {
      amps_Printf("No Pressure Reservoirs.\n");
    }
  }
}

/*--------------------------------------------------------------------------
 * WriteReservoirs
 *--------------------------------------------------------------------------*/

void WriteReservoirs(
    char *    file_prefix,
    Problem * problem,
    ReservoirData *reservoir_data,
    double    time,
    int       write_header)
{
  TimeCycleData    *time_cycle_data;
  ReservoirDataPhysical *reservoir_data_physical;
  ReservoirDataValue    *reservoir_data_value;
  ReservoirDataStat     *reservoir_data_stat;

  Subgrid          *subgrid;

  int cycle_number, interval_number;
  int reservoir, phase, concentration, indx;
  double value, stat;

  FILE             *file;

  char file_suffix[11] = "reservoirs";
  char filename[255];

  int p;

  if (ReservoirDataNumReservoirs(reservoir_data) > 0)
  {
    p = amps_Rank(amps_CommWorld);

    time_cycle_data = ReservoirDataTimeCycleData(reservoir_data);

    if (p == 0)
    {
      sprintf(filename, "%s.%s", file_prefix, file_suffix);

      if (write_header)
      {
        file = fopen(filename, "w");
      }
      else
      {
        file = fopen(filename, "a");
      }

      if (file == NULL)
      {
        amps_Printf("Error: can't open output file %s\n", filename);
        exit(1);
      }

      if (write_header)
      {
        fprintf(file, "%f %f %f %d %d %d %f %f %f\n",
                BackgroundX(GlobalsBackground),
                BackgroundY(GlobalsBackground),
                BackgroundZ(GlobalsBackground),
                BackgroundNX(GlobalsBackground),
                BackgroundNY(GlobalsBackground),
                BackgroundNZ(GlobalsBackground),
                BackgroundDX(GlobalsBackground),
                BackgroundDY(GlobalsBackground),
                BackgroundDZ(GlobalsBackground));

        fprintf(file, "%d %d %d\n",
                ReservoirDataNumPhases(reservoir_data),
                ReservoirDataNumContaminants(reservoir_data),
                ReservoirDataNumReservoirs(reservoir_data));

        for (reservoir = 0; reservoir < ReservoirDataNumFluxReservoirs(reservoir_data); reservoir++)
        {
          reservoir_data_physical = ReservoirDataFluxReservoirPhysical(reservoir_data, reservoir);

          fprintf(file, "%2d\n", ReservoirDataPhysicalNumber(reservoir_data_physical));

          fprintf(file, "%d\n", (int)strlen(ReservoirDataPhysicalName(reservoir_data_physical)));
          fprintf(file, "%s\n", ReservoirDataPhysicalName(reservoir_data_physical));

          fprintf(file, "%f %f %f %f %f %f %f\n",
                  ReservoirDataPhysicalXLower(reservoir_data_physical),
                  ReservoirDataPhysicalYLower(reservoir_data_physical),
                  ReservoirDataPhysicalZLower(reservoir_data_physical),
                  ReservoirDataPhysicalXUpper(reservoir_data_physical),
                  ReservoirDataPhysicalYUpper(reservoir_data_physical),
                  ReservoirDataPhysicalZUpper(reservoir_data_physical),
                  ReservoirDataPhysicalDiameter(reservoir_data_physical));
          fprintf(file, "1 %1d %1d\n",
                  ReservoirDataPhysicalAction(reservoir_data_physical),
                  ReservoirDataPhysicalMethod(reservoir_data_physical));
        }
        for (reservoir = 0; reservoir < ReservoirDataNumPressReservoirs(reservoir_data); reservoir++)
        {
          reservoir_data_physical = ReservoirDataPressReservoirPhysical(reservoir_data, reservoir);

          fprintf(file, "%2d\n", ReservoirDataPhysicalNumber(reservoir_data_physical));

          fprintf(file, "%d\n", (int)strlen(ReservoirDataPhysicalName(reservoir_data_physical)));
          fprintf(file, "%s\n", ReservoirDataPhysicalName(reservoir_data_physical));

          fprintf(file, "%f %f %f %f %f %f %f\n",
                  ReservoirDataPhysicalXLower(reservoir_data_physical),
                  ReservoirDataPhysicalYLower(reservoir_data_physical),
                  ReservoirDataPhysicalZLower(reservoir_data_physical),
                  ReservoirDataPhysicalXUpper(reservoir_data_physical),
                  ReservoirDataPhysicalYUpper(reservoir_data_physical),
                  ReservoirDataPhysicalZUpper(reservoir_data_physical),
                  ReservoirDataPhysicalDiameter(reservoir_data_physical));
          fprintf(file, "0 %1d %1d\n",
                  ReservoirDataPhysicalAction(reservoir_data_physical),
                  ReservoirDataPhysicalMethod(reservoir_data_physical));
        }
      }

      fprintf(file, "%f\n", time);

      for (reservoir = 0; reservoir < ReservoirDataNumFluxReservoirs(reservoir_data); reservoir++)
      {
        /* Write out important current physical data */
        reservoir_data_physical = ReservoirDataFluxReservoirPhysical(reservoir_data, reservoir);
        fprintf(file, "%2d\n", ReservoirDataPhysicalNumber(reservoir_data_physical));
        subgrid = ReservoirDataPhysicalSubgrid(reservoir_data_physical);
        fprintf(file, "%d %d %d %d %d %d %d %d %d\n",
                SubgridIX(subgrid),
                SubgridIY(subgrid),
                SubgridIZ(subgrid),
                SubgridNX(subgrid),
                SubgridNY(subgrid),
                SubgridNZ(subgrid),
                SubgridRX(subgrid),
                SubgridRY(subgrid),
                SubgridRZ(subgrid));

        /* Write out the current reservoir values */
        cycle_number = ReservoirDataPhysicalCycleNumber(reservoir_data_physical);
        interval_number = TimeCycleDataComputeIntervalNumber(problem, time, time_cycle_data, cycle_number);

        reservoir_data_value = ReservoirDataFluxReservoirIntervalValue(reservoir_data, reservoir, interval_number);

        if (ReservoirDataValuePhaseValues(reservoir_data_value))
        {
          for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
          {
            value = ReservoirDataValuePhaseValue(reservoir_data_value, phase);
            fprintf(file, " %f", value);
          }
          fprintf(file, "\n");
        }
        if (ReservoirDataPhysicalAction(reservoir_data_physical) == INJECTION_RESERVOIR)
        {
          if (ReservoirDataValueDeltaSaturationPtrs(reservoir_data_value))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              value = ReservoirDataValueDeltaSaturationPtr(reservoir_data_value, phase);
              fprintf(file, " %f", value);
            }
            fprintf(file, "\n");
          }
          else if (ReservoirDataValueSaturationValues(reservoir_data_value))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              value = ReservoirDataValueSaturationValue(reservoir_data_value, phase);
              fprintf(file, " %f", value);
            }
            fprintf(file, "\n");
          }
          if (ReservoirDataValueDeltaContaminantPtrs(reservoir_data_value))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
              {
                indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                value = ReservoirDataValueContaminantFraction(reservoir_data_value, indx)
                        * fabs(ReservoirDataValueDeltaContaminantPtr(reservoir_data_value, indx))
                        / ReservoirDataPhysicalSize(reservoir_data_physical);
                fprintf(file, " %f", value);
              }
            }
            fprintf(file, "\n");
          }
          else if (ReservoirDataValueContaminantValues(reservoir_data_value))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
              {
                indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                value = ReservoirDataValueContaminantValue(reservoir_data_value, indx);
                fprintf(file, " %f", value);
              }
            }
            fprintf(file, "\n");
          }
        }
        if (ReservoirDataValueContaminantFractions(reservoir_data_value))
        {
          for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
          {
            for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
            {
              indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
              value = ReservoirDataValueContaminantFraction(reservoir_data_value, indx);
              fprintf(file, " %f", value);
            }
          }
          fprintf(file, "\n");
        }

        /* Write out the current reservoir statistics */
        reservoir_data_stat = ReservoirDataFluxReservoirStat(reservoir_data, reservoir);
        if (ReservoirDataStatPhaseStats(reservoir_data_stat))
        {
          for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
          {
            stat = ReservoirDataStatPhaseStat(reservoir_data_stat, phase);
            fprintf(file, " %f", stat);
          }
          fprintf(file, "\n");
        }
        if (ReservoirDataStatSaturationStats(reservoir_data_stat))
        {
          for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
          {
            stat = ReservoirDataStatSaturationStat(reservoir_data_stat, phase);
            fprintf(file, " %f", stat);
          }
          fprintf(file, "\n");
        }
        if (ReservoirDataStatContaminantStats(reservoir_data_stat))
        {
          for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
          {
            for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
            {
              indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
              stat = ReservoirDataStatContaminantStat(reservoir_data_stat, indx);
              fprintf(file, " %f", stat);
            }
          }
          fprintf(file, "\n");
        }
        if (ReservoirDataStatDeltaPhases(reservoir_data_stat) && ReservoirDataStatDeltaContaminants(reservoir_data_stat))
        {
          for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
          {
            for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
            {
              indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
              if (ReservoirDataStatDeltaPhase(reservoir_data_stat, phase) == 0.0)
              {
                stat = 0.0;
              }
              else
              {
                stat = ReservoirDataStatDeltaContaminant(reservoir_data_stat, indx) / ReservoirDataStatDeltaPhase(reservoir_data_stat, phase);
              }
              fprintf(file, " %f", stat);
            }
          }
          fprintf(file, "\n");
        }
      }

      for (reservoir = 0; reservoir < ReservoirDataNumPressReservoirs(reservoir_data); reservoir++)
      {
        /* Write out important current physical data */
        reservoir_data_physical = ReservoirDataPressReservoirPhysical(reservoir_data, reservoir);
        fprintf(file, "%2d\n", ReservoirDataPhysicalNumber(reservoir_data_physical));
        subgrid = ReservoirDataPhysicalSubgrid(reservoir_data_physical);
        fprintf(file, "%d %d %d %d %d %d %d %d %d\n",
                SubgridIX(subgrid),
                SubgridIY(subgrid),
                SubgridIZ(subgrid),
                SubgridNX(subgrid),
                SubgridNY(subgrid),
                SubgridNZ(subgrid),
                SubgridRX(subgrid),
                SubgridRY(subgrid),
                SubgridRZ(subgrid));

        /* Write out the current reservoir values */
        cycle_number = ReservoirDataPhysicalCycleNumber(reservoir_data_physical);
        interval_number = TimeCycleDataComputeIntervalNumber(problem,
                                                             time, time_cycle_data, cycle_number);

        reservoir_data_value = ReservoirDataPressReservoirIntervalValue(reservoir_data, reservoir,
                                                                        interval_number);

        if (ReservoirDataValuePhaseValues(reservoir_data_value))
        {
          phase = 0;
          {
            value = ReservoirDataValuePhaseValue(reservoir_data_value, phase);
            fprintf(file, " %f", value);
          }
          fprintf(file, "\n");
        }
        if (ReservoirDataPhysicalAction(reservoir_data_physical) == INJECTION_RESERVOIR)
        {
          if (ReservoirDataValueDeltaSaturationPtrs(reservoir_data_value))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              value = ReservoirDataValueDeltaSaturationPtr(reservoir_data_value, phase);
              fprintf(file, " %f", value);
            }
            fprintf(file, "\n");
          }
          else if (ReservoirDataValueSaturationValues(reservoir_data_value))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              value = ReservoirDataValueSaturationValue(reservoir_data_value, phase);
              fprintf(file, " %f", value);
            }
            fprintf(file, "\n");
          }
          if (ReservoirDataValueDeltaContaminantPtrs(reservoir_data_value))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
              {
                indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                value = ReservoirDataValueContaminantFraction(reservoir_data_value, indx)
                        * fabs(ReservoirDataValueDeltaContaminantPtr(reservoir_data_value, indx))
                        / ReservoirDataPhysicalSize(reservoir_data_physical);
                fprintf(file, " %f", value);
              }
            }
            fprintf(file, "\n");
          }
          else if (ReservoirDataValueContaminantValues(reservoir_data_value))
          {
            for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
            {
              for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
              {
                indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
                value = ReservoirDataValueContaminantValue(reservoir_data_value, indx);
                fprintf(file, " %f", value);
              }
            }
            fprintf(file, "\n");
          }
        }
        if (ReservoirDataValueContaminantFractions(reservoir_data_value))
        {
          for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
          {
            for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
            {
              indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
              value = ReservoirDataValueContaminantFraction(reservoir_data_value, indx);
              fprintf(file, " %f", value);
            }
          }
          fprintf(file, "\n");
        }

        /* Write out the current reservoir statistics */
        reservoir_data_stat = ReservoirDataPressReservoirStat(reservoir_data, reservoir);
        if (ReservoirDataStatPhaseStats(reservoir_data_stat))
        {
          for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
          {
            stat = ReservoirDataStatPhaseStat(reservoir_data_stat, phase);
            fprintf(file, " %f", stat);
          }
          fprintf(file, "\n");
        }
        if (ReservoirDataStatSaturationStats(reservoir_data_stat))
        {
          for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
          {
            stat = ReservoirDataStatSaturationStat(reservoir_data_stat, phase);
            fprintf(file, " %f", stat);
          }
          fprintf(file, "\n");
        }
        if (ReservoirDataStatContaminantStats(reservoir_data_stat))
        {
          for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
          {
            for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
            {
              indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
              stat = ReservoirDataStatContaminantStat(reservoir_data_stat, indx);
              fprintf(file, " %f", stat);
            }
          }
          fprintf(file, "\n");
        }
        if (ReservoirDataStatDeltaPhases(reservoir_data_stat) && ReservoirDataStatDeltaContaminants(reservoir_data_stat))
        {
          for (phase = 0; phase < ReservoirDataNumPhases(reservoir_data); phase++)
          {
            for (concentration = 0; concentration < ReservoirDataNumContaminants(reservoir_data); concentration++)
            {
              indx = phase * ReservoirDataNumContaminants(reservoir_data) + concentration;
              if (ReservoirDataStatDeltaPhase(reservoir_data_stat, phase) == 0.0)
              {
                stat = 0.0;
              }
              else
              {
                stat = ReservoirDataStatDeltaContaminant(reservoir_data_stat, indx) / ReservoirDataStatDeltaPhase(reservoir_data_stat, phase);
              }
              fprintf(file, " %f", stat);
            }
          }
          fprintf(file, "\n");
        }
      }

      fclose(file);
    }
  }
}