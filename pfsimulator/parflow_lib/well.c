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

#include "parflow.h"

#include <string.h>


/*****************************************************************************
*
* The functions in this file are for manipulating the WellData structure
*   in ProblemData and work in conjuction with the WellPackage module.
*
* Because of the times things are called, the New function is twinky
* (it was basically put in to be symmetric with the New/Free paradigm
* used through out the code) and is invoked by SetProblemData.  The Alloc
* function actually allocates the data within the sub-structures and is
* invoked by the WellPackage (which has the data needed to compute the array
* sizes and such).  The Free is smart enough to clean up after both the New
* and Alloc functions and is called by SetProblemData.
*
*****************************************************************************/


/*--------------------------------------------------------------------------
 * NewWellData
 *--------------------------------------------------------------------------*/

WellData *NewWellData()
{
  WellData    *well_data;

  well_data = ctalloc(WellData, 1);

  WellDataNumPhases(well_data) = 0;
  WellDataNumContaminants(well_data) = 0;

  WellDataNumWells(well_data) = -1;

  WellDataTimeCycleData(well_data) = NULL;

  WellDataNumPressWells(well_data) = -1;
  WellDataPressWellPhysicals(well_data) = NULL;
  WellDataPressWellValues(well_data) = NULL;
  WellDataPressWellStats(well_data) = NULL;

  WellDataNumFluxWells(well_data) = -1;
  WellDataFluxWellPhysicals(well_data) = NULL;
  WellDataFluxWellValues(well_data) = NULL;
  WellDataFluxWellStats(well_data) = NULL;

  return well_data;
}


/*--------------------------------------------------------------------------
 * FreeWellData
 *--------------------------------------------------------------------------*/

void FreeWellData(
                  WellData *well_data)
{
  WellDataPhysical *well_data_physical;
  WellDataValue    *well_data_value;
  WellDataStat     *well_data_stat;
  int i, cycle_number, interval_division, interval_number;

  TimeCycleData   *time_cycle_data;

  if (well_data)
  {
    if (WellDataNumWells(well_data) > 0)
    {
      time_cycle_data = WellDataTimeCycleData(well_data);

      if (WellDataNumFluxWells(well_data) > 0)
      {
        for (i = 0; i < WellDataNumFluxWells(well_data); i++)
        {
          well_data_stat = WellDataFluxWellStat(well_data, i);
          if (WellDataStatDeltaPhases(well_data_stat))
          {
            tfree(WellDataStatDeltaPhases(well_data_stat));
          }
          if (WellDataStatPhaseStats(well_data_stat))
          {
            tfree(WellDataStatPhaseStats(well_data_stat));
          }
          if (WellDataStatDeltaSaturations(well_data_stat))
          {
            tfree(WellDataStatDeltaSaturations(well_data_stat));
          }
          if (WellDataStatSaturationStats(well_data_stat))
          {
            tfree(WellDataStatSaturationStats(well_data_stat));
          }
          if (WellDataStatDeltaContaminants(well_data_stat))
          {
            tfree(WellDataStatDeltaContaminants(well_data_stat));
          }
          if (WellDataStatContaminantStats(well_data_stat))
          {
            tfree(WellDataStatContaminantStats(well_data_stat));
          }
          tfree(well_data_stat);
        }
        if (WellDataFluxWellStats(well_data))
        {
          tfree(WellDataFluxWellStats(well_data));
        }
        for (i = 0; i < WellDataNumFluxWells(well_data); i++)
        {
          well_data_physical = WellDataFluxWellPhysical(well_data, i);
          cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
          interval_division = TimeCycleDataIntervalDivision(time_cycle_data, cycle_number);

          for (interval_number = 0; interval_number < interval_division; interval_number++)
          {
            well_data_value = WellDataFluxWellIntervalValue(well_data, i, interval_number);
            if (WellDataValuePhaseValues(well_data_value))
            {
              tfree(WellDataValuePhaseValues(well_data_value));
            }
            if (WellDataPhysicalAction(well_data_physical) == INJECTION_WELL)
            {
              if (WellDataValueSaturationValues(well_data_value))
              {
                tfree(WellDataValueSaturationValues(well_data_value));
              }
              if (WellDataValueContaminantValues(well_data_value))
              {
                tfree(WellDataValueContaminantValues(well_data_value));
              }
            }
            if (WellDataValueContaminantFractions(well_data_value))
            {
              tfree(WellDataValueContaminantFractions(well_data_value));
            }
            tfree(well_data_value);
          }
          if (WellDataFluxWellIntervalValues(well_data, i))
          {
            tfree(WellDataFluxWellIntervalValues(well_data, i));
          }
        }
        if (WellDataFluxWellValues(well_data))
        {
          tfree(WellDataFluxWellValues(well_data));
        }
        for (i = 0; i < WellDataNumFluxWells(well_data); i++)
        {
          well_data_physical = WellDataFluxWellPhysical(well_data, i);
          if (WellDataPhysicalName(well_data_physical))
          {
            tfree(WellDataPhysicalName(well_data_physical));
          }
          if (WellDataPhysicalSubgrid(well_data_physical))
          {
            FreeSubgrid(WellDataPhysicalSubgrid(well_data_physical));
          }
          tfree(well_data_physical);
        }
        if (WellDataFluxWellPhysicals(well_data))
        {
          tfree(WellDataFluxWellPhysicals(well_data));
        }
      }

      if (WellDataNumPressWells(well_data) > 0)
      {
        for (i = 0; i < WellDataNumPressWells(well_data); i++)
        {
          well_data_stat = WellDataPressWellStat(well_data, i);
          if (WellDataStatDeltaPhases(well_data_stat))
          {
            tfree(WellDataStatDeltaPhases(well_data_stat));
          }
          if (WellDataStatPhaseStats(well_data_stat))
          {
            tfree(WellDataStatPhaseStats(well_data_stat));
          }
          if (WellDataStatDeltaSaturations(well_data_stat))
          {
            tfree(WellDataStatDeltaSaturations(well_data_stat));
          }
          if (WellDataStatSaturationStats(well_data_stat))
          {
            tfree(WellDataStatSaturationStats(well_data_stat));
          }
          if (WellDataStatDeltaContaminants(well_data_stat))
          {
            tfree(WellDataStatDeltaContaminants(well_data_stat));
          }
          if (WellDataStatContaminantStats(well_data_stat))
          {
            tfree(WellDataStatContaminantStats(well_data_stat));
          }
          tfree(well_data_stat);
        }
        if (WellDataPressWellStats(well_data))
        {
          tfree(WellDataPressWellStats(well_data));
        }
        for (i = 0; i < WellDataNumPressWells(well_data); i++)
        {
          well_data_physical = WellDataPressWellPhysical(well_data, i);
          cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
          interval_division = TimeCycleDataIntervalDivision(time_cycle_data, cycle_number);

          for (interval_number = 0; interval_number < interval_division; interval_number++)
          {
            well_data_value = WellDataPressWellIntervalValue(well_data, i, interval_number);
            if (WellDataValuePhaseValues(well_data_value))
            {
              tfree(WellDataValuePhaseValues(well_data_value));
            }
            if (WellDataPhysicalAction(well_data_physical) == INJECTION_WELL)
            {
              if (WellDataValueSaturationValues(well_data_value))
              {
                tfree(WellDataValueSaturationValues(well_data_value));
              }
              if (WellDataValueContaminantValues(well_data_value))
              {
                tfree(WellDataValueContaminantValues(well_data_value));
              }
            }
            if (WellDataValueContaminantFractions(well_data_value))
            {
              tfree(WellDataValueContaminantFractions(well_data_value));
            }
            tfree(well_data_value);
          }
          if (WellDataPressWellIntervalValues(well_data, i))
          {
            tfree(WellDataPressWellIntervalValues(well_data, i));
          }
        }
        if (WellDataPressWellValues(well_data))
        {
          tfree(WellDataPressWellValues(well_data));
        }
        for (i = 0; i < WellDataNumPressWells(well_data); i++)
        {
          well_data_physical = WellDataPressWellPhysical(well_data, i);
          if (WellDataPhysicalName(well_data_physical))
          {
            tfree(WellDataPhysicalName(well_data_physical));
          }
          if (WellDataPhysicalSubgrid(well_data_physical))
          {
            FreeSubgrid(WellDataPhysicalSubgrid(well_data_physical));
          }
          tfree(well_data_physical);
        }
        if (WellDataPressWellPhysicals(well_data))
        {
          tfree(WellDataPressWellPhysicals(well_data));
        }
      }
      FreeTimeCycleData(time_cycle_data);
    }
    tfree(well_data);
  }
}


/*--------------------------------------------------------------------------
 * PrintWellData
 *--------------------------------------------------------------------------*/

void PrintWellData(
                   WellData *   well_data,
                   unsigned int print_mask)
{
  TimeCycleData    *time_cycle_data;

  WellDataPhysical *well_data_physical;
  WellDataValue    *well_data_value;
  WellDataStat     *well_data_stat;

  Subgrid          *subgrid;

  int cycle_number, interval_division, interval_number;
  int well, phase, concentration, indx;
  double value, stat;

  amps_Printf("Well Information\n");
  if (WellDataNumWells(well_data) == -1)
  {
    amps_Printf("Wells have not been setup.\n");
  }
  else if (WellDataNumWells(well_data) == 0)
  {
    amps_Printf("No Wells.\n");
  }
  else
  {
    time_cycle_data = WellDataTimeCycleData(well_data);

    PrintTimeCycleData(time_cycle_data);

    if (WellDataNumFluxWells(well_data) > 0)
    {
      amps_Printf("Info on Flux Wells :\n");
      for (well = 0; well < WellDataNumFluxWells(well_data); well++)
      {
        amps_Printf(" Flux Well Number : %02d\n", well);

        if ((print_mask & WELLDATA_PRINTPHYSICAL))
        {
          amps_Printf("  Well Physical Data :\n");
          well_data_physical = WellDataFluxWellPhysical(well_data, well);

          amps_Printf("   sequence number = %2d\n",
                      WellDataPhysicalNumber(well_data_physical));
          amps_Printf("   name = %s\n",
                      WellDataPhysicalName(well_data_physical));
          amps_Printf("   x_lower, y_lower, z_lower = %f %f %f\n",
                      WellDataPhysicalXLower(well_data_physical),
                      WellDataPhysicalYLower(well_data_physical),
                      WellDataPhysicalZLower(well_data_physical));
          amps_Printf("   x_upper, y_upper, z_upper = %f %f %f\n",
                      WellDataPhysicalXUpper(well_data_physical),
                      WellDataPhysicalYUpper(well_data_physical),
                      WellDataPhysicalZUpper(well_data_physical));
          amps_Printf("   diameter = %f\n",
                      WellDataPhysicalDiameter(well_data_physical));

          subgrid = WellDataPhysicalSubgrid(well_data_physical);
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
                      WellDataPhysicalSize(well_data_physical));
          amps_Printf("   action = %d\n",
                      WellDataPhysicalAction(well_data_physical));
          amps_Printf("   method = %d\n",
                      WellDataPhysicalMethod(well_data_physical));
        }

        if ((print_mask & WELLDATA_PRINTVALUES))
        {
          amps_Printf("  Well Values :\n");
          well_data_physical = WellDataFluxWellPhysical(well_data, well);
          cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
          interval_division = TimeCycleDataIntervalDivision(time_cycle_data, cycle_number);

          for (interval_number = 0; interval_number < interval_division; interval_number++)
          {
            amps_Printf("  Value[%2d] :\n", interval_number);
            well_data_value = WellDataFluxWellIntervalValue(well_data, well, interval_number);

            if (WellDataValuePhaseValues(well_data_value))
            {
              for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
              {
                value = WellDataValuePhaseValue(well_data_value, phase);
                amps_Printf("   value for phase %01d = %f\n", phase, value);
              }
            }
            else
            {
              amps_Printf("   no phase values present.\n");
            }
            if (WellDataPhysicalAction(well_data_physical) == INJECTION_WELL)
            {
              if (WellDataValueSaturationValues(well_data_value))
              {
                for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
                {
                  value = WellDataValueSaturationValue(well_data_value, phase);
                  amps_Printf("   s_bar[%01d] = %f\n", phase, value);
                }
              }
              else
              {
                amps_Printf("   no saturation values present.\n");
              }
              if (WellDataValueDeltaSaturationPtrs(well_data_value))
              {
                for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
                {
                  value = WellDataValueDeltaSaturationPtr(well_data_value, phase);
                  amps_Printf("   delta_saturations[%01d] = %f\n", phase, value);
                }
              }
              else
              {
                amps_Printf("   no saturation delta values present.\n");
              }
              if (WellDataValueContaminantValues(well_data_value))
              {
                for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
                {
                  for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
                  {
                    indx = phase * WellDataNumContaminants(well_data) + concentration;
                    value = WellDataValueContaminantValue(well_data_value, indx);
                    amps_Printf("   c_bar[%01d][%02d] = %f\n", phase, concentration, value);
                  }
                }
              }
              else
              {
                amps_Printf("   no contaminant values present.\n");
              }
              if (WellDataValueDeltaContaminantPtrs(well_data_value))
              {
                for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
                {
                  for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
                  {
                    indx = phase * WellDataNumContaminants(well_data) + concentration;
                    value = WellDataValueDeltaContaminantPtr(well_data_value, indx);
                    amps_Printf("  delta_concentration[%01d][%02d] = %f\n", phase, concentration, value);
                  }
                }
              }
              else
              {
                amps_Printf("   no concentration delta values present.\n");
              }
            }
            if (WellDataValueContaminantFractions(well_data_value))
            {
              for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
              {
                for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
                {
                  indx = phase * WellDataNumContaminants(well_data) + concentration;
                  value = WellDataValueContaminantFraction(well_data_value, indx);
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

        if ((print_mask & WELLDATA_PRINTSTATS))
        {
          amps_Printf("  Well Stats :\n");
          well_data_stat = WellDataFluxWellStat(well_data, well);
          if (WellDataStatDeltaPhases(well_data_stat))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              stat = WellDataStatDeltaPhase(well_data_stat, phase);
              amps_Printf("  delta_p[%01d] = %f\n", phase, stat);
            }
          }
          if (WellDataStatPhaseStats(well_data_stat))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              stat = WellDataStatPhaseStat(well_data_stat, phase);
              amps_Printf("  phase[%01d] = %f\n", phase, stat);
            }
          }
          if (WellDataStatDeltaSaturations(well_data_stat))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              stat = WellDataStatDeltaSaturation(well_data_stat, phase);
              amps_Printf("  delta_s[%01d] = %f\n", phase, stat);
            }
          }
          if (WellDataStatSaturationStats(well_data_stat))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              stat = WellDataStatSaturationStat(well_data_stat, phase);
              amps_Printf("  saturation[%01d] = %f\n", phase, stat);
            }
          }
          if (WellDataStatDeltaContaminants(well_data_stat))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
              {
                indx = phase * WellDataNumContaminants(well_data) + concentration;
                stat = WellDataStatDeltaContaminant(well_data_stat, indx);
                amps_Printf("  delta_c[%01d][%02d] = %f\n", phase, concentration, stat);
              }
            }
          }
          if (WellDataStatContaminantStats(well_data_stat))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
              {
                indx = phase * WellDataNumContaminants(well_data) + concentration;
                stat = WellDataStatContaminantStat(well_data_stat, indx);
                amps_Printf("  concentration[%01d][%02d] = %f\n", phase, concentration, stat);
              }
            }
          }
        }
      }
    }
    else
    {
      amps_Printf("No Flux Wells.\n");
    }

    if (WellDataNumPressWells(well_data) > 0)
    {
      amps_Printf("Info on Pressure Wells :\n");
      for (well = 0; well < WellDataNumPressWells(well_data); well++)
      {
        amps_Printf(" Pressure Well Number : %2d\n", well);
        if ((print_mask & WELLDATA_PRINTPHYSICAL))
        {
          amps_Printf("  Well Physical Data :\n");
          well_data_physical = WellDataPressWellPhysical(well_data, well);

          amps_Printf("   sequence number = %2d\n",
                      WellDataPhysicalNumber(well_data_physical));
          amps_Printf("   name = %s\n",
                      WellDataPhysicalName(well_data_physical));
          amps_Printf("   x_lower, y_lower, z_lower = %f %f %f\n",
                      WellDataPhysicalXLower(well_data_physical),
                      WellDataPhysicalYLower(well_data_physical),
                      WellDataPhysicalZLower(well_data_physical));
          amps_Printf("   x_upper, y_upper, z_upper = %f %f %f\n",
                      WellDataPhysicalXUpper(well_data_physical),
                      WellDataPhysicalYUpper(well_data_physical),
                      WellDataPhysicalZUpper(well_data_physical));
          amps_Printf("   diameter = %f\n",
                      WellDataPhysicalDiameter(well_data_physical));

          subgrid = WellDataPhysicalSubgrid(well_data_physical);
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
                      WellDataPhysicalSize(well_data_physical));
          amps_Printf("   action = %d\n",
                      WellDataPhysicalAction(well_data_physical));
          amps_Printf("   method = %d\n",
                      WellDataPhysicalMethod(well_data_physical));
        }

        if ((print_mask & WELLDATA_PRINTVALUES))
        {
          amps_Printf("  Well Values :\n");
          well_data_physical = WellDataPressWellPhysical(well_data, well);
          cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
          interval_division = TimeCycleDataIntervalDivision(time_cycle_data, cycle_number);

          for (interval_number = 0; interval_number < interval_division; interval_number++)
          {
            amps_Printf("  Value[%2d] :\n", interval_number);

            well_data_value = WellDataPressWellIntervalValue(well_data, well, interval_number);

            if (WellDataValuePhaseValues(well_data_value))
            {
              phase = 0;
              {
                value = WellDataValuePhaseValue(well_data_value, phase);
                amps_Printf("   value for phase %01d = %f\n", phase, value);
              }
            }
            else
            {
              amps_Printf("   no phase values present.\n");
            }
            if (WellDataPhysicalAction(well_data_physical) == INJECTION_WELL)
            {
              if (WellDataValueSaturationValues(well_data_value))
              {
                for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
                {
                  value = WellDataValueSaturationValue(well_data_value, phase);
                  amps_Printf("   s_bar[%01d] = %f\n", phase, value);
                }
              }
              else
              {
                amps_Printf("   no saturation values present.\n");
              }
              if (WellDataValueDeltaSaturationPtrs(well_data_value))
              {
                for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
                {
                  value = WellDataValueDeltaSaturationPtr(well_data_value, phase);
                  amps_Printf("   delta_saturations[%01d] = %f\n", phase, value);
                }
              }
              else
              {
                amps_Printf("   no saturation delta values present.\n");
              }
              if (WellDataValueContaminantValues(well_data_value))
              {
                for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
                {
                  for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
                  {
                    indx = phase * WellDataNumContaminants(well_data) + concentration;
                    value = WellDataValueContaminantValue(well_data_value, indx);
                    amps_Printf("   c_bar[%01d][%02d] = %f\n", phase, concentration, value);
                  }
                }
              }
              else
              {
                amps_Printf("   no component values present.\n");
              }
              if (WellDataValueDeltaContaminantPtrs(well_data_value))
              {
                for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
                {
                  for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
                  {
                    indx = phase * WellDataNumContaminants(well_data) + concentration;
                    value = WellDataValueDeltaContaminantPtr(well_data_value, indx);
                    amps_Printf("  delta_concentration[%01d][%02d] = %f\n", phase, concentration, value);
                  }
                }
              }
              else
              {
                amps_Printf("   no concentration delta values present.\n");
              }
            }
            if (WellDataValueContaminantFractions(well_data_value))
            {
              for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
              {
                for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
                {
                  indx = phase * WellDataNumContaminants(well_data) + concentration;
                  value = WellDataValueContaminantFraction(well_data_value, indx);
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

        if ((print_mask & WELLDATA_PRINTSTATS))
        {
          amps_Printf("  Well Stats :\n");
          well_data_stat = WellDataPressWellStat(well_data, well);
          if (WellDataStatDeltaPhases(well_data_stat))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              stat = WellDataStatDeltaPhase(well_data_stat, phase);
              amps_Printf("  delta_p[%01d] = %f\n", phase, stat);
            }
          }
          if (WellDataStatPhaseStats(well_data_stat))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              stat = WellDataStatPhaseStat(well_data_stat, phase);
              amps_Printf("  phase[%01d] = %f\n", phase, stat);
            }
          }
          if (WellDataStatDeltaSaturations(well_data_stat))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              stat = WellDataStatDeltaSaturation(well_data_stat, phase);
              amps_Printf("  delta_s[%01d] = %f\n", phase, stat);
            }
          }
          if (WellDataStatSaturationStats(well_data_stat))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              stat = WellDataStatSaturationStat(well_data_stat, phase);
              amps_Printf("  saturation[%01d] = %f\n", phase, stat);
            }
          }
          if (WellDataStatDeltaContaminants(well_data_stat))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
              {
                indx = phase * WellDataNumContaminants(well_data) + concentration;
                stat = WellDataStatDeltaContaminant(well_data_stat, indx);
                amps_Printf("  delta_c[%01d][%02d] = %f\n", phase, concentration, stat);
              }
            }
          }
          if (WellDataStatContaminantStats(well_data_stat))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
              {
                indx = phase * WellDataNumContaminants(well_data) + concentration;
                stat = WellDataStatContaminantStat(well_data_stat, indx);
                amps_Printf("  concentration[%01d][%02d] = %f\n", phase, concentration, stat);
              }
            }
          }
        }
      }
    }
    else
    {
      amps_Printf("No Pressure Wells.\n");
    }
  }
}

/*--------------------------------------------------------------------------
 * WriteWells
 *--------------------------------------------------------------------------*/

void WriteWells(
                char *    file_prefix,
                Problem * problem,
                WellData *well_data,
                double    time,
                int       write_header)
{
  TimeCycleData    *time_cycle_data;
  WellDataPhysical *well_data_physical;
  WellDataValue    *well_data_value;
  WellDataStat     *well_data_stat;

  Subgrid          *subgrid;

  int cycle_number, interval_number;
  int well, phase, concentration, indx;
  double value, stat;

  FILE             *file;

  char file_suffix[7] = "wells";
  char filename[255];

  int p;

  if (WellDataNumWells(well_data) > 0)
  {
    p = amps_Rank(amps_CommWorld);

    time_cycle_data = WellDataTimeCycleData(well_data);

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
                WellDataNumPhases(well_data),
                WellDataNumContaminants(well_data),
                WellDataNumWells(well_data));

        for (well = 0; well < WellDataNumFluxWells(well_data); well++)
        {
          well_data_physical = WellDataFluxWellPhysical(well_data, well);

          fprintf(file, "%2d\n", WellDataPhysicalNumber(well_data_physical));

          fprintf(file, "%d\n", (int)strlen(WellDataPhysicalName(well_data_physical)));
          fprintf(file, "%s\n", WellDataPhysicalName(well_data_physical));

          fprintf(file, "%f %f %f %f %f %f %f\n",
                  WellDataPhysicalXLower(well_data_physical),
                  WellDataPhysicalYLower(well_data_physical),
                  WellDataPhysicalZLower(well_data_physical),
                  WellDataPhysicalXUpper(well_data_physical),
                  WellDataPhysicalYUpper(well_data_physical),
                  WellDataPhysicalZUpper(well_data_physical),
                  WellDataPhysicalDiameter(well_data_physical));
          fprintf(file, "1 %1d %1d\n",
                  WellDataPhysicalAction(well_data_physical),
                  WellDataPhysicalMethod(well_data_physical));
        }
        for (well = 0; well < WellDataNumPressWells(well_data); well++)
        {
          well_data_physical = WellDataPressWellPhysical(well_data, well);

          fprintf(file, "%2d\n", WellDataPhysicalNumber(well_data_physical));

          fprintf(file, "%d\n", (int)strlen(WellDataPhysicalName(well_data_physical)));
          fprintf(file, "%s\n", WellDataPhysicalName(well_data_physical));

          fprintf(file, "%f %f %f %f %f %f %f\n",
                  WellDataPhysicalXLower(well_data_physical),
                  WellDataPhysicalYLower(well_data_physical),
                  WellDataPhysicalZLower(well_data_physical),
                  WellDataPhysicalXUpper(well_data_physical),
                  WellDataPhysicalYUpper(well_data_physical),
                  WellDataPhysicalZUpper(well_data_physical),
                  WellDataPhysicalDiameter(well_data_physical));
          fprintf(file, "0 %1d %1d\n",
                  WellDataPhysicalAction(well_data_physical),
                  WellDataPhysicalMethod(well_data_physical));
        }
      }

      fprintf(file, "%f\n", time);

      for (well = 0; well < WellDataNumFluxWells(well_data); well++)
      {
        /* Write out important current physical data */
        well_data_physical = WellDataFluxWellPhysical(well_data, well);
        fprintf(file, "%2d\n", WellDataPhysicalNumber(well_data_physical));
        subgrid = WellDataPhysicalSubgrid(well_data_physical);
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

        /* Write out the current well values */
        cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
        interval_number = TimeCycleDataComputeIntervalNumber(problem, time, time_cycle_data, cycle_number);

        well_data_value = WellDataFluxWellIntervalValue(well_data, well, interval_number);

        if (WellDataValuePhaseValues(well_data_value))
        {
          for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
          {
            value = WellDataValuePhaseValue(well_data_value, phase);
            fprintf(file, " %f", value);
          }
          fprintf(file, "\n");
        }
        if (WellDataPhysicalAction(well_data_physical) == INJECTION_WELL)
        {
          if (WellDataValueDeltaSaturationPtrs(well_data_value))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              value = WellDataValueDeltaSaturationPtr(well_data_value, phase);
              fprintf(file, " %f", value);
            }
            fprintf(file, "\n");
          }
          else if (WellDataValueSaturationValues(well_data_value))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              value = WellDataValueSaturationValue(well_data_value, phase);
              fprintf(file, " %f", value);
            }
            fprintf(file, "\n");
          }
          if (WellDataValueDeltaContaminantPtrs(well_data_value))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
              {
                indx = phase * WellDataNumContaminants(well_data) + concentration;
                value = WellDataValueContaminantFraction(well_data_value, indx)
                        * fabs(WellDataValueDeltaContaminantPtr(well_data_value, indx))
                        / WellDataPhysicalSize(well_data_physical);
                fprintf(file, " %f", value);
              }
            }
            fprintf(file, "\n");
          }
          else if (WellDataValueContaminantValues(well_data_value))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
              {
                indx = phase * WellDataNumContaminants(well_data) + concentration;
                value = WellDataValueContaminantValue(well_data_value, indx);
                fprintf(file, " %f", value);
              }
            }
            fprintf(file, "\n");
          }
        }
        if (WellDataValueContaminantFractions(well_data_value))
        {
          for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
          {
            for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
            {
              indx = phase * WellDataNumContaminants(well_data) + concentration;
              value = WellDataValueContaminantFraction(well_data_value, indx);
              fprintf(file, " %f", value);
            }
          }
          fprintf(file, "\n");
        }

        /* Write out the current well statistics */
        well_data_stat = WellDataFluxWellStat(well_data, well);
        if (WellDataStatPhaseStats(well_data_stat))
        {
          for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
          {
            stat = WellDataStatPhaseStat(well_data_stat, phase);
            fprintf(file, " %f", stat);
          }
          fprintf(file, "\n");
        }
        if (WellDataStatSaturationStats(well_data_stat))
        {
          for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
          {
            stat = WellDataStatSaturationStat(well_data_stat, phase);
            fprintf(file, " %f", stat);
          }
          fprintf(file, "\n");
        }
        if (WellDataStatContaminantStats(well_data_stat))
        {
          for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
          {
            for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
            {
              indx = phase * WellDataNumContaminants(well_data) + concentration;
              stat = WellDataStatContaminantStat(well_data_stat, indx);
              fprintf(file, " %f", stat);
            }
          }
          fprintf(file, "\n");
        }
        if (WellDataStatDeltaPhases(well_data_stat) && WellDataStatDeltaContaminants(well_data_stat))
        {
          for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
          {
            for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
            {
              indx = phase * WellDataNumContaminants(well_data) + concentration;
              if (WellDataStatDeltaPhase(well_data_stat, phase) == 0.0)
              {
                stat = 0.0;
              }
              else
              {
                stat = WellDataStatDeltaContaminant(well_data_stat, indx) / WellDataStatDeltaPhase(well_data_stat, phase);
              }
              fprintf(file, " %f", stat);
            }
          }
          fprintf(file, "\n");
        }
      }

      for (well = 0; well < WellDataNumPressWells(well_data); well++)
      {
        /* Write out important current physical data */
        well_data_physical = WellDataPressWellPhysical(well_data, well);
        fprintf(file, "%2d\n", WellDataPhysicalNumber(well_data_physical));
        subgrid = WellDataPhysicalSubgrid(well_data_physical);
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

        /* Write out the current well values */
        cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
        interval_number = TimeCycleDataComputeIntervalNumber(problem,
                                                             time, time_cycle_data, cycle_number);

        well_data_value = WellDataPressWellIntervalValue(well_data, well,
                                                         interval_number);

        if (WellDataValuePhaseValues(well_data_value))
        {
          phase = 0;
          {
            value = WellDataValuePhaseValue(well_data_value, phase);
            fprintf(file, " %f", value);
          }
          fprintf(file, "\n");
        }
        if (WellDataPhysicalAction(well_data_physical) == INJECTION_WELL)
        {
          if (WellDataValueDeltaSaturationPtrs(well_data_value))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              value = WellDataValueDeltaSaturationPtr(well_data_value, phase);
              fprintf(file, " %f", value);
            }
            fprintf(file, "\n");
          }
          else if (WellDataValueSaturationValues(well_data_value))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              value = WellDataValueSaturationValue(well_data_value, phase);
              fprintf(file, " %f", value);
            }
            fprintf(file, "\n");
          }
          if (WellDataValueDeltaContaminantPtrs(well_data_value))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
              {
                indx = phase * WellDataNumContaminants(well_data) + concentration;
                value = WellDataValueContaminantFraction(well_data_value, indx)
                        * fabs(WellDataValueDeltaContaminantPtr(well_data_value, indx))
                        / WellDataPhysicalSize(well_data_physical);
                fprintf(file, " %f", value);
              }
            }
            fprintf(file, "\n");
          }
          else if (WellDataValueContaminantValues(well_data_value))
          {
            for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
            {
              for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
              {
                indx = phase * WellDataNumContaminants(well_data) + concentration;
                value = WellDataValueContaminantValue(well_data_value, indx);
                fprintf(file, " %f", value);
              }
            }
            fprintf(file, "\n");
          }
        }
        if (WellDataValueContaminantFractions(well_data_value))
        {
          for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
          {
            for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
            {
              indx = phase * WellDataNumContaminants(well_data) + concentration;
              value = WellDataValueContaminantFraction(well_data_value, indx);
              fprintf(file, " %f", value);
            }
          }
          fprintf(file, "\n");
        }

        /* Write out the current well statistics */
        well_data_stat = WellDataPressWellStat(well_data, well);
        if (WellDataStatPhaseStats(well_data_stat))
        {
          for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
          {
            stat = WellDataStatPhaseStat(well_data_stat, phase);
            fprintf(file, " %f", stat);
          }
          fprintf(file, "\n");
        }
        if (WellDataStatSaturationStats(well_data_stat))
        {
          for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
          {
            stat = WellDataStatSaturationStat(well_data_stat, phase);
            fprintf(file, " %f", stat);
          }
          fprintf(file, "\n");
        }
        if (WellDataStatContaminantStats(well_data_stat))
        {
          for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
          {
            for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
            {
              indx = phase * WellDataNumContaminants(well_data) + concentration;
              stat = WellDataStatContaminantStat(well_data_stat, indx);
              fprintf(file, " %f", stat);
            }
          }
          fprintf(file, "\n");
        }
        if (WellDataStatDeltaPhases(well_data_stat) && WellDataStatDeltaContaminants(well_data_stat))
        {
          for (phase = 0; phase < WellDataNumPhases(well_data); phase++)
          {
            for (concentration = 0; concentration < WellDataNumContaminants(well_data); concentration++)
            {
              indx = phase * WellDataNumContaminants(well_data) + concentration;
              if (WellDataStatDeltaPhase(well_data_stat, phase) == 0.0)
              {
                stat = 0.0;
              }
              else
              {
                stat = WellDataStatDeltaContaminant(well_data_stat, indx) / WellDataStatDeltaPhase(well_data_stat, phase);
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
