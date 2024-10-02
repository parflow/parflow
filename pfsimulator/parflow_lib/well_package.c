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

#define PRESSURE_WELL   0
#define FLUX_WELL       1

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int num_phases;
  int num_contaminants;

  /* well info */
  int num_units;
  int num_wells;
  int num_press_wells;
  int num_flux_wells;

  int       *type;
  void     **data;

  /* Timing Cycle information */
  int num_cycles;

  int       *interval_divisions;
  int      **intervals;
  int       *repeat_counts;

  NameArray well_names;
} PublicXtra;

typedef void InstanceXtra;

typedef struct {
  char    *name;
  int action;
  int mechanism;
  double xlocation;
  double ylocation;
  double z_lower, z_upper;
  int method;
  int cycle_number;
  double **phase_values;
  double **saturation_values;
  double **contaminant_values;
} Type0;                      /* basic vertical well */

typedef struct {
  char    *name;
  int mechanism_ext;
  int mechanism_inj;
  double xlocation;
  double ylocation;
  double z_lower_ext, z_upper_ext;
  double z_lower_inj, z_upper_inj;
  int method_ext;
  int method_inj;
  int cycle_number;
  double **phase_values_ext;
  double **phase_values_inj;
  double **contaminant_fractions;
} Type1;                      /* basic vertical well, recirculating */

/*--------------------------------------------------------------------------
 * WellPackage
 *--------------------------------------------------------------------------*/

void         WellPackage(
                         ProblemData *problem_data)
{
  PFModule         *this_module = ThisPFModule;
  PublicXtra       *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0            *dummy0;
  Type1            *dummy1;

  Subgrid          *new_subgrid;

  TimeCycleData    *time_cycle_data;

  WellData         *well_data = ProblemDataWellData(problem_data);
  WellDataPhysical *well_data_physical;
  WellDataValue    *well_data_value;
  WellDataStat     *well_data_stat = NULL;

  int i, sequence_number, phase, contaminant, indx, press_well, flux_well;

  int ix, iy;
  int iz_lower, iz_upper;
  int nx, ny, nz;
  double dx, dy, dz;
  int rx, ry, rz;
  int process;
  int well_action, action, mechanism, method;
  int cycle_length, cycle_number, interval_division, interval_number;

  double          **phase_values;
  double subgrid_volume;
  double x_lower, x_upper, y_lower, y_upper,
         z_lower, z_upper;

  /* Allocate the well data */
  WellDataNumPhases(well_data) = (public_xtra->num_phases);
  WellDataNumContaminants(well_data) = (public_xtra->num_contaminants);

  WellDataNumWells(well_data) = (public_xtra->num_wells);

  if ((public_xtra->num_wells) > 0)
  {
    WellDataNumPressWells(well_data) = (public_xtra->num_press_wells);
    if ((public_xtra->num_press_wells) > 0)
    {
      WellDataPressWellPhysicals(well_data) = ctalloc(WellDataPhysical *, (public_xtra->num_press_wells));
      WellDataPressWellValues(well_data) = ctalloc(WellDataValue * *, (public_xtra->num_press_wells));
      WellDataPressWellStats(well_data) = ctalloc(WellDataStat *, (public_xtra->num_press_wells));
    }

    WellDataNumFluxWells(well_data) = (public_xtra->num_flux_wells);
    if ((public_xtra->num_flux_wells) > 0)
    {
      WellDataFluxWellPhysicals(well_data) = ctalloc(WellDataPhysical *, (public_xtra->num_flux_wells));
      WellDataFluxWellValues(well_data) = ctalloc(WellDataValue * *, (public_xtra->num_flux_wells));
      WellDataFluxWellStats(well_data) = ctalloc(WellDataStat *, (public_xtra->num_flux_wells));
    }
  }

  press_well = 0;
  flux_well = 0;
  sequence_number = 0;

  if ((public_xtra->num_units) > 0)
  {
    /* Load the time cycle data */
    time_cycle_data = NewTimeCycleData((public_xtra->num_cycles), (public_xtra->interval_divisions));

    for (cycle_number = 0; cycle_number < (public_xtra->num_cycles); cycle_number++)
    {
      TimeCycleDataIntervalDivision(time_cycle_data, cycle_number) = (public_xtra->interval_divisions[cycle_number]);
      cycle_length = 0;
      for (interval_number = 0; interval_number < (public_xtra->interval_divisions[cycle_number]); interval_number++)
      {
        cycle_length += (public_xtra->intervals[cycle_number])[interval_number];
        TimeCycleDataInterval(time_cycle_data, cycle_number, interval_number) = (public_xtra->intervals[cycle_number])[interval_number];
      }
      TimeCycleDataRepeatCount(time_cycle_data, cycle_number) = (public_xtra->repeat_counts[cycle_number]);
      TimeCycleDataCycleLength(time_cycle_data, cycle_number) = cycle_length;
    }

    WellDataTimeCycleData(well_data) = time_cycle_data;

    /* Load the well data */
    for (i = 0; i < (public_xtra->num_units); i++)
    {
      switch ((public_xtra->type[i]))
      {
        case 0:
        {
          dummy0 = (Type0*)(public_xtra->data[i]);

          ix = IndexSpaceX((dummy0->xlocation), 0);
          iy = IndexSpaceY((dummy0->ylocation), 0);
          iz_lower = IndexSpaceZ((dummy0->z_lower), 0);
          iz_upper = IndexSpaceZ((dummy0->z_upper), 0);

          nx = 1;
          ny = 1;
          nz = iz_upper - iz_lower + 1;

          rx = 0;
          ry = 0;
          rz = 0;

          process = amps_Rank(amps_CommWorld);

          new_subgrid = NewSubgrid(ix, iy, iz_lower,
                                   nx, ny, nz,
                                   rx, ry, rz,
                                   process);

          dx = SubgridDX(new_subgrid);
          dy = SubgridDY(new_subgrid);
          dz = SubgridDZ(new_subgrid);

          subgrid_volume = (nx * dx) * (ny * dy) * (nz * dz);

          if ((dummy0->mechanism) == PRESSURE_WELL)
          {
            /* Put in physical data for this well */
            well_data_physical = ctalloc(WellDataPhysical, 1);
            WellDataPhysicalNumber(well_data_physical) = sequence_number;
            WellDataPhysicalName(well_data_physical) = ctalloc(char, strlen((dummy0->name)) + 1);
            strcpy(WellDataPhysicalName(well_data_physical), (dummy0->name));
            WellDataPhysicalXLower(well_data_physical) = (dummy0->xlocation);
            WellDataPhysicalYLower(well_data_physical) = (dummy0->ylocation);
            WellDataPhysicalZLower(well_data_physical) = (dummy0->z_lower);
            WellDataPhysicalXUpper(well_data_physical) = (dummy0->xlocation);
            WellDataPhysicalYUpper(well_data_physical) = (dummy0->ylocation);
            WellDataPhysicalZUpper(well_data_physical) = (dummy0->z_upper);
            WellDataPhysicalDiameter(well_data_physical) = pfmin(dx, dy);
            WellDataPhysicalSubgrid(well_data_physical) = new_subgrid;
            WellDataPhysicalSize(well_data_physical) = subgrid_volume;
            WellDataPhysicalAction(well_data_physical) = (dummy0->action);
            WellDataPhysicalMethod(well_data_physical) = (dummy0->method);
            WellDataPhysicalCycleNumber(well_data_physical) = (dummy0->cycle_number);
            WellDataPhysicalAveragePermeabilityX(well_data_physical) = 0.0;
            WellDataPhysicalAveragePermeabilityY(well_data_physical) = 0.0;
            WellDataPhysicalAveragePermeabilityZ(well_data_physical) = 0.0;
            WellDataPressWellPhysical(well_data, press_well) = well_data_physical;

            /* Put in values for this well */
            interval_division = TimeCycleDataIntervalDivision(time_cycle_data, WellDataPhysicalCycleNumber(well_data_physical));
            WellDataPressWellIntervalValues(well_data, press_well) = ctalloc(WellDataValue *, interval_division);
            for (interval_number = 0; interval_number < interval_division; interval_number++)
            {
              well_data_value = ctalloc(WellDataValue, 1);

              WellDataValuePhaseValues(well_data_value) = ctalloc(double, 1);
              phase = 0;
              {
                WellDataValuePhaseValue(well_data_value, phase) = ((dummy0->phase_values[interval_number])[phase]);
              }
              if ((dummy0->action) == INJECTION_WELL)
              {
                WellDataValueSaturationValues(well_data_value) = ctalloc(double, (public_xtra->num_phases));
                for (phase = 0; phase < (public_xtra->num_phases); phase++)
                {
                  WellDataValueSaturationValue(well_data_value, phase) = ((dummy0->saturation_values[interval_number])[phase]);
                }
                WellDataValueContaminantValues(well_data_value) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
                for (phase = 0; phase < (public_xtra->num_phases); phase++)
                {
                  for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                  {
                    indx = phase * (public_xtra->num_contaminants) + contaminant;
                    WellDataValueContaminantValue(well_data_value, indx) = ((dummy0->contaminant_values[interval_number])[indx]);
                  }
                }
                WellDataValueContaminantFractions(well_data_value) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
                for (phase = 0; phase < (public_xtra->num_phases); phase++)
                {
                  for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                  {
                    indx = phase * (public_xtra->num_contaminants) + contaminant;
                    WellDataValueContaminantFraction(well_data_value, indx) = 1.0;
                  }
                }
              }
              else
              {
                WellDataValueContaminantFractions(well_data_value) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
                for (phase = 0; phase < (public_xtra->num_phases); phase++)
                {
                  for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                  {
                    indx = phase * (public_xtra->num_contaminants) + contaminant;
                    WellDataValueContaminantFraction(well_data_value, indx) = 1.0;
                  }
                }
              }
              WellDataPressWellIntervalValue(well_data, press_well, interval_number) = well_data_value;
            }

            /* Put in informational statistics for this well */
            well_data_stat = ctalloc(WellDataStat, 1);
            WellDataStatDeltaPhases(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
            for (phase = 0; phase < (public_xtra->num_phases); phase++)
            {
              WellDataStatDeltaPhase(well_data_stat, phase) = 0.0;
            }
            WellDataStatPhaseStats(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
            for (phase = 0; phase < (public_xtra->num_phases); phase++)
            {
              WellDataStatPhaseStat(well_data_stat, phase) = 0.0;
            }
            WellDataStatDeltaSaturations(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
            for (phase = 0; phase < (public_xtra->num_phases); phase++)
            {
              WellDataStatDeltaSaturation(well_data_stat, phase) = 0.0;
            }
            WellDataStatSaturationStats(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
            for (phase = 0; phase < (public_xtra->num_phases); phase++)
            {
              WellDataStatSaturationStat(well_data_stat, phase) = 0.0;
            }
            WellDataStatDeltaContaminants(well_data_stat) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
            for (phase = 0; phase < (public_xtra->num_phases); phase++)
            {
              for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
              {
                indx = phase * (public_xtra->num_contaminants) + contaminant;
                WellDataStatDeltaContaminant(well_data_stat, indx) = 0.0;
              }
            }
            WellDataStatContaminantStats(well_data_stat) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
            for (phase = 0; phase < (public_xtra->num_phases); phase++)
            {
              for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
              {
                indx = phase * (public_xtra->num_contaminants) + contaminant;
                WellDataStatContaminantStat(well_data_stat, indx) = 0.0;
              }
            }
            WellDataPressWellStat(well_data, press_well) = well_data_stat;

            press_well++;
          }
          else if ((dummy0->mechanism) == FLUX_WELL)
          {
            well_data_physical = ctalloc(WellDataPhysical, 1);
            WellDataPhysicalNumber(well_data_physical) = sequence_number;
            WellDataPhysicalName(well_data_physical) = ctalloc(char, strlen((dummy0->name)) + 1);
            strcpy(WellDataPhysicalName(well_data_physical), (dummy0->name));
            WellDataPhysicalXLower(well_data_physical) = (dummy0->xlocation);
            WellDataPhysicalYLower(well_data_physical) = (dummy0->ylocation);
            WellDataPhysicalZLower(well_data_physical) = (dummy0->z_lower);
            WellDataPhysicalXUpper(well_data_physical) = (dummy0->xlocation);
            WellDataPhysicalYUpper(well_data_physical) = (dummy0->ylocation);
            WellDataPhysicalZUpper(well_data_physical) = (dummy0->z_upper);
            WellDataPhysicalDiameter(well_data_physical) = pfmin(dx, dy);
            WellDataPhysicalSubgrid(well_data_physical) = new_subgrid;
            WellDataPhysicalSize(well_data_physical) = subgrid_volume;
            WellDataPhysicalAction(well_data_physical) = (dummy0->action);
            WellDataPhysicalMethod(well_data_physical) = (dummy0->method);
            WellDataPhysicalCycleNumber(well_data_physical) = (dummy0->cycle_number);
            WellDataPhysicalAveragePermeabilityX(well_data_physical) = 0.0;
            WellDataPhysicalAveragePermeabilityY(well_data_physical) = 0.0;
            WellDataPhysicalAveragePermeabilityZ(well_data_physical) = 0.0;
            WellDataFluxWellPhysical(well_data, flux_well) = well_data_physical;

            /* Put in values for this well */
            interval_division = TimeCycleDataIntervalDivision(time_cycle_data, WellDataPhysicalCycleNumber(well_data_physical));
            WellDataFluxWellIntervalValues(well_data, flux_well) = ctalloc(WellDataValue *, interval_division);
            for (interval_number = 0; interval_number < interval_division; interval_number++)
            {
              well_data_value = ctalloc(WellDataValue, 1);

              WellDataValuePhaseValues(well_data_value) = ctalloc(double, (public_xtra->num_phases));
              for (phase = 0; phase < (public_xtra->num_phases); phase++)
              {
                WellDataValuePhaseValue(well_data_value, phase) = ((dummy0->phase_values[interval_number])[phase]);
              }
              if ((dummy0->action) == INJECTION_WELL)
              {
                WellDataValueSaturationValues(well_data_value) = ctalloc(double, (public_xtra->num_phases));
                for (phase = 0; phase < (public_xtra->num_phases); phase++)
                {
                  WellDataValueSaturationValue(well_data_value, phase) = ((dummy0->saturation_values[interval_number])[phase]);
                }
                WellDataValueContaminantValues(well_data_value) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
                for (phase = 0; phase < (public_xtra->num_phases); phase++)
                {
                  for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                  {
                    indx = phase * (public_xtra->num_contaminants) + contaminant;
                    WellDataValueContaminantValue(well_data_value, indx) = ((dummy0->contaminant_values[interval_number])[indx]);
                  }
                }
                WellDataValueContaminantFractions(well_data_value) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
                for (phase = 0; phase < (public_xtra->num_phases); phase++)
                {
                  for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                  {
                    indx = phase * (public_xtra->num_contaminants) + contaminant;
                    WellDataValueContaminantFraction(well_data_value, indx) = 1.0;
                  }
                }
              }
              else
              {
                WellDataValueContaminantFractions(well_data_value) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
                for (phase = 0; phase < (public_xtra->num_phases); phase++)
                {
                  for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                  {
                    indx = phase * (public_xtra->num_contaminants) + contaminant;
                    WellDataValueContaminantFraction(well_data_value, indx) = 1.0;
                  }
                }
              }
              WellDataFluxWellIntervalValue(well_data, flux_well, interval_number) = well_data_value;
            }

            /* Put in informational statistics for this well */
            well_data_stat = ctalloc(WellDataStat, 1);
            WellDataStatDeltaPhases(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
            for (phase = 0; phase < (public_xtra->num_phases); phase++)
            {
              WellDataStatDeltaPhase(well_data_stat, phase) = 0.0;
            }
            WellDataStatPhaseStats(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
            for (phase = 0; phase < (public_xtra->num_phases); phase++)
            {
              WellDataStatPhaseStat(well_data_stat, phase) = 0.0;
            }
            WellDataStatDeltaSaturations(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
            for (phase = 0; phase < (public_xtra->num_phases); phase++)
            {
              WellDataStatDeltaSaturation(well_data_stat, phase) = 0.0;
            }
            WellDataStatSaturationStats(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
            for (phase = 0; phase < (public_xtra->num_phases); phase++)
            {
              WellDataStatSaturationStat(well_data_stat, phase) = 0.0;
            }
            WellDataStatDeltaContaminants(well_data_stat) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
            for (phase = 0; phase < (public_xtra->num_phases); phase++)
            {
              for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
              {
                indx = phase * (public_xtra->num_contaminants) + contaminant;
                WellDataStatDeltaContaminant(well_data_stat, indx) = 0.0;
              }
            }
            WellDataStatContaminantStats(well_data_stat) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
            for (phase = 0; phase < (public_xtra->num_phases); phase++)
            {
              for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
              {
                indx = phase * (public_xtra->num_contaminants) + contaminant;
                WellDataStatContaminantStat(well_data_stat, indx) = 0.0;
              }
            }
            WellDataFluxWellStat(well_data, flux_well) = well_data_stat;

            flux_well++;
          }
          sequence_number++;
          break;
        }

        case 1:
        {
          dummy1 = (Type1*)(public_xtra->data[i]);

          x_lower = (dummy1->xlocation);
          y_lower = (dummy1->ylocation);
          x_upper = (dummy1->xlocation);
          y_upper = (dummy1->ylocation);

          /* well_action = 0 means we're doing extraction, well_action = 1 means we're doing injection    */
          /* The ordering of the extraction and injection wells is important.  The partner_ptr of the     */
          /*   injection well needs to point to allocated data (in the extraction well).  If the order is */
          /*   reversed then this storage wont exist.                                                     */

          for (well_action = 0; well_action < 2; well_action++)
          {
            ix = IndexSpaceX((dummy1->xlocation), 0);
            iy = IndexSpaceY((dummy1->ylocation), 0);
            if (well_action == 0)
            {
              z_lower = (dummy1->z_lower_ext);
              z_upper = (dummy1->z_upper_ext);

              action = EXTRACTION_WELL;
              phase_values = (dummy1->phase_values_ext);
              mechanism = (dummy1->mechanism_ext);
              method = (dummy1->method_ext);
            }
            else
            {
              z_lower = (dummy1->z_lower_inj);
              z_upper = (dummy1->z_upper_inj);

              action = INJECTION_WELL;
              phase_values = (dummy1->phase_values_inj);
              mechanism = (dummy1->mechanism_inj);
              method = (dummy1->method_inj);
            }

            iz_lower = IndexSpaceZ(z_lower, 0);
            iz_upper = IndexSpaceZ(z_upper, 0);

            nx = 1;
            ny = 1;
            nz = iz_upper - iz_lower + 1;

            rx = 0;
            ry = 0;
            rz = 0;

            process = amps_Rank(amps_CommWorld);

            new_subgrid = NewSubgrid(ix, iy, iz_lower,
                                     nx, ny, nz,
                                     rx, ry, rz,
                                     process);
            dx = SubgridDX(new_subgrid);
            dy = SubgridDY(new_subgrid);
            dz = SubgridDZ(new_subgrid);

            subgrid_volume = (nx * dx) * (ny * dy) * (nz * dz);

            if (mechanism == PRESSURE_WELL)
            {
              /* Put in physical data for this well */
              well_data_physical = ctalloc(WellDataPhysical, 1);
              WellDataPhysicalNumber(well_data_physical) = sequence_number;
              if (action == EXTRACTION_WELL)
              {
                WellDataPhysicalName(well_data_physical) = ctalloc(char, strlen((dummy1->name)) + 14);
                strcpy(WellDataPhysicalName(well_data_physical), (dummy1->name));
                strcat(WellDataPhysicalName(well_data_physical), " (extraction)");
              }
              else
              {
                WellDataPhysicalName(well_data_physical) = ctalloc(char, strlen((dummy1->name)) + 13);
                strcpy(WellDataPhysicalName(well_data_physical), (dummy1->name));
                strcat(WellDataPhysicalName(well_data_physical), " (injection)");
              }

              WellDataPhysicalXLower(well_data_physical) = x_lower;
              WellDataPhysicalYLower(well_data_physical) = y_lower;
              WellDataPhysicalZLower(well_data_physical) = z_lower;
              WellDataPhysicalXUpper(well_data_physical) = x_upper;
              WellDataPhysicalYUpper(well_data_physical) = y_upper;
              WellDataPhysicalZUpper(well_data_physical) = z_upper;

              WellDataPhysicalDiameter(well_data_physical) = pfmin(dx, dx);
              WellDataPhysicalSubgrid(well_data_physical) = new_subgrid;
              WellDataPhysicalSize(well_data_physical) = subgrid_volume;
              WellDataPhysicalAction(well_data_physical) = action;
              WellDataPhysicalMethod(well_data_physical) = method;
              WellDataPhysicalCycleNumber(well_data_physical) = (dummy1->cycle_number);
              WellDataPhysicalAveragePermeabilityX(well_data_physical) = 0.0;
              WellDataPhysicalAveragePermeabilityY(well_data_physical) = 0.0;
              WellDataPhysicalAveragePermeabilityZ(well_data_physical) = 0.0;
              WellDataPressWellPhysical(well_data, press_well) = well_data_physical;

              /* Put in values for this well */
              interval_division = TimeCycleDataIntervalDivision(time_cycle_data, WellDataPhysicalCycleNumber(well_data_physical));
              WellDataPressWellIntervalValues(well_data, press_well) = ctalloc(WellDataValue *, interval_division);
              for (interval_number = 0; interval_number < interval_division; interval_number++)
              {
                well_data_value = ctalloc(WellDataValue, 1);

                WellDataValuePhaseValues(well_data_value) = ctalloc(double, 1);
                phase = 0;
                {
                  WellDataValuePhaseValue(well_data_value, phase) = ((phase_values[interval_number])[phase]);
                }
                if (action == INJECTION_WELL)
                {
                  /* This is where the dependence of the injection well on the extraction well occurs */
                  WellDataValueDeltaSaturationPtrs(well_data_value) = WellDataStatDeltaSaturations(well_data_stat);
                  WellDataValueDeltaContaminantPtrs(well_data_value) = WellDataStatDeltaContaminants(well_data_stat);

                  WellDataValueContaminantFractions(well_data_value) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
                  for (phase = 0; phase < (public_xtra->num_phases); phase++)
                  {
                    for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                    {
                      indx = phase * (public_xtra->num_contaminants) + contaminant;
                      WellDataValueContaminantFraction(well_data_value, indx) = 1.0 - ((dummy1->contaminant_fractions[interval_number])[indx]);
                    }
                  }
                }
                else
                {
                  WellDataValueContaminantFractions(well_data_value) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
                  for (phase = 0; phase < (public_xtra->num_phases); phase++)
                  {
                    for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                    {
                      indx = phase * (public_xtra->num_contaminants) + contaminant;
                      WellDataValueContaminantFraction(well_data_value, indx) = ((dummy1->contaminant_fractions[interval_number])[indx]);
                    }
                  }
                }
                WellDataPressWellIntervalValue(well_data, press_well, interval_number) = well_data_value;
              }

              /* Put in informational statistics for this well */
              well_data_stat = ctalloc(WellDataStat, 1);
              WellDataStatDeltaPhases(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
              for (phase = 0; phase < (public_xtra->num_phases); phase++)
              {
                WellDataStatDeltaPhase(well_data_stat, phase) = 0.0;
              }
              WellDataStatPhaseStats(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
              for (phase = 0; phase < (public_xtra->num_phases); phase++)
              {
                WellDataStatPhaseStat(well_data_stat, phase) = 0.0;
              }
              WellDataStatDeltaSaturations(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
              for (phase = 0; phase < (public_xtra->num_phases); phase++)
              {
                WellDataStatDeltaSaturation(well_data_stat, phase) = 0.0;
              }
              WellDataStatSaturationStats(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
              for (phase = 0; phase < (public_xtra->num_phases); phase++)
              {
                WellDataStatSaturationStat(well_data_stat, phase) = 0.0;
              }
              WellDataStatDeltaContaminants(well_data_stat) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
              for (phase = 0; phase < (public_xtra->num_phases); phase++)
              {
                for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                {
                  indx = phase * (public_xtra->num_contaminants) + contaminant;
                  WellDataStatDeltaContaminant(well_data_stat, indx) = 0.0;
                }
              }
              WellDataStatContaminantStats(well_data_stat) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
              for (phase = 0; phase < (public_xtra->num_phases); phase++)
              {
                for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                {
                  indx = phase * (public_xtra->num_contaminants) + contaminant;
                  WellDataStatContaminantStat(well_data_stat, indx) = 0.0;
                }
              }
              WellDataFluxWellStat(well_data, flux_well) = well_data_stat;

              press_well++;
            }
            else if (mechanism == FLUX_WELL)
            {
              well_data_physical = ctalloc(WellDataPhysical, 1);
              WellDataPhysicalNumber(well_data_physical) = sequence_number;
              if (action == EXTRACTION_WELL)
              {
                WellDataPhysicalName(well_data_physical) = ctalloc(char, strlen((dummy1->name)) + 14);
                strcpy(WellDataPhysicalName(well_data_physical), (dummy1->name));
                strcat(WellDataPhysicalName(well_data_physical), " (extraction)");
              }
              else
              {
                WellDataPhysicalName(well_data_physical) = ctalloc(char, strlen((dummy1->name)) + 13);
                strcpy(WellDataPhysicalName(well_data_physical), (dummy1->name));
                strcat(WellDataPhysicalName(well_data_physical), " (injection)");
              }

              WellDataPhysicalXLower(well_data_physical) = x_lower;
              WellDataPhysicalYLower(well_data_physical) = y_lower;
              WellDataPhysicalZLower(well_data_physical) = z_lower;
              WellDataPhysicalXUpper(well_data_physical) = x_upper;
              WellDataPhysicalYUpper(well_data_physical) = y_upper;
              WellDataPhysicalZUpper(well_data_physical) = z_upper;

              WellDataPhysicalDiameter(well_data_physical) = pfmin(dx, dx);
              WellDataPhysicalSubgrid(well_data_physical) = new_subgrid;
              WellDataPhysicalSize(well_data_physical) = subgrid_volume;
              WellDataPhysicalAction(well_data_physical) = action;
              WellDataPhysicalMethod(well_data_physical) = method;
              WellDataPhysicalCycleNumber(well_data_physical) = (dummy1->cycle_number);
              WellDataPhysicalAveragePermeabilityX(well_data_physical) = 0.0;
              WellDataPhysicalAveragePermeabilityY(well_data_physical) = 0.0;
              WellDataPhysicalAveragePermeabilityZ(well_data_physical) = 0.0;
              WellDataFluxWellPhysical(well_data, flux_well) = well_data_physical;

              /* Put in values for this well */
              interval_division = TimeCycleDataIntervalDivision(time_cycle_data, WellDataPhysicalCycleNumber(well_data_physical));
              WellDataFluxWellIntervalValues(well_data, flux_well) = ctalloc(WellDataValue *, interval_division);
              for (interval_number = 0; interval_number < interval_division; interval_number++)
              {
                well_data_value = ctalloc(WellDataValue, 1);

                WellDataValuePhaseValues(well_data_value) = ctalloc(double, (public_xtra->num_phases));
                for (phase = 0; phase < (public_xtra->num_phases); phase++)
                {
                  WellDataValuePhaseValue(well_data_value, phase) = ((phase_values[interval_number])[phase]);
                }
                if (action == INJECTION_WELL)
                {
                  /* This is where the dependence of the injection well on the extraction well occurs */
                  WellDataValueDeltaSaturationPtrs(well_data_value) = WellDataStatDeltaSaturations(well_data_stat);
                  WellDataValueDeltaContaminantPtrs(well_data_value) = WellDataStatDeltaContaminants(well_data_stat);

                  WellDataValueContaminantFractions(well_data_value) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
                  for (phase = 0; phase < (public_xtra->num_phases); phase++)
                  {
                    for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                    {
                      indx = phase * (public_xtra->num_contaminants) + contaminant;
                      WellDataValueContaminantFraction(well_data_value, indx) = 1.0 - ((dummy1->contaminant_fractions[interval_number])[indx]);
                    }
                  }
                }
                else
                {
                  WellDataValueContaminantFractions(well_data_value) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
                  for (phase = 0; phase < (public_xtra->num_phases); phase++)
                  {
                    for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                    {
                      indx = phase * (public_xtra->num_contaminants) + contaminant;
                      WellDataValueContaminantFraction(well_data_value, indx) = ((dummy1->contaminant_fractions[interval_number])[indx]);
                    }
                  }
                }
                WellDataFluxWellIntervalValue(well_data, flux_well, interval_number) = well_data_value;
              }

              /* Put in informational statistics for this well */
              well_data_stat = ctalloc(WellDataStat, 1);
              WellDataStatDeltaPhases(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
              for (phase = 0; phase < (public_xtra->num_phases); phase++)
              {
                WellDataStatDeltaPhase(well_data_stat, phase) = 0.0;
              }
              WellDataStatPhaseStats(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
              for (phase = 0; phase < (public_xtra->num_phases); phase++)
              {
                WellDataStatPhaseStat(well_data_stat, phase) = 0.0;
              }
              WellDataStatDeltaSaturations(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
              for (phase = 0; phase < (public_xtra->num_phases); phase++)
              {
                WellDataStatDeltaSaturation(well_data_stat, phase) = 0.0;
              }
              WellDataStatSaturationStats(well_data_stat) = ctalloc(double, (public_xtra->num_phases));
              for (phase = 0; phase < (public_xtra->num_phases); phase++)
              {
                WellDataStatSaturationStat(well_data_stat, phase) = 0.0;
              }
              WellDataStatDeltaContaminants(well_data_stat) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
              for (phase = 0; phase < (public_xtra->num_phases); phase++)
              {
                for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                {
                  indx = phase * (public_xtra->num_contaminants) + contaminant;
                  WellDataStatDeltaContaminant(well_data_stat, indx) = 0.0;
                }
              }
              WellDataStatContaminantStats(well_data_stat) = ctalloc(double, (public_xtra->num_phases) * (public_xtra->num_contaminants));
              for (phase = 0; phase < (public_xtra->num_phases); phase++)
              {
                for (contaminant = 0; contaminant < (public_xtra->num_contaminants); contaminant++)
                {
                  indx = phase * (public_xtra->num_contaminants) + contaminant;
                  WellDataStatContaminantStat(well_data_stat, indx) = 0.0;
                }
              }
              WellDataFluxWellStat(well_data, flux_well) = well_data_stat;

              flux_well++;
            }
            sequence_number++;
          }
          break;
        }
      }
    }
  }
}


/*--------------------------------------------------------------------------
 * WellPackageInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule *WellPackageInitInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

#if 0
  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
#endif
  instance_xtra = NULL;

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * WellPackageFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  WellPackageFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * WellPackageNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *WellPackageNewPublicXtra(
                                    int num_phases,
                                    int num_contaminants)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0         *dummy0;
  Type1         *dummy1;

  int num_units;
  int i, interval_division, interval_number;

  int num_cycles;
  int global_cycle;


  char *well_names;
  char *well_name;

  char *cycle_name;

  char key[IDB_MAX_KEY_LEN];

  char *switch_name;

  int phase;
  int contaminant;

  NameArray inputtype_na;
  NameArray action_na;
  NameArray mechanism_na;
  NameArray methodpress_na;
  NameArray methodflux_na;

  inputtype_na = NA_NewNameArray("Vertical Recirc");
  action_na = NA_NewNameArray("Injection Extraction");
  mechanism_na = NA_NewNameArray("Pressure Flux");
  methodpress_na = NA_NewNameArray("Standard");
  methodflux_na = NA_NewNameArray("Standard Weighted Patterned");

  public_xtra = ctalloc(PublicXtra, 1);


  (public_xtra->num_phases) = num_phases;
  (public_xtra->num_contaminants) = num_contaminants;

  char* EMPTY_NAMES_LIST = "";
  well_names = GetStringDefault("Wells.Names", EMPTY_NAMES_LIST);

  public_xtra->well_names = NA_NewNameArray(well_names);

  num_units = NA_Sizeof(public_xtra->well_names);


  num_cycles = public_xtra->num_cycles = num_units;

  public_xtra->interval_divisions = ctalloc(int, num_cycles);
  public_xtra->intervals = ctalloc(int *, num_cycles);
  public_xtra->repeat_counts = ctalloc(int, num_cycles);

  public_xtra->num_units = num_units;

  public_xtra->num_press_wells = 0;
  public_xtra->num_flux_wells = 0;

  if (num_units > 0)
  {
    (public_xtra->type) = ctalloc(int, num_units);
    (public_xtra->data) = ctalloc(void *, num_units);

    for (i = 0; i < num_units; i++)
    {
      well_name = NA_IndexToName(public_xtra->well_names, i);

      sprintf(key, "Wells.%s.InputType", well_name);
      switch_name = GetString(key);
      public_xtra->type[i] = NA_NameToIndexExitOnError(inputtype_na, switch_name, key);

      switch ((public_xtra->type[i]))
      {
        case 0:
        {
          dummy0 = ctalloc(Type0, 1);

          /*** Read in the physical data for the well ***/
          dummy0->name = strdup(well_name);

          sprintf(key, "Wells.%s.Action", well_name);
          switch_name = GetString(key);
          dummy0->action = NA_NameToIndexExitOnError(action_na, switch_name, key);

          sprintf(key, "Wells.%s.Type", well_name);
          switch_name = GetString(key);
          dummy0->mechanism = NA_NameToIndexExitOnError(mechanism_na, switch_name, key);

          sprintf(key, "Wells.%s.X", well_name);
          dummy0->xlocation = GetDouble(key);

          sprintf(key, "Wells.%s.Y", well_name);
          dummy0->ylocation = GetDouble(key);

          sprintf(key, "Wells.%s.ZUpper", well_name);
          dummy0->z_upper = GetDouble(key);


          sprintf(key, "Wells.%s.ZLower", well_name);
          dummy0->z_lower = GetDouble(key);

          if ((dummy0->mechanism) == PRESSURE_WELL)
          {
            sprintf(key, "Wells.%s.Method", well_name);
            switch_name = GetString(key);
            (dummy0->method) = NA_NameToIndexExitOnError(methodpress_na, switch_name, key);
          }
          else if ((dummy0->mechanism) == FLUX_WELL)
          {
            sprintf(key, "Wells.%s.Method", well_name);
            switch_name = GetString(key);
            (dummy0->method) = NA_NameToIndexExitOnError(methodflux_na, switch_name, key);
          }

          sprintf(key, "Wells.%s.Cycle", well_name);
          cycle_name = GetString(key);
          global_cycle = NA_NameToIndexExitOnError(GlobalsCycleNames, cycle_name, key);

          dummy0->cycle_number = i;

          interval_division = public_xtra->interval_divisions[i] =
            GlobalsIntervalDivisions[global_cycle];

          public_xtra->repeat_counts[i] =
            GlobalsRepeatCounts[global_cycle];

          (public_xtra->intervals[i]) = ctalloc(int, interval_division);
          for (interval_number = 0; interval_number < interval_division;
               interval_number++)
          {
            public_xtra->intervals[i][interval_number] =
              GlobalsIntervals[global_cycle][interval_number];
          }

          dummy0->phase_values = ctalloc(double *, interval_division);

          if ((dummy0->action) == INJECTION_WELL)
          {
            (dummy0->saturation_values) = ctalloc(double *,
                                                  interval_division);
            (dummy0->contaminant_values) = ctalloc(double *,
                                                   interval_division);
          }
          else if ((dummy0->action) == EXTRACTION_WELL)
          {
            (dummy0->saturation_values) = NULL;
            (dummy0->contaminant_values) = NULL;
          }

          /*** Read in the values for the well ***/
          for (interval_number = 0; interval_number < interval_division;
               interval_number++)
          {
            if ((dummy0->mechanism) == PRESSURE_WELL)
            {
              dummy0->phase_values[interval_number] = ctalloc(double, 1);

              sprintf(key, "Wells.%s.%s.Pressure.Value",
                      well_name,
                      NA_IndexToName(
                                     GlobalsIntervalNames[global_cycle],
                                     interval_number));

              dummy0->phase_values[interval_number][0] = GetDouble(key);
            }
            else if ((dummy0->mechanism) == FLUX_WELL)
            {
              (dummy0->phase_values[interval_number]) = ctalloc(double,
                                                                num_phases);

              for (phase = 0; phase < num_phases; phase++)
              {
                sprintf(key, "Wells.%s.%s.Flux.%s.Value",
                        well_name,
                        NA_IndexToName(
                                       GlobalsIntervalNames[global_cycle],
                                       interval_number),
                        NA_IndexToName(GlobalsPhaseNames, phase));

                dummy0->phase_values[interval_number][phase] =
                  GetDouble(key);
              }
            }


            if ((dummy0->action) == INJECTION_WELL)
            {
              dummy0->saturation_values[interval_number] =
                ctalloc(double, num_phases);

              for (phase = 0; phase < num_phases; phase++)
              {
                sprintf(key, "Wells.%s.%s.Saturation.%s.Value",
                        well_name,
                        NA_IndexToName(
                                       GlobalsIntervalNames[global_cycle],
                                       interval_number),
                        NA_IndexToName(GlobalsPhaseNames, phase));

                dummy0->saturation_values[interval_number][phase] =
                  GetDouble(key);
              }

              dummy0->contaminant_values[interval_number] =
                ctalloc(double, num_phases * num_contaminants);

              for (phase = 0;
                   phase < num_phases;
                   phase++)
              {
                for (contaminant = 0;
                     contaminant < num_contaminants;
                     contaminant++)
                {
                  sprintf(key, "Wells.%s.%s.Concentration.%s.%s.Value",
                          well_name,
                          NA_IndexToName(
                                         GlobalsIntervalNames[global_cycle],
                                         interval_number),
                          NA_IndexToName(GlobalsPhaseNames, phase),
                          NA_IndexToName(GlobalsContaminatNames,
                                         contaminant));
                  dummy0->contaminant_values[interval_number]
                  [phase + contaminant] = GetDouble(key);
                }
              }
            }
          }

          /*** Bump the counter for the well type ***/
          if ((dummy0->mechanism) == PRESSURE_WELL)
          {
            (public_xtra->num_press_wells)++;
          }
          else if ((dummy0->mechanism) == FLUX_WELL)
          {
            (public_xtra->num_flux_wells)++;
          }

          (public_xtra->data[i]) = (void*)dummy0;

          break;
        }

        case 1:
        {
          dummy1 = ctalloc(Type1, 1);

          /*** Read in the physical data for the well ***/

          dummy1->name = strdup(well_name);

          sprintf(key, "Wells.%s.ExtractionType", well_name);
          switch_name = GetString(key);
          dummy1->mechanism_ext =
            NA_NameToIndexExitOnError(mechanism_na, switch_name, key);


          sprintf(key, "Wells.%s.InjectionType", well_name);
          switch_name = GetString(key);
          dummy1->mechanism_inj =
            NA_NameToIndexExitOnError(mechanism_na, switch_name, key);

          sprintf(key, "Wells.%s.X", well_name);
          dummy1->xlocation = GetDouble(key);

          sprintf(key, "Wells.%s.Y", well_name);
          dummy1->ylocation = GetDouble(key);

          sprintf(key, "Wells.%s.ExtractionZUpper", well_name);
          dummy1->z_upper_ext = GetDouble(key);


          sprintf(key, "Wells.%s.ExtractionZLower", well_name);
          dummy1->z_lower_ext = GetDouble(key);


          sprintf(key, "Wells.%s.InjectionZUpper", well_name);
          dummy1->z_upper_inj = GetDouble(key);


          sprintf(key, "Wells.%s.InjectionZLower", well_name);
          dummy1->z_lower_inj = GetDouble(key);

          if ((dummy1->mechanism_ext) == PRESSURE_WELL)
          {
            sprintf(key, "Wells.%s.ExtractionMethod", well_name);
            switch_name = GetString(key);
            (dummy1->method_ext) = NA_NameToIndexExitOnError(methodpress_na, switch_name, key);
          }
          else if ((dummy1->mechanism_ext) == FLUX_WELL)
          {
            sprintf(key, "Wells.%s.ExtractionMethod", well_name);
            switch_name = GetString(key);
            (dummy1->method_ext) = NA_NameToIndexExitOnError(methodflux_na, switch_name, key);
          }

          if ((dummy1->mechanism_inj) == PRESSURE_WELL)
          {
            sprintf(key, "Wells.%s.InjectionMethod", well_name);
            switch_name = GetString(key);
            (dummy1->method_inj) = NA_NameToIndexExitOnError(methodpress_na, switch_name, key);
          }
          else if ((dummy1->mechanism_inj) == FLUX_WELL)
          {
            sprintf(key, "Wells.%s.InjectionMethod", well_name);
            switch_name = GetString(key);
            (dummy1->method_inj) = NA_NameToIndexExitOnError(methodflux_na, switch_name, key);
          }

          sprintf(key, "Wells.%s.Cycle", well_name);
          cycle_name = GetString(key);

          global_cycle = NA_NameToIndexExitOnError(GlobalsCycleNames, cycle_name, key);

          dummy1->cycle_number = i;

          interval_division = public_xtra->interval_divisions[i] =
            GlobalsIntervalDivisions[global_cycle];

          public_xtra->repeat_counts[i] =
            GlobalsRepeatCounts[global_cycle];

          (public_xtra->intervals[i]) = ctalloc(int, interval_division);
          for (interval_number = 0; interval_number < interval_division;
               interval_number++)
          {
            public_xtra->intervals[i][interval_number] =
              GlobalsIntervals[global_cycle][interval_number];
          }



          (dummy1->phase_values_ext) = ctalloc(double *, interval_division);
          (dummy1->phase_values_inj) = ctalloc(double *, interval_division);
          (dummy1->contaminant_fractions) = ctalloc(double *, interval_division);

          /*** Read in the values for the well ***/
          for (interval_number = 0; interval_number < interval_division; interval_number++)
          {
            /*** Read in the values for the extraction well ***/
            if ((dummy1->mechanism_ext) == PRESSURE_WELL)
            {
              dummy1->phase_values_ext[interval_number] =
                ctalloc(double, 1);

              sprintf(key, "Wells.%s.%s.Extraction.Pressure.Value",
                      well_name,
                      NA_IndexToName(
                                     GlobalsIntervalNames[global_cycle],
                                     interval_number));

              dummy1->phase_values_ext[interval_number][0] =
                GetDouble(key);
            }
            else if ((dummy1->mechanism_ext) == FLUX_WELL)
            {
              dummy1->phase_values_ext[interval_number] =
                ctalloc(double, num_phases);

              for (phase = 0; phase < num_phases; phase++)
              {
                sprintf(key, "Wells.%s.%s.Extraction.Flux.%s.Value",
                        well_name,
                        NA_IndexToName(
                                       GlobalsIntervalNames[global_cycle],
                                       interval_number),
                        NA_IndexToName(GlobalsPhaseNames, phase));

                dummy1->phase_values_ext[interval_number][phase] =
                  GetDouble(key);
              }
            }

            /*** Read in the values for the injection well ***/
            if ((dummy1->mechanism_inj) == PRESSURE_WELL)
            {
              dummy1->phase_values_inj[interval_number] =
                ctalloc(double, 1);

              sprintf(key, "Wells.%s.%s.Injection.Pressure.Value",
                      well_name,
                      NA_IndexToName(
                                     GlobalsIntervalNames[global_cycle],
                                     interval_number));

              dummy1->phase_values_inj[interval_number][0] =
                GetDouble(key);
            }
            else if ((dummy1->mechanism_inj) == FLUX_WELL)
            {
              dummy1->phase_values_inj[interval_number] =
                ctalloc(double, num_phases);

              for (phase = 0; phase < num_phases; phase++)
              {
                sprintf(key, "Wells.%s.%s.Injection.Flux.%s.Value",
                        well_name,
                        NA_IndexToName(
                                       GlobalsIntervalNames[global_cycle],
                                       interval_number),
                        NA_IndexToName(GlobalsPhaseNames, phase));

                dummy1->phase_values_inj[interval_number][phase] =
                  GetDouble(key);
              }
            }

            /* read in the fractions */
            (dummy1->contaminant_fractions[interval_number]) =
              ctalloc(double, num_phases * num_contaminants);

            for (phase = 0; phase < num_phases; phase++)
            {
              for (contaminant = 0;
                   contaminant < num_contaminants;
                   contaminant++)
              {
                sprintf(key,
                        "Wells.%s.%s.Injection.Concentration.%s.%s.Fraction",
                        well_name,
                        NA_IndexToName(
                                       GlobalsIntervalNames[global_cycle],
                                       interval_number),
                        NA_IndexToName(GlobalsPhaseNames, phase),
                        NA_IndexToName(GlobalsContaminatNames,
                                       contaminant));
                dummy1->contaminant_fractions[interval_number]
                [phase + contaminant] = GetDouble(key);
              }
            }
          }

          /*** Bump the counter for both well types ***/
          if ((dummy1->mechanism_inj) == PRESSURE_WELL)
          {
            (public_xtra->num_press_wells)++;
          }
          else if ((dummy1->mechanism_inj) == FLUX_WELL)
          {
            (public_xtra->num_flux_wells)++;
          }
          if ((dummy1->mechanism_ext) == PRESSURE_WELL)
          {
            (public_xtra->num_press_wells)++;
          }
          else if ((dummy1->mechanism_ext) == FLUX_WELL)
          {
            (public_xtra->num_flux_wells)++;
          }

          (public_xtra->data[i]) = (void*)dummy1;

          break;
        }

        default:
        {
          InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
        }
      }
    }
  }

  (public_xtra->num_wells) = (public_xtra->num_press_wells) + (public_xtra->num_flux_wells);

  NA_FreeNameArray(methodflux_na);
  NA_FreeNameArray(methodpress_na);
  NA_FreeNameArray(mechanism_na);
  NA_FreeNameArray(action_na);
  NA_FreeNameArray(inputtype_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*-------------------------------------------------------------------------
 * WellPackageFreePublicXtra
 *-------------------------------------------------------------------------*/

void  WellPackageFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0         *dummy0;
  Type1         *dummy1;

  int num_units, num_cycles;
  int i, interval_number, interval_division;

  if (public_xtra)
  {
    NA_FreeNameArray(public_xtra->well_names);

    /* Free the well information */
    num_units = (public_xtra->num_units);
    if (num_units > 0)
    {
      for (i = 0; i < num_units; i++)
      {
        switch ((public_xtra->type[i]))
        {
          case 0:
          {
            dummy0 = (Type0*)(public_xtra->data[i]);

            interval_division = (public_xtra->interval_divisions[(dummy0->cycle_number)]);

            for (interval_number = 0; interval_number < interval_division; interval_number++)
            {
              if ((dummy0->contaminant_values))
              {
                if ((dummy0->contaminant_values[interval_number]))
                {
                  tfree((dummy0->contaminant_values[interval_number]));
                }
              }
              if ((dummy0->saturation_values))
              {
                if ((dummy0->saturation_values[interval_number]))
                {
                  tfree((dummy0->saturation_values[interval_number]));
                }
              }
              if ((dummy0->phase_values))
              {
                if ((dummy0->phase_values[interval_number]))
                {
                  tfree((dummy0->phase_values[interval_number]));
                }
              }
            }
            if ((dummy0->contaminant_values))
            {
              tfree((dummy0->contaminant_values));
            }
            if ((dummy0->saturation_values))
            {
              tfree((dummy0->saturation_values));
            }
            if ((dummy0->phase_values))
            {
              tfree((dummy0->phase_values));
            }
            if ((dummy0->name))
            {
              tfree((dummy0->name));
            }

            tfree(dummy0);

            break;
          }

          case 1:
          {
            dummy1 = (Type1*)(public_xtra->data[i]);

            interval_division = (public_xtra->interval_divisions[(dummy1->cycle_number)]);

            for (interval_number = 0; interval_number < interval_division; interval_number++)
            {
              if ((dummy1->contaminant_fractions))
              {
                if ((dummy1->contaminant_fractions[interval_number]))
                {
                  tfree((dummy1->contaminant_fractions[interval_number]));
                }
              }
              if ((dummy1->phase_values_ext))
              {
                if ((dummy1->phase_values_ext[interval_number]))
                {
                  tfree((dummy1->phase_values_ext[interval_number]));
                }
              }
              if ((dummy1->phase_values_inj))
              {
                if ((dummy1->phase_values_inj[interval_number]))
                {
                  tfree((dummy1->phase_values_inj[interval_number]));
                }
              }
            }
            if ((dummy1->contaminant_fractions))
            {
              tfree((dummy1->contaminant_fractions));
            }
            if ((dummy1->phase_values_ext))
            {
              tfree((dummy1->phase_values_ext));
            }
            if ((dummy1->phase_values_inj))
            {
              tfree((dummy1->phase_values_inj));
            }
            if ((dummy1->name))
            {
              tfree((dummy1->name));
            }

            tfree(dummy1);

            break;
          }
        }
      }

      tfree(public_xtra->data);
      tfree(public_xtra->type);
    }

    /* Free the time cycling information */
    num_cycles = (public_xtra->num_cycles);

    tfree((public_xtra->repeat_counts));

    for (i = 0; i < num_cycles; i++)
    {
      tfree((public_xtra->intervals[i]));
    }
    tfree((public_xtra->intervals));

    tfree((public_xtra->interval_divisions));


    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * WellPackageSizeOfTempData
 *--------------------------------------------------------------------------*/

int  WellPackageSizeOfTempData()
{
  return 0;
}
