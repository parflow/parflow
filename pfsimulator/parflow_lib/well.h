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

#ifndef _WELL_HEADER
#define _WELL_HEADER

/*----------------------------------------------------------------
 * Well Physical Values structure
 *----------------------------------------------------------------*/

typedef struct {
  int number;
  char          *name;
  double x_lower, y_lower, z_lower;
  double x_upper, y_upper, z_upper;
  double diameter;
  Subgrid       *subgrid;
  double size;
  int action;
  int method;
  int cycle_number;
  double average_permeability_x;
  double average_permeability_y;
  double average_permeability_z;
} WellDataPhysical;

/*----------------------------------------------------------------
 * Well Data Values structure
 *----------------------------------------------------------------*/

typedef struct {
  double        *phase_values;            /* 1 (press) or num_phases (flux) */
  double        *saturation_values;       /*           num_phases           */
  double        *delta_saturation_ptrs;   /*           num_phases           */
  double        *contaminant_values;      /* num_phases * num_contaminants  */
  double        *delta_contaminant_ptrs;  /* num_phases * num_contaminants  */
  double        *contaminant_fractions;   /* num_phases * num_contaminants  */
} WellDataValue;

/*----------------------------------------------------------------
 * Well Data Stats structure
 *----------------------------------------------------------------*/

typedef struct {
  double        *delta_phases;        /*          num_phases           */
  double        *phase_stats;         /*          num_phases           */
  double        *delta_saturations;   /*          num_phases           */
  double        *saturation_stats;    /*          num_phases           */
  double        *delta_contaminants;  /* num_phases * num_contaminants */
  double        *contaminant_stats;   /* num_phases * num_contaminants */
} WellDataStat;

/*----------------------------------------------------------------
 * Well Data structure
 *----------------------------------------------------------------*/

typedef struct {
  int num_phases;
  int num_contaminants;

  int num_wells;

  /* Pressure well section */
  int num_press_wells;

  WellDataPhysical  **press_well_physicals;
  WellDataValue    ***press_well_values;
  WellDataStat      **press_well_stats;

  /* Flux well section */
  int num_flux_wells;

  WellDataPhysical  **flux_well_physicals;
  WellDataValue    ***flux_well_values;
  WellDataStat      **flux_well_stats;

  /* time info */
  TimeCycleData      *time_cycle_data;
} WellData;

/*--------------------------------------------------------------------------
 * Accessor macros: WellDataPhysical
 *--------------------------------------------------------------------------*/
#define WellDataPhysicalNumber(well_data_physical) \
        ((well_data_physical)->number)

#define WellDataPhysicalName(well_data_physical) \
        ((well_data_physical)->name)

#define WellDataPhysicalXLower(well_data_physical) \
        ((well_data_physical)->x_lower)

#define WellDataPhysicalYLower(well_data_physical) \
        ((well_data_physical)->y_lower)

#define WellDataPhysicalZLower(well_data_physical) \
        ((well_data_physical)->z_lower)

#define WellDataPhysicalXUpper(well_data_physical) \
        ((well_data_physical)->x_upper)

#define WellDataPhysicalYUpper(well_data_physical) \
        ((well_data_physical)->y_upper)

#define WellDataPhysicalZUpper(well_data_physical) \
        ((well_data_physical)->z_upper)

#define WellDataPhysicalSubgrid(well_data_physical) \
        ((well_data_physical)->subgrid)

#define WellDataPhysicalDiameter(well_data_physical) \
        ((well_data_physical)->diameter)

#define WellDataPhysicalSize(well_data_physical) \
        ((well_data_physical)->size)

#define WellDataPhysicalAction(well_data_physical) \
        ((well_data_physical)->action)

#define WellDataPhysicalMethod(well_data_physical) \
        ((well_data_physical)->method)

#define WellDataPhysicalCycleNumber(well_data_physical) \
        ((well_data_physical)->cycle_number)

#define WellDataPhysicalAveragePermeabilityX(well_data_physical) \
        ((well_data_physical)->average_permeability_x)

#define WellDataPhysicalAveragePermeabilityY(well_data_physical) \
        ((well_data_physical)->average_permeability_y)

#define WellDataPhysicalAveragePermeabilityZ(well_data_physical) \
        ((well_data_physical)->average_permeability_z)

/*--------------------------------------------------------------------------
 * Accessor macros: WellDataValue
 *--------------------------------------------------------------------------*/
#define WellDataValueIntervalValues(well_data_values) \
        (((well_data)->press_well_values[i])[interval_number])

#define WellDataValuePhaseValues(well_data_value) \
        ((well_data_value)->phase_values)
#define WellDataValuePhaseValue(well_data_value, i) \
        ((well_data_value)->phase_values[i])

#define WellDataValueSaturationValues(well_data_value) \
        ((well_data_value)->saturation_values)
#define WellDataValueSaturationValue(well_data_value, i) \
        ((well_data_value)->saturation_values[i])

#define WellDataValueDeltaSaturationPtrs(well_data_value) \
        ((well_data_value)->delta_saturation_ptrs)
#define WellDataValueDeltaSaturationPtr(well_data_value, i) \
        ((well_data_value)->delta_saturation_ptrs[i])

#define WellDataValueContaminantValues(well_data_value) \
        ((well_data_value)->contaminant_values)
#define WellDataValueContaminantValue(well_data_value, i) \
        ((well_data_value)->contaminant_values[i])

#define WellDataValueDeltaContaminantPtrs(well_data_value) \
        ((well_data_value)->delta_contaminant_ptrs)
#define WellDataValueDeltaContaminantPtr(well_data_value, i) \
        ((well_data_value)->delta_contaminant_ptrs[i])

#define WellDataValueContaminantFractions(well_data_value) \
        ((well_data_value)->contaminant_fractions)
#define WellDataValueContaminantFraction(well_data_value, i) \
        ((well_data_value)->contaminant_fractions[i])

/*--------------------------------------------------------------------------
 * Accessor macros: WellDataStat
 *--------------------------------------------------------------------------*/
#define WellDataStatDeltaPhases(well_data_stat) \
        ((well_data_stat)->delta_phases)
#define WellDataStatDeltaPhase(well_data_stat, i) \
        ((well_data_stat)->delta_phases[i])

#define WellDataStatPhaseStats(well_data_stat) \
        ((well_data_stat)->phase_stats)
#define WellDataStatPhaseStat(well_data_stat, i) \
        ((well_data_stat)->phase_stats[i])

#define WellDataStatDeltaSaturations(well_data_stat) \
        ((well_data_stat)->delta_saturations)
#define WellDataStatDeltaSaturation(well_data_stat, i) \
        ((well_data_stat)->delta_saturations[i])

#define WellDataStatSaturationStats(well_data_stat) \
        ((well_data_stat)->saturation_stats)
#define WellDataStatSaturationStat(well_data_stat, i) \
        ((well_data_stat)->saturation_stats[i])

#define WellDataStatDeltaContaminants(well_data_stat) \
        ((well_data_stat)->delta_contaminants)
#define WellDataStatDeltaContaminant(well_data_stat, i) \
        ((well_data_stat)->delta_contaminants[i])

#define WellDataStatContaminantStats(well_data_stat) \
        ((well_data_stat)->contaminant_stats)
#define WellDataStatContaminantStat(well_data_stat, i) \
        ((well_data_stat)->contaminant_stats[i])

/*--------------------------------------------------------------------------
 * Accessor macros: WellData
 *--------------------------------------------------------------------------*/
#define WellDataNumPhases(well_data) ((well_data)->num_phases)
#define WellDataNumContaminants(well_data) ((well_data)->num_contaminants)

#define WellDataNumWells(well_data) ((well_data)->num_wells)

#define WellDataTimeCycleData(well_data) ((well_data)->time_cycle_data)

/*-------------------------- Pressure well data ----------------------------*/
#define WellDataNumPressWells(well_data) ((well_data)->num_press_wells)

#define WellDataPressWellPhysicals(well_data) \
        ((well_data)->press_well_physicals)
#define WellDataPressWellPhysical(well_data, i) \
        ((well_data)->press_well_physicals[i])

#define WellDataPressWellValues(well_data) \
        ((well_data)->press_well_values)
#define WellDataPressWellIntervalValues(well_data, i) \
        ((well_data)->press_well_values[i])
#define WellDataPressWellIntervalValue(well_data, i, interval_number) \
        (((well_data)->press_well_values[i])[interval_number])

#define WellDataPressWellStats(well_data) \
        ((well_data)->press_well_stats)
#define WellDataPressWellStat(well_data, i) \
        ((well_data)->press_well_stats[i])

/*---------------------------- Flux well data ------------------------------*/
#define WellDataNumFluxWells(well_data)      ((well_data)->num_flux_wells)

#define WellDataFluxWellPhysicals(well_data) \
        ((well_data)->flux_well_physicals)
#define WellDataFluxWellPhysical(well_data, i) \
        ((well_data)->flux_well_physicals[i])

#define WellDataFluxWellValues(well_data) \
        ((well_data)->flux_well_values)
#define WellDataFluxWellIntervalValues(well_data, i) \
        ((well_data)->flux_well_values[i])
#define WellDataFluxWellIntervalValue(well_data, i, interval_number) \
        (((well_data)->flux_well_values[i])[interval_number])

#define WellDataFluxWellStats(well_data) \
        ((well_data)->flux_well_stats)
#define WellDataFluxWellStat(well_data, i) \
        ((well_data)->flux_well_stats[i])

/*--------------------------------------------------------------------------
 * Well Data constants used in the program.
 *--------------------------------------------------------------------------*/

/*       Actions        */
#define INJECTION_WELL  0
#define EXTRACTION_WELL 1

/*       Methods        */
/* These should match  with */
/* the Name Array Defs from */
/* the  Well Package input. */
#define PRESS_STANDARD 0
#define FLUX_STANDARD  0
#define FLUX_WEIGHTED  1
#define FLUX_PATTERNED 2

/*     Write Options      */
#define WELLDATA_DONTWRITEHEADER 0
#define WELLDATA_WRITEHEADER     1

/*     Print Flags      */
#define WELLDATA_PRINTPHYSICAL  0x0001
#define WELLDATA_PRINTVALUES    0x0002
#define WELLDATA_PRINTSTATS     0x0004

#endif
