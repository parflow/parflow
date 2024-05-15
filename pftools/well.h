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
* Header file for `well.c'
*
* (C) 1996 Regents of the University of California.
*
* see info_header.h for complete information
*
*                               History
*-----------------------------------------------------------------------------
* $Revision: 1.5 $
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#ifndef WELL_HEADER
#define WELL_HEADER

#include "general.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Data structure length constants
 *--------------------------------------------------------------------------*/

/*       Length        */
#define MAXNAMELEN      2048

/*--------------------------------------------------------------------------
 * Well Data constants
 *--------------------------------------------------------------------------*/

/*       Actions        */
#define INJECTION_WELL  0
#define EXTRACTION_WELL 1


/*----------------------------------------------------------------
 * Background structure
 *----------------------------------------------------------------*/

typedef struct {
  double X, Y, Z;
  int NX, NY, NZ;
  double DX, DY, DZ;
} Background;

/*----------------------------------------------------------------
 * Problem Data structure
 *----------------------------------------------------------------*/

typedef struct {
  int num_phases, num_components, num_wells;
} ProblemData;

/*----------------------------------------------------------------
 * Well Header Values structure
 *----------------------------------------------------------------*/

typedef struct {
  int number;
  char          *name;
  double x_lower, y_lower, z_lower;
  double x_upper, y_upper, z_upper;
  double diameter;
  int type;
  int action;
} WellDataHeader;

/*----------------------------------------------------------------
 * Well Physical Values structure
 *----------------------------------------------------------------*/

typedef struct {
  int number;
  int ix, iy, iz;
  int nx, ny, nz;
  int rx, ry, rz;
} WellDataPhysical;

/*----------------------------------------------------------------
 * Well Data Values structure
 *----------------------------------------------------------------*/

typedef struct {
  double        *phase_values;
  double        *saturation_values;
  double        *component_values;
  double        *component_fractions;
} WellDataValue;

/*----------------------------------------------------------------
 * Well Data Stats structure
 *----------------------------------------------------------------*/

typedef struct {
  double        *phase_stats;
  double        *saturation_stats;
  double        *component_stats;
  double        *concentration_stats;
} WellDataStat;


/*--------------------------------------------------------------------------
 * Accessor macros: Background
 *--------------------------------------------------------------------------*/
#define BackgroundX(background)  ((background)->X)

#define BackgroundY(background)  ((background)->Y)

#define BackgroundZ(background)  ((background)->Z)

#define BackgroundNX(background) ((background)->NX)

#define BackgroundNY(background) ((background)->NY)

#define BackgroundNZ(background) ((background)->NZ)

#define BackgroundDX(background) ((background)->DX)

#define BackgroundDY(background) ((background)->DY)

#define BackgroundDZ(background) ((background)->DZ)

/*--------------------------------------------------------------------------
 * Accessor macros: ProblemData
 *--------------------------------------------------------------------------*/
#define ProblemDataNumPhases(problem_data) \
  ((problem_data)->num_phases)

#define ProblemDataNumComponents(problem_data) \
  ((problem_data)->num_components)

#define ProblemDataNumWells(problem_data) \
  ((problem_data)->num_wells)

/*--------------------------------------------------------------------------
 * Accessor macros: WellDataHeader
 *--------------------------------------------------------------------------*/
#define WellDataHeaderNumber(well_data_header) \
  ((well_data_header)->number)

#define WellDataHeaderName(well_data_header) \
  ((well_data_header)->name)

#define WellDataHeaderXLower(well_data_header) \
  ((well_data_header)->x_lower)

#define WellDataHeaderYLower(well_data_header) \
  ((well_data_header)->y_lower)

#define WellDataHeaderZLower(well_data_header) \
  ((well_data_header)->z_lower)

#define WellDataHeaderXUpper(well_data_header) \
  ((well_data_header)->x_upper)

#define WellDataHeaderYUpper(well_data_header) \
  ((well_data_header)->y_upper)

#define WellDataHeaderZUpper(well_data_header) \
  ((well_data_header)->z_upper)

#define WellDataHeaderDiameter(well_data_header) \
  ((well_data_header)->diameter)

#define WellDataHeaderType(well_data_header) \
  ((well_data_header)->type)

#define WellDataHeaderAction(well_data_header) \
  ((well_data_header)->action)

/*--------------------------------------------------------------------------
 * Accessor macros: WellDataPhysical
 *--------------------------------------------------------------------------*/
#define WellDataPhysicalNumber(well_data_physical) \
  ((well_data_physical)->number)

#define WellDataPhysicalIX(well_data_physical) \
  ((well_data_physical)->ix)

#define WellDataPhysicalIY(well_data_physical) \
  ((well_data_physical)->iy)

#define WellDataPhysicalIZ(well_data_physical) \
  ((well_data_physical)->iz)

#define WellDataPhysicalNX(well_data_physical) \
  ((well_data_physical)->nx)

#define WellDataPhysicalNY(well_data_physical) \
  ((well_data_physical)->ny)

#define WellDataPhysicalNZ(well_data_physical) \
  ((well_data_physical)->nz)

#define WellDataPhysicalRX(well_data_physical) \
  ((well_data_physical)->rx)

#define WellDataPhysicalRY(well_data_physical) \
  ((well_data_physical)->ry)

#define WellDataPhysicalRZ(well_data_physical) \
  ((well_data_physical)->rz)

/*--------------------------------------------------------------------------
 * Accessor macros: WellDataValue
 *--------------------------------------------------------------------------*/
#define WellDataValuePhaseValues(well_data_value) \
  ((well_data_value)->phase_values)
#define WellDataValuePhaseValue(well_data_value, i) \
  ((well_data_value)->phase_values[i])

#define WellDataValueSaturationValues(well_data_value) \
  ((well_data_value)->saturation_values)
#define WellDataValueSaturationValue(well_data_value, i) \
  ((well_data_value)->saturation_values[i])

#define WellDataValueComponentValues(well_data_value) \
  ((well_data_value)->component_values)
#define WellDataValueComponentValue(well_data_value, i) \
  ((well_data_value)->component_values[i])

#define WellDataValueComponentFractions(well_data_value) \
  ((well_data_value)->component_fractions)
#define WellDataValueComponentFraction(well_data_value, i) \
  ((well_data_value)->component_fractions[i])

/*--------------------------------------------------------------------------
 * Accessor macros: WellDataStat
 *--------------------------------------------------------------------------*/
#define WellDataStatPhaseStats(well_data_stat) \
  ((well_data_stat)->phase_stats)
#define WellDataStatPhaseStat(well_data_stat, i) \
  ((well_data_stat)->phase_stats[i])

#define WellDataStatSaturationStats(well_data_stat) \
  ((well_data_stat)->saturation_stats)
#define WellDataStatSaturationStat(well_data_stat, i) \
  ((well_data_stat)->saturation_stats[i])

#define WellDataStatComponentStats(well_data_stat) \
  ((well_data_stat)->component_stats)
#define WellDataStatComponentStat(well_data_stat, i) \
  ((well_data_stat)->component_stats[i])

#define WellDataStatConcentrationStats(well_data_stat) \
  ((well_data_stat)->concentration_stats)
#define WellDataStatConcentrationStat(well_data_stat, i) \
  ((well_data_stat)->concentration_stats[i])


/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

/* well.c */
Background * NewBackground();
void FreeBackground(Background *background);
void ReadBackground(FILE *fd, Background *background);
void WriteBackground(FILE *fd, Background *background);
void PrintBackground(Background *background);

ProblemData *NewProblemData();
void FreeProblemData(ProblemData *problem_data);
void ReadProblemData(FILE *fd, ProblemData *problem_data);
void WriteProblemData(FILE *fd, ProblemData *problem_data);
void PrintProblemData(ProblemData *problem_data);

WellDataHeader *NewWellDataHeader();
void FreeWellDataHeader(WellDataHeader *well_data_header);
void ReadWellDataHeader(FILE *fd, WellDataHeader *well_data_header);
void WriteWellDataHeader(FILE *fd, WellDataHeader *well_data_header);
void PrintWellDataHeader(WellDataHeader *well_data_header);

WellDataPhysical *NewWellDataPhysical();
void FreeWellDataPhysical(WellDataPhysical *well_data_physical);
void ReadWellDataPhysical(FILE *fd, WellDataPhysical *well_data_physical);
void WriteWellDataPhysical(FILE *fd, WellDataPhysical *well_data_physical);
void PrintWellDataPhysical(WellDataPhysical *well_data_physical);
void CopyWellDataPhysical(WellDataPhysical *updt_well_data_physical, WellDataPhysical *well_data_physical);

WellDataValue *NewWellDataValue(int num_phases, int num_components);
void FreeWellDataValue(WellDataValue *well_data_value);
void ReadWellDataValue(FILE *fd, WellDataValue *well_data_value, int action, int type, int num_phases, int num_components);
void WriteWellDataValue(FILE *fd, WellDataValue *well_data_value, int action, int type, int num_phases, int num_components);
void PrintWellDataValue(WellDataValue *well_data_value, int action, int type, int num_phases, int num_components);
void CopyWellDataValue(WellDataValue *updt_well_data_value, WellDataValue *well_data_value, int action, int type, int num_phases, int num_components);

WellDataStat *NewWellDataStat(int num_phases, int num_components);
void FreeWellDataStat(WellDataStat *well_data_stat);
void ReadWellDataStat(FILE *fd, WellDataStat *well_data_stat, int num_phases, int num_components);
void WriteWellDataStat(FILE *fd, WellDataStat *well_data_stat, int num_phases, int num_components);
void PrintWellDataStat(WellDataStat *well_data_stat, int num_phases, int num_components);
void InitWellDataStat(WellDataStat *well_data_stat, int num_phases, int num_components);
void UpdateWellDataStat(WellDataStat *updt_well_data_stat, WellDataStat *well_data_stat, int num_phases, int num_components);
void CopyWellDataStat(WellDataStat *updt_well_data_stat, WellDataStat *well_data_stat, int num_phases, int num_components);

#ifdef __cplusplus
}
#endif

#endif
