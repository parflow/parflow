/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.5 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * Header file for `well.c'
 *
 * (C) 1996 Regents of the University of California.
 *
 * see info_header.h for complete information
 *
 *                               History
 *-----------------------------------------------------------------------------
 $Revision: 1.5 $
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#ifndef WELL_HEADER
#define WELL_HEADER

#include <stdio.h>
#include <string.h>
#include <malloc.h>

#include "general.h"

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

typedef struct
{
   double          X,  Y,  Z;
   int            NX, NY, NZ;
   double         DX, DY, DZ;
} Background;

/*----------------------------------------------------------------
 * Problem Data structure
 *----------------------------------------------------------------*/

typedef struct
{
   int            num_phases, num_components, num_wells;
} ProblemData;

/*----------------------------------------------------------------
 * Well Header Values structure
 *----------------------------------------------------------------*/

typedef struct
{
   int            number;
   char          *name;
   double         x_lower, y_lower, z_lower;
   double         x_upper, y_upper, z_upper;
   double         diameter;
   int            type;
   int            action;
} WellDataHeader;

/*----------------------------------------------------------------
 * Well Physical Values structure
 *----------------------------------------------------------------*/

typedef struct
{
   int            number;
   int            ix, iy, iz;
   int            nx, ny, nz;
   int            rx, ry, rz;
} WellDataPhysical;

/*----------------------------------------------------------------
 * Well Data Values structure
 *----------------------------------------------------------------*/

typedef struct
{
   double        *phase_values;
   double        *saturation_values;
   double        *component_values;
   double        *component_fractions;
} WellDataValue;

/*----------------------------------------------------------------
 * Well Data Stats structure
 *----------------------------------------------------------------*/

typedef struct
{
   double        *phase_stats;
   double        *saturation_stats;
   double        *component_stats;
   double        *concentration_stats;
} WellDataStat;


/*--------------------------------------------------------------------------
 * Accessor macros: Background
 *--------------------------------------------------------------------------*/
#define BackgroundX(background)  ((background) ->  X)

#define BackgroundY(background)  ((background) ->  Y)

#define BackgroundZ(background)  ((background) ->  Z)

#define BackgroundNX(background) ((background) -> NX)

#define BackgroundNY(background) ((background) -> NY)

#define BackgroundNZ(background) ((background) -> NZ)

#define BackgroundDX(background) ((background) -> DX)

#define BackgroundDY(background) ((background) -> DY)

#define BackgroundDZ(background) ((background) -> DZ)

/*--------------------------------------------------------------------------
 * Accessor macros: ProblemData
 *--------------------------------------------------------------------------*/
#define ProblemDataNumPhases(problem_data)\
        ((problem_data) -> num_phases)

#define ProblemDataNumComponents(problem_data)\
        ((problem_data) -> num_components)

#define ProblemDataNumWells(problem_data)\
        ((problem_data) -> num_wells)

/*--------------------------------------------------------------------------
 * Accessor macros: WellDataHeader
 *--------------------------------------------------------------------------*/
#define WellDataHeaderNumber(well_data_header)\
        ((well_data_header) -> number)

#define WellDataHeaderName(well_data_header)\
        ((well_data_header) -> name)

#define WellDataHeaderXLower(well_data_header)\
        ((well_data_header) -> x_lower)

#define WellDataHeaderYLower(well_data_header)\
        ((well_data_header) -> y_lower)

#define WellDataHeaderZLower(well_data_header)\
        ((well_data_header) -> z_lower)

#define WellDataHeaderXUpper(well_data_header)\
        ((well_data_header) -> x_upper)

#define WellDataHeaderYUpper(well_data_header)\
        ((well_data_header) -> y_upper)

#define WellDataHeaderZUpper(well_data_header)\
        ((well_data_header) -> z_upper)

#define WellDataHeaderDiameter(well_data_header)\
        ((well_data_header) -> diameter)

#define WellDataHeaderType(well_data_header)\
        ((well_data_header) -> type)

#define WellDataHeaderAction(well_data_header)\
        ((well_data_header) -> action)

/*--------------------------------------------------------------------------
 * Accessor macros: WellDataPhysical
 *--------------------------------------------------------------------------*/
#define WellDataPhysicalNumber(well_data_physical)\
        ((well_data_physical) -> number)

#define WellDataPhysicalIX(well_data_physical)\
        ((well_data_physical) -> ix)

#define WellDataPhysicalIY(well_data_physical)\
        ((well_data_physical) -> iy)

#define WellDataPhysicalIZ(well_data_physical)\
        ((well_data_physical) -> iz)

#define WellDataPhysicalNX(well_data_physical)\
        ((well_data_physical) -> nx)

#define WellDataPhysicalNY(well_data_physical)\
        ((well_data_physical) -> ny)

#define WellDataPhysicalNZ(well_data_physical)\
        ((well_data_physical) -> nz)

#define WellDataPhysicalRX(well_data_physical)\
        ((well_data_physical) -> rx)

#define WellDataPhysicalRY(well_data_physical)\
        ((well_data_physical) -> ry)

#define WellDataPhysicalRZ(well_data_physical)\
        ((well_data_physical) -> rz)

/*--------------------------------------------------------------------------
 * Accessor macros: WellDataValue
 *--------------------------------------------------------------------------*/
#define WellDataValuePhaseValues(well_data_value)\
        ((well_data_value) -> phase_values)
#define WellDataValuePhaseValue(well_data_value,i)\
        ((well_data_value) -> phase_values[i])

#define WellDataValueSaturationValues(well_data_value)\
        ((well_data_value) -> saturation_values)
#define WellDataValueSaturationValue(well_data_value,i)\
        ((well_data_value) -> saturation_values[i])

#define WellDataValueComponentValues(well_data_value)\
        ((well_data_value) -> component_values)
#define WellDataValueComponentValue(well_data_value,i)\
        ((well_data_value) -> component_values[i])

#define WellDataValueComponentFractions(well_data_value)\
        ((well_data_value) -> component_fractions)
#define WellDataValueComponentFraction(well_data_value,i)\
        ((well_data_value) -> component_fractions[i])

/*--------------------------------------------------------------------------
 * Accessor macros: WellDataStat
 *--------------------------------------------------------------------------*/
#define WellDataStatPhaseStats(well_data_stat)\
        ((well_data_stat) -> phase_stats)
#define WellDataStatPhaseStat(well_data_stat,i)\
        ((well_data_stat) -> phase_stats[i])

#define WellDataStatSaturationStats(well_data_stat)\
        ((well_data_stat) -> saturation_stats)
#define WellDataStatSaturationStat(well_data_stat,i)\
        ((well_data_stat) -> saturation_stats[i])

#define WellDataStatComponentStats(well_data_stat)\
        ((well_data_stat) -> component_stats)
#define WellDataStatComponentStat(well_data_stat,i)\
        ((well_data_stat) -> component_stats[i])

#define WellDataStatConcentrationStats(well_data_stat)\
        ((well_data_stat) -> concentration_stats)
#define WellDataStatConcentrationStat(well_data_stat,i)\
        ((well_data_stat) -> concentration_stats[i])


/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

#ifdef __STDC__
# define        ANSI_PROTO(s) s
#else
# define ANSI_PROTO(s) ()
#endif

/* well.c */
Background *NewBackground ANSI_PROTO(());
void FreeBackground ANSI_PROTO((Background *background));
void ReadBackground ANSI_PROTO((FILE *fd, Background *background));
void WriteBackground ANSI_PROTO((FILE *fd, Background *background));
void PrintBackground ANSI_PROTO((Background *background));

ProblemData *NewProblemData ANSI_PROTO(());
void FreeProblemData ANSI_PROTO((ProblemData *problem_data));
void ReadProblemData ANSI_PROTO((FILE *fd, ProblemData *problem_data));
void WriteProblemData ANSI_PROTO((FILE *fd, ProblemData *problem_data));
void PrintProblemData ANSI_PROTO((ProblemData *problem_data));

WellDataHeader *NewWellDataHeader ANSI_PROTO(());
void FreeWellDataHeader ANSI_PROTO((WellDataHeader *well_data_header));
void ReadWellDataHeader ANSI_PROTO((FILE *fd, WellDataHeader *well_data_header));
void WriteWellDataHeader ANSI_PROTO((FILE *fd, WellDataHeader *well_data_header));
void PrintWellDataHeader ANSI_PROTO((WellDataHeader *well_data_header));

WellDataPhysical *NewWellDataPhysical ANSI_PROTO(());
void FreeWellDataPhysical ANSI_PROTO((WellDataPhysical *well_data_physical));
void ReadWellDataPhysical ANSI_PROTO((FILE *fd, WellDataPhysical *well_data_physical));
void WriteWellDataPhysical ANSI_PROTO((FILE *fd, WellDataPhysical *well_data_physical));
void PrintWellDataPhysical ANSI_PROTO((WellDataPhysical *well_data_physical));
void CopyWellDataPhysical ANSI_PROTO((WellDataPhysical *updt_well_data_physical, WellDataPhysical *well_data_physical));

WellDataValue *NewWellDataValue ANSI_PROTO((int num_phases, int num_components));
void FreeWellDataValue ANSI_PROTO((WellDataValue *well_data_value));
void ReadWellDataValue ANSI_PROTO((FILE *fd, WellDataValue *well_data_value, int action, int type, int num_phases, int num_components));
void WriteWellDataValue ANSI_PROTO((FILE *fd, WellDataValue *well_data_value, int action, int type, int num_phases, int num_components));
void PrintWellDataValue ANSI_PROTO((WellDataValue *well_data_value, int action, int type, int num_phases, int num_components));
void CopyWellDataValue ANSI_PROTO((WellDataValue *updt_well_data_value, WellDataValue *well_data_value, int action, int type, int num_phases, int num_components));

WellDataStat *NewWellDataStat ANSI_PROTO((int num_phases, int num_components));
void FreeWellDataStat ANSI_PROTO((WellDataStat *well_data_stat));
void ReadWellDataStat ANSI_PROTO((FILE *fd, WellDataStat *well_data_stat, int num_phases, int num_components));
void WriteWellDataStat ANSI_PROTO((FILE *fd, WellDataStat *well_data_stat, int num_phases, int num_components));
void PrintWellDataStat ANSI_PROTO((WellDataStat *well_data_stat, int num_phases, int num_components));
void InitWellDataStat ANSI_PROTO((WellDataStat *well_data_stat, int num_phases, int num_components));
void UpdateWellDataStat ANSI_PROTO((WellDataStat *updt_well_data_stat, WellDataStat *well_data_stat, int num_phases, int num_components));
void CopyWellDataStat ANSI_PROTO((WellDataStat *updt_well_data_stat, WellDataStat *well_data_stat, int num_phases, int num_components));

#undef ANSI_PROTO

#endif
