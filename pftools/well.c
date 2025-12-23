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

#include "well.h"

/*****************************************************************************
*
* The functions in this file are for manipulating the WellData structure
*   in ProblemData and work in conjunction with the WellPackage module.
*
*****************************************************************************/

/*--------------------------------------------------------------------------
 * NewBackground
 *--------------------------------------------------------------------------*/

Background *NewBackground()
{
  Background *background;

  background = talloc(Background, 1);

  return background;
}

/*--------------------------------------------------------------------------
 * FreeBackground
 *--------------------------------------------------------------------------*/

void FreeBackground(Background *background)
{
  if (background)
  {
    free(background);
  }
}

/*--------------------------------------------------------------------------
 * ReadBackground
 *--------------------------------------------------------------------------*/

void ReadBackground(
                    FILE *      fd,
                    Background *background)
{
  fscanf(fd, "%lf %lf %lf %d %d %d %lf %lf %lf\n",
         &BackgroundX(background),
         &BackgroundY(background),
         &BackgroundZ(background),
         &BackgroundNX(background),
         &BackgroundNY(background),
         &BackgroundNZ(background),
         &BackgroundDX(background),
         &BackgroundDY(background),
         &BackgroundDZ(background));
}

/*--------------------------------------------------------------------------
 * WriteBackground
 *--------------------------------------------------------------------------*/

void WriteBackground(
                     FILE *      fd,
                     Background *background)
{
  fprintf(fd, "%f %f %f %d %d %d %f %f %f\n",
          BackgroundX(background),
          BackgroundY(background),
          BackgroundZ(background),
          BackgroundNX(background),
          BackgroundNY(background),
          BackgroundNZ(background),
          BackgroundDX(background),
          BackgroundDY(background),
          BackgroundDZ(background));
}

/*--------------------------------------------------------------------------
 * PrintBackground
 *--------------------------------------------------------------------------*/

void PrintBackground(
                     Background *background)
{
  printf("%f %f %f %d %d %d %f %f %f\n",
         BackgroundX(background),
         BackgroundY(background),
         BackgroundZ(background),
         BackgroundNX(background),
         BackgroundNY(background),
         BackgroundNZ(background),
         BackgroundDX(background),
         BackgroundDY(background),
         BackgroundDZ(background));
}

/*--------------------------------------------------------------------------
 * NewProblemData
 *--------------------------------------------------------------------------*/

ProblemData *NewProblemData()
{
  ProblemData *problem_data;

  problem_data = talloc(ProblemData, 1);

  return problem_data;
}

/*--------------------------------------------------------------------------
 * FreeProblemData
 *--------------------------------------------------------------------------*/

void FreeProblemData(
                     ProblemData *problem_data)
{
  if (problem_data)
  {
    free(problem_data);
  }
}

/*--------------------------------------------------------------------------
 * ReadProblemData
 *--------------------------------------------------------------------------*/

void ReadProblemData(
                     FILE *       fd,
                     ProblemData *problem_data)
{
  fscanf(fd, "%d %d %d\n",
         &ProblemDataNumPhases(problem_data),
         &ProblemDataNumComponents(problem_data),
         &ProblemDataNumWells(problem_data));
}

/*--------------------------------------------------------------------------
 * WriteProblemData
 *--------------------------------------------------------------------------*/

void WriteProblemData(
                      FILE *       fd,
                      ProblemData *problem_data)
{
  fprintf(fd, "%d %d %d\n",
          ProblemDataNumPhases(problem_data),
          ProblemDataNumComponents(problem_data),
          ProblemDataNumWells(problem_data));
}

/*--------------------------------------------------------------------------
 * PrintProblemData
 *--------------------------------------------------------------------------*/

void PrintProblemData(
                      ProblemData *problem_data)
{
  printf("%d %d %d\n",
         ProblemDataNumPhases(problem_data),
         ProblemDataNumComponents(problem_data),
         ProblemDataNumWells(problem_data));
}

/*--------------------------------------------------------------------------
 * NewWellDataHeader
 *--------------------------------------------------------------------------*/

WellDataHeader *NewWellDataHeader()
{
  WellDataHeader *well_data_header;

  well_data_header = talloc(WellDataHeader, 1);

  WellDataHeaderName(well_data_header) = talloc(char, MAXNAMELEN);

  return well_data_header;
}

/*--------------------------------------------------------------------------
 * FreeWellDataHeader
 *--------------------------------------------------------------------------*/

void FreeWellDataHeader(
                        WellDataHeader *well_data_header)
{
  if (well_data_header)
  {
    if (WellDataHeaderName(well_data_header))
    {
      free(WellDataHeaderName(well_data_header));
    }
    free(well_data_header);
  }
}

/*--------------------------------------------------------------------------
 * ReadWellDataHeader
 *--------------------------------------------------------------------------*/

void ReadWellDataHeader(
                        FILE *          fd,
                        WellDataHeader *well_data_header)
{
  int i, string_length;
  char      *tmp_string;

  fscanf(fd, "%2d\n", &WellDataHeaderNumber(well_data_header));

  fscanf(fd, "%d\n", &string_length);
  string_length++;
  tmp_string = talloc(char, string_length);
  for (i = 0; i < string_length; i++)
  {
    fscanf(fd, "%c", &tmp_string[i]);
  }
  fscanf(fd, "\n");
  strcpy(WellDataHeaderName(well_data_header), tmp_string);
  free(tmp_string);

  fscanf(fd, "%lf %lf %lf %lf %lf %lf %lf\n",
         &WellDataHeaderXLower(well_data_header),
         &WellDataHeaderYLower(well_data_header),
         &WellDataHeaderZLower(well_data_header),
         &WellDataHeaderXUpper(well_data_header),
         &WellDataHeaderYUpper(well_data_header),
         &WellDataHeaderZUpper(well_data_header),
         &WellDataHeaderDiameter(well_data_header));

  fscanf(fd, "%d %d\n",
         &WellDataHeaderType(well_data_header),
         &WellDataHeaderAction(well_data_header));
}

/*--------------------------------------------------------------------------
 * WriteWellDataHeader
 *--------------------------------------------------------------------------*/

void WriteWellDataHeader(
                         FILE *          fd,
                         WellDataHeader *well_data_header)
{
  fprintf(fd, "%2d\n", WellDataHeaderNumber(well_data_header));

  fprintf(fd, "%s\n", WellDataHeaderName(well_data_header));

  fprintf(fd, "%f %f %f %f %f %f %f\n",
          WellDataHeaderXLower(well_data_header),
          WellDataHeaderYLower(well_data_header),
          WellDataHeaderZLower(well_data_header),
          WellDataHeaderXUpper(well_data_header),
          WellDataHeaderYUpper(well_data_header),
          WellDataHeaderZUpper(well_data_header),
          WellDataHeaderDiameter(well_data_header));

  fprintf(fd, "%1d %1d\n",
          WellDataHeaderType(well_data_header),
          WellDataHeaderAction(well_data_header));
}

/*--------------------------------------------------------------------------
 * PrintWellDataHeader
 *--------------------------------------------------------------------------*/

void PrintWellDataHeader(
                         WellDataHeader *well_data_header)
{
  printf("%2d\n", WellDataHeaderNumber(well_data_header));

  printf("%s\n", WellDataHeaderName(well_data_header));

  printf("%f %f %f %f %f %f %f\n",
         WellDataHeaderXLower(well_data_header),
         WellDataHeaderYLower(well_data_header),
         WellDataHeaderZLower(well_data_header),
         WellDataHeaderXUpper(well_data_header),
         WellDataHeaderYUpper(well_data_header),
         WellDataHeaderZUpper(well_data_header),
         WellDataHeaderDiameter(well_data_header));

  printf("%1d %1d\n",
         WellDataHeaderType(well_data_header),
         WellDataHeaderAction(well_data_header));
}

/*--------------------------------------------------------------------------
 * NewWellDataPhysical
 *--------------------------------------------------------------------------*/

WellDataPhysical *NewWellDataPhysical()
{
  WellDataPhysical *well_data_physical;

  well_data_physical = talloc(WellDataPhysical, 1);

  return well_data_physical;
}

/*--------------------------------------------------------------------------
 * FreeWellDataPhysical
 *--------------------------------------------------------------------------*/

void FreeWellDataPhysical(
                          WellDataPhysical *well_data_physical)
{
  if (well_data_physical)
  {
    free(well_data_physical);
  }
}

/*--------------------------------------------------------------------------
 * ReadWellDataPhysical
 *--------------------------------------------------------------------------*/

void ReadWellDataPhysical(
                          FILE *            fd,
                          WellDataPhysical *well_data_physical)
{
  fscanf(fd, "%d\n", &WellDataPhysicalNumber(well_data_physical));

  fscanf(fd, "%d %d %d %d %d %d %d %d %d\n",
         &WellDataPhysicalIX(well_data_physical),
         &WellDataPhysicalIY(well_data_physical),
         &WellDataPhysicalIZ(well_data_physical),
         &WellDataPhysicalNX(well_data_physical),
         &WellDataPhysicalNY(well_data_physical),
         &WellDataPhysicalNZ(well_data_physical),
         &WellDataPhysicalRX(well_data_physical),
         &WellDataPhysicalRY(well_data_physical),
         &WellDataPhysicalRZ(well_data_physical));
}

/*--------------------------------------------------------------------------
 * WriteWellDataPhysical
 *--------------------------------------------------------------------------*/

void WriteWellDataPhysical(
                           FILE *            fd,
                           WellDataPhysical *well_data_physical)
{
  fprintf(fd, "%2d\n", WellDataHeaderNumber(well_data_physical));

  fprintf(fd, "%d %d %d %d %d %d %d %d %d\n",
          WellDataPhysicalIX(well_data_physical),
          WellDataPhysicalIY(well_data_physical),
          WellDataPhysicalIZ(well_data_physical),
          WellDataPhysicalNX(well_data_physical),
          WellDataPhysicalNY(well_data_physical),
          WellDataPhysicalNZ(well_data_physical),
          WellDataPhysicalRX(well_data_physical),
          WellDataPhysicalRY(well_data_physical),
          WellDataPhysicalRZ(well_data_physical));
}

/*--------------------------------------------------------------------------
 * PrintWellDataPhysical
 *--------------------------------------------------------------------------*/

void PrintWellDataPhysical(
                           WellDataPhysical *well_data_physical)
{
  printf("%2d\n", WellDataHeaderNumber(well_data_physical));

  printf("%d %d %d %d %d %d %d %d %d\n",
         WellDataPhysicalIX(well_data_physical),
         WellDataPhysicalIY(well_data_physical),
         WellDataPhysicalIZ(well_data_physical),
         WellDataPhysicalNX(well_data_physical),
         WellDataPhysicalNY(well_data_physical),
         WellDataPhysicalNZ(well_data_physical),
         WellDataPhysicalRX(well_data_physical),
         WellDataPhysicalRY(well_data_physical),
         WellDataPhysicalRZ(well_data_physical));
}

/*--------------------------------------------------------------------------
 * CopyWellDataPhysical
 *--------------------------------------------------------------------------*/

void CopyWellDataPhysical(
                          WellDataPhysical *updt_well_data_physical,
                          WellDataPhysical *well_data_physical)
{
  WellDataHeaderNumber(updt_well_data_physical) = WellDataHeaderNumber(well_data_physical);

  WellDataPhysicalIX(updt_well_data_physical) = WellDataPhysicalIX(well_data_physical);
  WellDataPhysicalIY(updt_well_data_physical) = WellDataPhysicalIY(well_data_physical);
  WellDataPhysicalIZ(updt_well_data_physical) = WellDataPhysicalIZ(well_data_physical);
  WellDataPhysicalNX(updt_well_data_physical) = WellDataPhysicalNX(well_data_physical);
  WellDataPhysicalNY(updt_well_data_physical) = WellDataPhysicalNY(well_data_physical);
  WellDataPhysicalNZ(updt_well_data_physical) = WellDataPhysicalNZ(well_data_physical);
  WellDataPhysicalRX(updt_well_data_physical) = WellDataPhysicalRX(well_data_physical);
  WellDataPhysicalRY(updt_well_data_physical) = WellDataPhysicalRY(well_data_physical);
  WellDataPhysicalRZ(updt_well_data_physical) = WellDataPhysicalRZ(well_data_physical);
}

/*--------------------------------------------------------------------------
 * NewWellDataValue
 *--------------------------------------------------------------------------*/

WellDataValue *NewWellDataValue(
                                int num_phases,
                                int num_components)
{
  WellDataValue    *well_data_value;

  well_data_value = talloc(WellDataValue, 1);

  WellDataValuePhaseValues(well_data_value) = talloc(double, num_phases);
  WellDataValueSaturationValues(well_data_value) = talloc(double, num_phases);
  WellDataValueComponentValues(well_data_value) = talloc(double, num_phases * num_components);
  WellDataValueComponentFractions(well_data_value) = talloc(double, num_phases * num_components);

  return well_data_value;
}

/*--------------------------------------------------------------------------
 * FreeWellDataValue
 *--------------------------------------------------------------------------*/

void FreeWellDataValue(
                       WellDataValue *well_data_value)
{
  if (well_data_value)
  {
    if (WellDataValueComponentFractions(well_data_value))
    {
      free(WellDataValueComponentFractions(well_data_value));
    }
    if (WellDataValueComponentValues(well_data_value))
    {
      free(WellDataValueComponentValues(well_data_value));
    }
    if (WellDataValueSaturationValues(well_data_value))
    {
      free(WellDataValueSaturationValues(well_data_value));
    }
    if (WellDataValuePhaseValues(well_data_value))
    {
      free(WellDataValuePhaseValues(well_data_value));
    }
    free(well_data_value);
  }
}

/*--------------------------------------------------------------------------
 * ReadWellDataValue
 *--------------------------------------------------------------------------*/

void ReadWellDataValue(
                       FILE *         fd,
                       WellDataValue *well_data_value,
                       int            action,
                       int            type,
                       int            num_phases,
                       int            num_components)
{
  int i, j, indx, num_values;
  double value;

  if (type == 1)
  {
    num_values = num_phases;
  }
  else
  {
    num_values = 1;
  }

  for (i = 0; i < num_values; i++)
  {
    fscanf(fd, "%lf", &value);
    WellDataValuePhaseValue(well_data_value, i) = value;
  }

  if (action == 0)
  {
    for (i = 0; i < num_phases; i++)
    {
      fscanf(fd, "%lf", &value);
      WellDataValueSaturationValue(well_data_value, i) = value;
    }

    for (i = 0; i < num_phases; i++)
    {
      for (j = 0; j < num_components; j++)
      {
        indx = i * num_components + j;
        fscanf(fd, "%lf", &value);
        WellDataValueComponentValue(well_data_value, indx) = value;
      }
    }
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      fscanf(fd, "%lf", &value);
      WellDataValueComponentFraction(well_data_value, indx) = value;
    }
  }
}

/*--------------------------------------------------------------------------
 * WriteWellDataValue
 *--------------------------------------------------------------------------*/

void WriteWellDataValue(
                        FILE *         fd,
                        WellDataValue *well_data_value,
                        int            action,
                        int            type,
                        int            num_phases,
                        int            num_components)
{
  int i, j, indx, num_values;
  double value;

  if (type == 1)
  {
    num_values = num_phases;
  }
  else
  {
    num_values = 1;
  }

  for (i = 0; i < num_values; i++)
  {
    value = WellDataValuePhaseValue(well_data_value, i);
    fprintf(fd, " %f", value);
  }
  fprintf(fd, "\n");

  if (action == 0)
  {
    for (i = 0; i < num_phases; i++)
    {
      value = WellDataValueSaturationValue(well_data_value, i);
      fprintf(fd, " %f", value);
    }
    fprintf(fd, "\n");

    for (i = 0; i < num_phases; i++)
    {
      for (j = 0; j < num_components; j++)
      {
        indx = i * num_components + j;
        value = WellDataValueComponentValue(well_data_value, indx);
        fprintf(fd, " %f", value);
      }
    }
    fprintf(fd, "\n");
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      value = WellDataValueComponentFraction(well_data_value, indx);
      fprintf(fd, " %f", value);
    }
  }
  fprintf(fd, "\n");
}

/*--------------------------------------------------------------------------
 * PrintWellDataValue
 *--------------------------------------------------------------------------*/

void PrintWellDataValue(
                        WellDataValue *well_data_value,
                        int            action,
                        int            type,
                        int            num_phases,
                        int            num_components)
{
  int i, j, indx, num_values;
  double value;

  if (type == 1)
  {
    num_values = num_phases;
  }
  else
  {
    num_values = 1;
  }

  for (i = 0; i < num_values; i++)
  {
    value = WellDataValuePhaseValue(well_data_value, i);
    printf(" %f", value);
  }
  printf("\n");

  if (action == 0)
  {
    for (i = 0; i < num_phases; i++)
    {
      value = WellDataValueSaturationValue(well_data_value, i);
      printf(" %f", value);
    }
    printf("\n");

    for (i = 0; i < num_phases; i++)
    {
      for (j = 0; j < num_components; j++)
      {
        indx = i * num_components + j;
        value = WellDataValueComponentValue(well_data_value, indx);
        printf(" %f", value);
      }
    }
    printf("\n");
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      value = WellDataValueComponentFraction(well_data_value, indx);
      printf(" %f", value);
    }
  }
  printf("\n");
}

/*--------------------------------------------------------------------------
 * CopyWellDataValue
 *--------------------------------------------------------------------------*/

void CopyWellDataValue(
                       WellDataValue *updt_well_data_value,
                       WellDataValue *well_data_value,
                       int            action,
                       int            type,
                       int            num_phases,
                       int            num_components)
{
  int i, j, indx, num_values;

  if (type == 1)
  {
    num_values = num_phases;
  }
  else
  {
    num_values = 1;
  }

  for (i = 0; i < num_values; i++)
  {
    WellDataValuePhaseValue(updt_well_data_value, i) = WellDataValuePhaseValue(well_data_value, i);
  }

  if (action == 0)
  {
    for (i = 0; i < num_phases; i++)
    {
      WellDataValueSaturationValue(updt_well_data_value, i) = WellDataValueSaturationValue(well_data_value, i);
    }

    for (i = 0; i < num_phases; i++)
    {
      for (j = 0; j < num_components; j++)
      {
        indx = i * num_components + j;
        WellDataValueComponentValue(updt_well_data_value, indx) = WellDataValueComponentValue(well_data_value, indx);
      }
    }
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      WellDataValueComponentFraction(updt_well_data_value, indx) = WellDataValueComponentFraction(well_data_value, indx);
    }
  }
}

/*--------------------------------------------------------------------------
 * NewWellDataStat
 *--------------------------------------------------------------------------*/

WellDataStat *NewWellDataStat(
                              int num_phases,
                              int num_components)
{
  WellDataStat    *well_data_stat;

  well_data_stat = talloc(WellDataStat, 1);

  WellDataStatPhaseStats(well_data_stat) = talloc(double, num_phases);
  WellDataStatSaturationStats(well_data_stat) = talloc(double, num_phases);
  WellDataStatComponentStats(well_data_stat) = talloc(double, num_phases * num_components);
  WellDataStatConcentrationStats(well_data_stat) = talloc(double, num_phases * num_components);

  return well_data_stat;
}

/*--------------------------------------------------------------------------
 * FreeWellDataStat
 *--------------------------------------------------------------------------*/

void FreeWellDataStat(
                      WellDataStat *well_data_stat)
{
  if (well_data_stat)
  {
    if (WellDataStatConcentrationStats(well_data_stat))
    {
      free(WellDataStatConcentrationStats(well_data_stat));
    }
    if (WellDataStatComponentStats(well_data_stat))
    {
      free(WellDataStatComponentStats(well_data_stat));
    }
    if (WellDataStatSaturationStats(well_data_stat))
    {
      free(WellDataStatSaturationStats(well_data_stat));
    }
    if (WellDataStatPhaseStats(well_data_stat))
    {
      free(WellDataStatPhaseStats(well_data_stat));
    }
    free(well_data_stat);
  }
}

/*--------------------------------------------------------------------------
 * ReadWellDataStat
 *--------------------------------------------------------------------------*/

void ReadWellDataStat(
                      FILE *        fd,
                      WellDataStat *well_data_stat,
                      int           num_phases,
                      int           num_components)
{
  int i, j, indx;
  double value;

  for (i = 0; i < num_phases; i++)
  {
    fscanf(fd, "%lf", &value);
    WellDataStatPhaseStat(well_data_stat, i) = value;
  }

  for (i = 0; i < num_phases; i++)
  {
    fscanf(fd, "%lf", &value);
    WellDataStatSaturationStat(well_data_stat, i) = value;
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      fscanf(fd, "%lf", &value);
      WellDataStatComponentStat(well_data_stat, indx) = value;
    }
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      fscanf(fd, "%lf", &value);
      WellDataStatConcentrationStat(well_data_stat, indx) = value;
    }
  }
}

/*--------------------------------------------------------------------------
 * WriteWellDataStat
 *--------------------------------------------------------------------------*/

void WriteWellDataStat(
                       FILE *        fd,
                       WellDataStat *well_data_stat,
                       int           num_phases,
                       int           num_components)
{
  int i, j, indx;
  double value;

  for (i = 0; i < num_phases; i++)
  {
    value = WellDataStatPhaseStat(well_data_stat, i);
    fprintf(fd, " %f", value);
  }
  fprintf(fd, "\n");

  for (i = 0; i < num_phases; i++)
  {
    value = WellDataStatSaturationStat(well_data_stat, i);
    fprintf(fd, " %f", value);
  }
  fprintf(fd, "\n");

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      value = WellDataStatComponentStat(well_data_stat, indx);
      fprintf(fd, " %f", value);
    }
  }
  fprintf(fd, "\n");

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      value = WellDataStatConcentrationStat(well_data_stat, indx);
      fprintf(fd, " %f", value);
    }
  }
  fprintf(fd, "\n");
}

/*--------------------------------------------------------------------------
 * PrintWellDataStat
 *--------------------------------------------------------------------------*/

void PrintWellDataStat(
                       WellDataStat *well_data_stat,
                       int           num_phases,
                       int           num_components)
{
  int i, j, indx;
  double value;

  for (i = 0; i < num_phases; i++)
  {
    value = WellDataStatPhaseStat(well_data_stat, i);
    printf(" %f", value);
  }
  printf("\n");

  for (i = 0; i < num_phases; i++)
  {
    value = WellDataStatSaturationStat(well_data_stat, i);
    printf(" %f", value);
  }
  printf("\n");

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      value = WellDataStatComponentStat(well_data_stat, indx);
      printf(" %f", value);
    }
  }
  printf("\n");

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      value = WellDataStatConcentrationStat(well_data_stat, indx);
      printf(" %f", value);
    }
  }
  printf("\n");
}

/*--------------------------------------------------------------------------
 * InitWellDataStat
 *--------------------------------------------------------------------------*/

void InitWellDataStat(
                      WellDataStat *well_data_stat,
                      int           num_phases,
                      int           num_components)
{
  int i, j, indx;

  for (i = 0; i < num_phases; i++)
  {
    WellDataStatPhaseStat(well_data_stat, i) = 0.0;
  }

  for (i = 0; i < num_phases; i++)
  {
    WellDataStatSaturationStat(well_data_stat, i) = 0.0;
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      WellDataStatComponentStat(well_data_stat, indx) = 0.0;
    }
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      WellDataStatConcentrationStat(well_data_stat, indx) = 0.0;
    }
  }
}

/*--------------------------------------------------------------------------
 * UpdateWellDataStat
 *--------------------------------------------------------------------------*/

void UpdateWellDataStat(
                        WellDataStat *updt_well_data_stat,
                        WellDataStat *well_data_stat,
                        int           num_phases,
                        int           num_components)
{
  int i, j, indx;

  for (i = 0; i < num_phases; i++)
  {
    WellDataStatPhaseStat(updt_well_data_stat, i) += WellDataStatPhaseStat(well_data_stat, i);
  }

  for (i = 0; i < num_phases; i++)
  {
    WellDataStatSaturationStat(updt_well_data_stat, i) += WellDataStatSaturationStat(well_data_stat, i);
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      WellDataStatComponentStat(updt_well_data_stat, indx) += WellDataStatComponentStat(well_data_stat, indx);
    }
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      WellDataStatConcentrationStat(updt_well_data_stat, indx) += WellDataStatConcentrationStat(well_data_stat, indx);
    }
  }
}

/*--------------------------------------------------------------------------
 * CopyWellDataStat
 *--------------------------------------------------------------------------*/

void CopyWellDataStat(
                      WellDataStat *updt_well_data_stat,
                      WellDataStat *well_data_stat,
                      int           num_phases,
                      int           num_components)
{
  int i, j, indx;

  for (i = 0; i < num_phases; i++)
  {
    WellDataStatPhaseStat(updt_well_data_stat, i) = WellDataStatPhaseStat(well_data_stat, i);
  }

  for (i = 0; i < num_phases; i++)
  {
    WellDataStatSaturationStat(updt_well_data_stat, i) = WellDataStatSaturationStat(well_data_stat, i);
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      WellDataStatComponentStat(updt_well_data_stat, indx) = WellDataStatComponentStat(well_data_stat, indx);
    }
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      WellDataStatConcentrationStat(updt_well_data_stat, indx) = WellDataStatConcentrationStat(well_data_stat, indx);
    }
  }
}
