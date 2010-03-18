/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER*/

#ifndef _BC_PRESSURE_HEADER
#define _BC_PRESSURE_HEADER

/*----------------------------------------------------------------
 * BCPressure Values
 *----------------------------------------------------------------*/

/* Pressure condition, constant */
typedef struct
{
   int     reference_solid;
   int     reference_patch;
   double  value;
   double *value_at_interfaces;
} BCPressureType0;

/* Pressure condition, piecewise linear */
typedef struct
{
   double   xlower;
   double   ylower;
   double   xupper;
   double   yupper;
   int      num_points;
   double  *points;
   double  *values;
   double  *value_at_interfaces;
} BCPressureType1;

/* Flux condition, constant flux rate */
typedef struct
{
   double  value;
} BCPressureType2;

/* Flux condition, constant volumetric rate */
typedef struct
{
   double  value;
} BCPressureType3;

/* Pressure condition, read from file (temporary) */
typedef struct
{
   char  *filename;
} BCPressureType4;

/* Flux condition, read from file (temporary) */
typedef struct
{
   char  *filename;
} BCPressureType5;

/* Pressure Dir. condition for MATH problems */
typedef struct
{
   int    function_type;
} BCPressureType6;

/* //sk Overland flow, constant rain*/
typedef struct
{
   double  value;
} BCPressureType7;


/*----------------------------------------------------------------
 * BCPressure Data structure
 *----------------------------------------------------------------*/

typedef struct
{
   int                 num_phases;

   int                 num_patches;

   int                *types;
   int                *patch_indexes;
   int                *cycle_numbers;
   int                *bc_types;
   void             ***values;

   /* time info */
   TimeCycleData      *time_cycle_data;
} BCPressureData;

/*--------------------------------------------------------------------------
 * Accessor macros: BCPressureValues
 *--------------------------------------------------------------------------*/
#define BCPressureType0Value(type0)\
        ((type0) -> value)

#define BCPressureType0RefSolid(type0)\
        ((type0) -> reference_solid)
#define BCPressureType0RefPatch(type0)\
        ((type0) -> reference_patch)

#define BCPressureType0ValueAtInterfaces(type0)\
        ((type0) -> value_at_interfaces)
#define BCPressureType0ValueAtInterface(type0,i)\
        ((type0) -> value_at_interfaces[i-1])
/*--------------------------------------------------------------------------*/
#define BCPressureType1XLower(type1)\
        ((type1) -> xlower)

#define BCPressureType1YLower(type1)\
        ((type1) -> ylower)

#define BCPressureType1XUpper(type1)\
        ((type1) -> xupper)

#define BCPressureType1YUpper(type1)\
        ((type1) -> yupper)

#define BCPressureType1NumPoints(type1)\
        ((type1) -> num_points)

#define BCPressureType1Points(type1)\
        ((type1) -> points)
#define BCPressureType1Point(type1,i)\
        ((type1) -> points[i])

#define BCPressureType1Values(type1)\
        ((type1) -> values)
#define BCPressureType1Value(type1,i)\
        ((type1) -> values[i])

#define BCPressureType1ValueAtInterfaces(type1)\
        ((type1) -> value_at_interfaces)
#define BCPressureType1ValueAtInterface(type1,i)\
        ((type1) -> value_at_interfaces[i-1])
/*--------------------------------------------------------------------------*/
#define BCPressureType2Value(type2)\
        ((type2) -> value)
/*--------------------------------------------------------------------------*/
#define BCPressureType3Value(type3)\
        ((type3) -> value)

/*--------------------------------------------------------------------------*/
#define BCPressureType4FileName(type4)\
        ((type4) -> filename)

/*--------------------------------------------------------------------------*/
#define BCPressureType5FileName(type5)\
        ((type5) -> filename)

/*--------------------------------------------------------------------------*/
#define BCPressureType6FunctionType(type6)\
        ((type6) -> function_type)

/*--------------------------------------------------------------------------*/
#define BCPressureType7Value(type7)\
        ((type7) -> value)

/*--------------------------------------------------------------------------
 * Accessor macros: BCPressureData
 *--------------------------------------------------------------------------*/
#define BCPressureDataNumPhases(bc_pressure_data)\
        ((bc_pressure_data) -> num_phases)

#define BCPressureDataNumPatches(bc_pressure_data)\
        ((bc_pressure_data) -> num_patches)

#define BCPressureDataTypes(bc_pressure_data)\
        ((bc_pressure_data) -> types)
#define BCPressureDataType(bc_pressure_data,i)\
        ((bc_pressure_data) -> types[i])

#define BCPressureDataCycleNumbers(bc_pressure_data)\
        ((bc_pressure_data) -> cycle_numbers)
#define BCPressureDataCycleNumber(bc_pressure_data,i)\
        ((bc_pressure_data) -> cycle_numbers[i])

#define BCPressureDataPatchIndexes(bc_pressure_data)\
        ((bc_pressure_data) -> patch_indexes)
#define BCPressureDataPatchIndex(bc_pressure_data,i)\
        ((bc_pressure_data) -> patch_indexes[i])

#define BCPressureDataBCTypes(bc_pressure_data)\
        ((bc_pressure_data) -> bc_types)
#define BCPressureDataBCType(bc_pressure_data,i)\
        ((bc_pressure_data) -> bc_types[i])

#define BCPressureDataValues(bc_pressure_data)\
        ((bc_pressure_data) -> values)
#define BCPressureDataIntervalValues(bc_pressure_data,i)\
        ((bc_pressure_data) -> values[i])
#define BCPressureDataIntervalValue(bc_pressure_data,i,interval_number)\
        (((bc_pressure_data) -> values[i])[interval_number])

#define BCPressureDataTimeCycleData(bc_pressure_data)\
        ((bc_pressure_data) -> time_cycle_data)

/*--------------------------------------------------------------------------
 * BCPressure Data constants used in the program.
 *--------------------------------------------------------------------------*/

#endif
