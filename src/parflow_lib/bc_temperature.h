/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#ifndef _BC_TEMPERATURE_HEADER
#define _BC_TEMPERATURE_HEADER

/*----------------------------------------------------------------
 * BCTemperature Values
 *----------------------------------------------------------------*/

/* Pressure condition, constant */
typedef struct
{
   int     reference_solid;
   int     reference_patch;
   double  value;
   double *value_at_interfaces;
} BCTemperatureType0;

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
} BCTemperatureType1;

/* Flux condition, constant flux rate */
typedef struct
{
   double  value;
} BCTemperatureType2;

/* Flux condition, constant volumetric rate */
typedef struct
{
   double  value;
} BCTemperatureType3;

/* Pressure condition, read from file (temporary) */
typedef struct
{
   char  *filename;
} BCTemperatureType4;

/* Flux condition, read from file (temporary) */
typedef struct
{
   char  *filename;
} BCTemperatureType5;

/* Pressure Dir. condition for MATH problems */
typedef struct
{
   int    function_type;
} BCTemperatureType6;

/* //sk Overland flow, constant rain*/
typedef struct
{
   double  value;
} BCTemperatureType7;


/*----------------------------------------------------------------
 * BCTemperature Data structure
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
} BCTemperatureData;

/*--------------------------------------------------------------------------
 * Accessor macros: BCTemperatureValues
 *--------------------------------------------------------------------------*/
#define BCTemperatureType0Value(type0)\
        ((type0) -> value)

#define BCTemperatureType0RefSolid(type0)\
        ((type0) -> reference_solid)
#define BCTemperatureType0RefPatch(type0)\
        ((type0) -> reference_patch)

#define BCTemperatureType0ValueAtInterfaces(type0)\
        ((type0) -> value_at_interfaces)
#define BCTemperatureType0ValueAtInterface(type0,i)\
        ((type0) -> value_at_interfaces[i-1])
/*--------------------------------------------------------------------------*/
#define BCTemperatureType1XLower(type1)\
        ((type1) -> xlower)

#define BCTemperatureType1YLower(type1)\
        ((type1) -> ylower)

#define BCTemperatureType1XUpper(type1)\
        ((type1) -> xupper)

#define BCTemperatureType1YUpper(type1)\
        ((type1) -> yupper)

#define BCTemperatureType1NumPoints(type1)\
        ((type1) -> num_points)

#define BCTemperatureType1Points(type1)\
        ((type1) -> points)
#define BCTemperatureType1Point(type1,i)\
        ((type1) -> points[i])

#define BCTemperatureType1Values(type1)\
        ((type1) -> values)
#define BCTemperatureType1Value(type1,i)\
        ((type1) -> values[i])

#define BCTemperatureType1ValueAtInterfaces(type1)\
        ((type1) -> value_at_interfaces)
#define BCTemperatureType1ValueAtInterface(type1,i)\
        ((type1) -> value_at_interfaces[i-1])
/*--------------------------------------------------------------------------*/
#define BCTemperatureType2Value(type2)\
        ((type2) -> value)
/*--------------------------------------------------------------------------*/
#define BCTemperatureType3Value(type3)\
        ((type3) -> value)

/*--------------------------------------------------------------------------*/
#define BCTemperatureType4FileName(type4)\
        ((type4) -> filename)

/*--------------------------------------------------------------------------*/
#define BCTemperatureType5FileName(type5)\
        ((type5) -> filename)

/*--------------------------------------------------------------------------*/
#define BCTemperatureType6FunctionType(type6)\
        ((type6) -> function_type)

/*--------------------------------------------------------------------------*/
#define BCTemperatureType7Value(type7)\
        ((type7) -> value)

/*--------------------------------------------------------------------------
 * Accessor macros: BCTemperatureData
 *--------------------------------------------------------------------------*/
#define BCTemperatureDataNumPhases(bc_temperature_data)\
        ((bc_temperature_data) -> num_phases)

#define BCTemperatureDataNumPatches(bc_temperature_data)\
        ((bc_temperature_data) -> num_patches)

#define BCTemperatureDataTypes(bc_temperature_data)\
        ((bc_temperature_data) -> types)
#define BCTemperatureDataType(bc_temperature_data,i)\
        ((bc_temperature_data) -> types[i])

#define BCTemperatureDataCycleNumbers(bc_temperature_data)\
        ((bc_temperature_data) -> cycle_numbers)
#define BCTemperatureDataCycleNumber(bc_temperature_data,i)\
        ((bc_temperature_data) -> cycle_numbers[i])

#define BCTemperatureDataPatchIndexes(bc_temperature_data)\
        ((bc_temperature_data) -> patch_indexes)
#define BCTemperatureDataPatchIndex(bc_temperature_data,i)\
        ((bc_temperature_data) -> patch_indexes[i])

#define BCTemperatureDataBCTypes(bc_temperature_data)\
        ((bc_temperature_data) -> bc_types)
#define BCTemperatureDataBCType(bc_temperature_data,i)\
        ((bc_temperature_data) -> bc_types[i])

#define BCTemperatureDataValues(bc_temperature_data)\
        ((bc_temperature_data) -> values)
#define BCTemperatureDataIntervalValues(bc_temperature_data,i)\
        ((bc_temperature_data) -> values[i])
#define BCTemperatureDataIntervalValue(bc_temperature_data,i,interval_number)\
        (((bc_temperature_data) -> values[i])[interval_number])

#define BCTemperatureDataTimeCycleData(bc_temperature_data)\
        ((bc_temperature_data) -> time_cycle_data)

/*--------------------------------------------------------------------------
 * BCTemperature Data constants used in the program.
 *--------------------------------------------------------------------------*/

#endif
