/*BHEADER*********************************************************************
* (c) 1995   The Regents of the University of California
*
* See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
* notice, contact person, and disclaimer.
*
* $Revision: 1.1.1.1 $
*********************************************************************EHEADER*/

#ifndef _BC_PRESSURE_HEADER
#define _BC_PRESSURE_HEADER

/*----------------------------------------------------------------
 * BCPressure Types
 * NOTE: These are not the values used to branch inside of
 *       BCStructPatchLoops, those are defined in problem_bc.h
 *----------------------------------------------------------------*/
#define DirEquilRefPatch 0
#define DirEquilPLinear  1
#define FluxConst        2
#define FluxVolumetric   3
#define PressureFile     4
#define FluxFile         5
#define ExactSolution    6
#define OverlandFlow     7
#define OverlandFlowPFB  8


/*----------------------------------------------------------------
 * Table to generate Type structs used in BCPressurePackage functions
 * These contain information for all interval steps to be read later
 *----------------------------------------------------------------*/
#define BC_TYPE_TABLE                               \
  BC_TYPE(DirEquilRefPatch, {                       \
      int reference_solid;                          \
      int reference_patch;                          \
      double *values;                               \
      double **value_at_interface;                  \
    })                                              \
  BC_TYPE(DirEquilPLinear, {                        \
      double *xlower;                               \
      double *ylower;                               \
      double *xupper;                               \
      double *yupper;                               \
      int    *num_points;                           \
      double **points;                              \
      double **values;                              \
      double **value_at_interface;                  \
    })                                              \
  BC_TYPE(FluxConst, {                              \
      double *values;                               \
    })                                              \
  BC_TYPE(FluxVolumetric, {                         \
      double *values;                               \
    })                                              \
  BC_TYPE(PressureFile, {                           \
      char **filenames;                             \
    })                                              \
  BC_TYPE(FluxFile, {                               \
      char **filenames;                             \
    })                                              \
  BC_TYPE(ExactSolution, {                          \
      int function_type;                            \
    })                                              \
  BC_TYPE(OverlandFlow, {                           \
      double *values;                               \
    })                                              \
  BC_TYPE(OverlandFlowPFB, {                        \
      char **filenames;                             \
    })

/*----------------------------------------------------------------
 * Constructor, getter, and setter for Type structs in BCPressurePackage
 *----------------------------------------------------------------*/
#define NewTypeStruct(type, var)                  \
  Type ## type * var = ctalloc(Type ## type, 1)
#define StoreTypeStruct(public_xtra, var, i)                  \
  (public_xtra)->data[(i)] = (void*)(var);
#define GetTypeStruct(type, var, public_xtra, i)              \
  Type ## type * var = (Type ## type *)(public_xtra->data[i])

// MCB: These two macros aren't really necessary but they do make the code cleaner
#define ForEachPatch(num_patches, i)           \
  for (i = 0; i < num_patches; i++)
#define ForEachInterval(interval_division, interval_number)  \
  for (interval_number = 0; interval_number < interval_division; interval_number++)


/*----------------------------------------------------------------
 * BCPressurePackage Actions
 *----------------------------------------------------------------*/
#define InputType(public_xtra, i)               \
  ((public_xtra)->input_types[(i)])

/* Send cases wrapped in {} for sanity */
#define Do_SetupPatchTypes(public_xtra, interval, i, cases)    \
  switch(InputType(public_xtra, i))                            \
  {                                                            \
    cases;                                                     \
  }

#define SetupPatchType(type, body)                               \
  case type:                                                     \
  {                                                              \
    body;                                                        \
    break;                                                       \
  }


#define Do_SetupPatchIntervals(public_xtra, interval, i, cases)  \
  switch(InputType(public_xtra, i))                            \
  {                                                            \
    cases;                                               \
  }

#define SetupPatchInterval(type, body)          \
  case type:                                    \
  {                                             \
    body;                                       \
    break;                                      \
  }

#define Do_FreePatches(public_xtra, i, ...)                   \
  switch(InputType(public_xtra, i))                           \
  {                                                           \
    __VA_ARGS__;                                              \
  }

#define FreePatch(type, body)                   \
  case type:                                    \
  {                                             \
    body;                                       \
    break;                                      \
  }

/*----------------------------------------------------------------
 * BCPressure Values
 * These contain information for a particular interval
 * @MCB: Is there a way we could generate these based on BC_TYPE_TABLE?
 *       Would remove the need to define what is basically the same
 *       struct twice.
 *----------------------------------------------------------------*/
#define BC_INTERVAL_TYPE_TABLE                      \
  BC_TYPE(DirEquilRefPatch, {                       \
      int reference_solid;                          \
      int reference_patch;                          \
      double value;                                 \
      double *value_at_interfaces;                  \
    })                                              \
  BC_TYPE(DirEquilPLinear, {                        \
      double xlower;                                \
      double ylower;                                \
      double xupper;                                \
      double yupper;                                \
      int    num_points;                            \
      double *points;                               \
      double *values;                               \
      double *value_at_interfaces;                  \
    })                                              \
  BC_TYPE(FluxConst, {                              \
      double value;                                 \
    })                                              \
  BC_TYPE(FluxVolumetric, {                         \
      double value;                                 \
    })                                              \
  BC_TYPE(PressureFile, {                           \
      char *filename;                               \
    })                                              \
  BC_TYPE(FluxFile, {                               \
      char *filename;                               \
    })                                              \
  BC_TYPE(ExactSolution, {                          \
      int function_type;                            \
    })                                              \
  BC_TYPE(OverlandFlow, {                           \
      double value;                                 \
    })                                              \
  BC_TYPE(OverlandFlowPFB, {                        \
      char *filename;                               \
    })


/* @MCB: Interval structs are used in multiple C files, generate the definitions in the header */
#define BC_TYPE(type, values) typedef struct values BCPressureType ## type;
BC_INTERVAL_TYPE_TABLE
#undef BC_TYPE

#define NewBCPressureTypeStruct(type, varname)  \
  BCPressureType ## type * varname = ctalloc(BCPressureType ## type, 1)
#define GetBCPressureTypeStruct(type, varname, bc_pressure_data, ipatch, interval_number) \
  BCPressureType ## type * varname \
  = (BCPressureType ## type *)BCPressureDataIntervalValue(bc_pressure_data, \
                                                          (ipatch), (interval_number))

/*----------------------------------------------------------------
 * BCPressure Data structure
 *----------------------------------------------------------------*/

typedef struct {
  int num_phases;

  int num_patches;

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
 * @MCB: With the new macro system these don't make as much sense, replace/deprecate?
 *--------------------------------------------------------------------------*/
#define BCPressureType0Value(type0) \
  ((type0)->value)

#define BCPressureType0RefSolid(type0) \
  ((type0)->reference_solid)
#define BCPressureType0RefPatch(type0) \
  ((type0)->reference_patch)

#define BCPressureType0ValueAtInterfaces(type0) \
  ((type0)->value_at_interfaces)
#define BCPressureType0ValueAtInterface(type0, i) \
  ((type0)->value_at_interfaces[i - 1])
/*--------------------------------------------------------------------------*/
#define BCPressureType1XLower(type1) \
  ((type1)->xlower)

#define BCPressureType1YLower(type1) \
  ((type1)->ylower)

#define BCPressureType1XUpper(type1) \
  ((type1)->xupper)

#define BCPressureType1YUpper(type1) \
  ((type1)->yupper)

#define BCPressureType1NumPoints(type1) \
  ((type1)->num_points)

#define BCPressureType1Points(type1) \
  ((type1)->points)
#define BCPressureType1Point(type1, i) \
  ((type1)->points[i])

#define BCPressureType1Values(type1) \
  ((type1)->values)
#define BCPressureType1Value(type1, i) \
  ((type1)->values[i])

#define BCPressureType1ValueAtInterfaces(type1) \
  ((type1)->value_at_interfaces)
#define BCPressureType1ValueAtInterface(type1, i) \
  ((type1)->value_at_interfaces[i - 1])
/*--------------------------------------------------------------------------*/
#define BCPressureType2Value(type2) \
  ((type2)->value)
/*--------------------------------------------------------------------------*/
#define BCPressureType3Value(type3) \
  ((type3)->value)

/*--------------------------------------------------------------------------*/
#define BCPressureType4FileName(type4) \
  ((type4)->filename)

/*--------------------------------------------------------------------------*/
#define BCPressureType5FileName(type5) \
  ((type5)->filename)

/*--------------------------------------------------------------------------*/
#define BCPressureType6FunctionType(type6) \
  ((type6)->function_type)

/*--------------------------------------------------------------------------*/
#define BCPressureType7Value(type7) \
  ((type7)->value)
/*--------------------------------------------------------------------------*/
#define BCPressureType8FileName(type8) \
  ((type8)->filename)

/*--------------------------------------------------------------------------
 * Accessor macros: BCPressureData
 *--------------------------------------------------------------------------*/
#define BCPressureDataNumPhases(bc_pressure_data) \
  ((bc_pressure_data)->num_phases)

#define BCPressureDataNumPatches(bc_pressure_data) \
  ((bc_pressure_data)->num_patches)

#define BCPressureDataTypes(bc_pressure_data) \
  ((bc_pressure_data)->types)
#define BCPressureDataType(bc_pressure_data, i) \
  ((bc_pressure_data)->types[i])

#define BCPressureDataCycleNumbers(bc_pressure_data) \
  ((bc_pressure_data)->cycle_numbers)
#define BCPressureDataCycleNumber(bc_pressure_data, i) \
  ((bc_pressure_data)->cycle_numbers[i])

#define BCPressureDataPatchIndexes(bc_pressure_data) \
  ((bc_pressure_data)->patch_indexes)
#define BCPressureDataPatchIndex(bc_pressure_data, i) \
  ((bc_pressure_data)->patch_indexes[i])

#define BCPressureDataBCTypes(bc_pressure_data) \
  ((bc_pressure_data)->bc_types)
#define BCPressureDataBCType(bc_pressure_data, i) \
  ((bc_pressure_data)->bc_types[i])

#define BCPressureDataValues(bc_pressure_data) \
  ((bc_pressure_data)->values)
#define BCPressureDataIntervalValues(bc_pressure_data, i) \
  ((bc_pressure_data)->values[i])
#define BCPressureDataIntervalValue(bc_pressure_data, i, interval_number) \
  (((bc_pressure_data)->values[i])[interval_number])

#define BCPressureDataTimeCycleData(bc_pressure_data) \
  ((bc_pressure_data)->time_cycle_data)

/*--------------------------------------------------------------------------
 * BCPressure Data constants used in the program.
 *--------------------------------------------------------------------------*/

#endif
