#ifndef _BC_PRESSURE_PACKAGE_H
#define _BC_PRESSURE_PACKAGE_H

/***********************************
 *
 * Define BC Types for use in NewPublicXtra
 *
 ***********************************/

#define DirEquilRefPatch 0
#define DirEquilPLinear  1
#define FluxConst        2
#define FluxVolumetric   3
#define PressureFile     4
#define FluxFile         5
#define ExactSolution    6
#define OverlandFlow     7
#define OverlandFlowPFB  8

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


#define TOSTR_(x) #x
#define TOSTR(x) TOSTR_(x)


#define BC_TYPE_NAMES                           \
  DirEquilRefPatch                              \
  DirEquilPLinear                               \
  FluxConst                                     \
  FluxVolumetric                                \
  PressureFile                                  \
  FluxFile                                      \
  ExactSolution                                 \
  OverlandFlow                                  \
  OverlandFlowPFB

#define NewTypeStruct(type, var)                  \
  Type ## type * var = ctalloc((Type ## type), 1)
#define StoreTypeStruct(public_xtra, var, i)                  \
  (public_xtra)->data[(i)] = (void*)(var);
#define GetTypeStruct(type, var, public_xtra, i)              \
  Type ## type * var = (Type ## type *)(public_xtra->data[i])
#define InputType(public_xtra, i)               \
  ((public_xtra)->input_types[(i)])

// MCB: These two macros aren't really necessary but they do make the code cleaner
#define ForEachPatch(num_patches, i)           \
  for (i = 0; i < num_patches; i++)
#define ForEachInterval(interval_division, interval_number)  \
  for (internval_number = 0; interval_number < interval_division; interval_number++)


#define Do_SetupPatchTypes(public_xtra, interval, i, ...)      \
  switch(InputType(public_xtra, i))                            \
  {                                                            \
    __VA_ARGS__;                                               \
  }

#define SetupPatchType(type, body)                             \
  case type:                                                   \
  {                                                            \
    body;                                                      \
    break;                                                     \
  }

#define Do_SetupPatchIntervals(public_xtra, interval, i, ...)  \
  switch(InputType(public_xtra, i))                            \
  {                                                            \
    __VA_ARGS__;                                               \
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


#endif // _BC_PRESSURE_PACKAGE_H
