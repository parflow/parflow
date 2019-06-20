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
  OverlandFlowPFB                               \
  TestBC

#define NewTypeStruct(type, var)                  \
  Type ## type * var = ctalloc((Type ## type), 1)
#define GetTypeStruct(type, var, public_xtra, i)              \
  Type ## type * var = (Type ## type *)(public_xtra->data[i])
#define InputType(public_xtra, i)               \
  ((public_xtra)->input_types[(i)])

#endif // _BC_PRESSURE_PACKAGE_H
