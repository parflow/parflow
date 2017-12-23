/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#ifndef _MESSAGES_HEADER
#define _MESSAGES_HEADER

#include <fca/fca.h>

typedef struct {
  int nX;
  int nY;
  int nZ;
} GridDefinition;

typedef struct {
  const char *stampName;
  float value;
} StampLog;


// someof the most interesting variables
/*PFModule  *geometries;*/
/*PFModule  *domain;*/
/*PFModule  *permeability;*/
/*PFModule  *porosity;*/
/*PFModule  *wells;*/
/*PFModule  *bc_pressure;*/
/*PFModule  *specific_storage;  //sk*/
/*PFModule  *x_slope;  //sk*/
/*PFModule  *y_slope;  //sk*/
/*PFModule  *mann;  //sk*/
/*PFModule  *dz_mult;   //RMM*/
/*"slopes",               [> slopes <]*/
/*"mannings",             [> mannings <]*/
/*"top",                  [> top <]*/
/*"wells",                [> well data <]*/
/*"evaptrans",            [> evaptrans <]*/
/*"evaptrans_sum",        [> evaptrans_sum <]*/
/*"overland_sum",         [> overland_sum <]*/

// TODO: rename this file into something more fitting!
typedef enum {
  /// just for trigger:
  VARIABLE_PRESSURE = 0,
  VARIABLE_SATURATION,

  // to steer:
  // et plus, see ProblemData struct.
  VARIABLE_POROSITY,
  VARIABLE_MANNING,
  VARIABLE_PERMEABILITY_X,
  VARIABLE_PERMEABILITY_Y,
  VARIABLE_PERMEABILITY_Z,

  VARIABLE_LAST
} Variable;

extern const char *VARIABLE_TO_NAME[VARIABLE_LAST];
Variable NameToVariable(const char *name);

typedef struct {
  double time;
  Variable variable;
  GridDefinition grid;
  int nx;
  int ny;
  int nz;
  int ix;
  int iy;
  int iz;
} GridMessageMetadata;


typedef enum {
  ACTION_GET_GRID_DEFINITION,
  ACTION_TRIGGER_SNAPSHOT,
  ACTION_SET,
  ACTION_ADD,
  ACTION_MULTIPLY
} Action;

typedef struct {
  Variable variable;
  Action action;
  // REM: parameter size is parsed by type ;)
} ActionMessageMetadata;


typedef struct {
  int nx;
  int ny;
  int nz;
  int ix;
  int iy;
  int iz;
} SteerMessageMetadata;


extern void SendActionMessage(fca_module mod, fca_port port, Action action, Variable variable,
                              void *parameter, size_t parameterSize);

#define MergeMessageParser(function_name) \
  size_t function_name(const void *buffer, size_t size, void *cbdata)
extern void ParseMergedMessage(fca_port port,
                               size_t (*cb)(const void *buffer, size_t size, void *cbdata),
                               void *cbdata);


extern void SendLogMessage(fca_module mod, fca_port port, StampLog log[], size_t n);


// Some Fancy Reader code generation:
#define GenerateMessageReaderH(type) \
  typedef struct { \
    type ## MessageMetadata * m; \
    double *data; \
  } type ## Message; \
\
\
  extern inline type ## Message Read ## type ## Message(void *buffer)

#define GenerateMessageReaderC(type) \
  type ## Message Read ## type ## Message(void *buffer) \
  { \
    type ## Message res; \
    res.m = (type ## MessageMetadata*)buffer; \
    res.data = (double*)(res.m + 1); \
    return res; \
  }

GenerateMessageReaderH(Grid);
GenerateMessageReaderH(Steer);
GenerateMessageReaderH(Action);

#endif
