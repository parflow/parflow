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
  int nx;
  int ny;
  int nz;
  int ix;
  int iy;
  int iz;
  int nX;
  int nY;
  int nZ;
  double time; //TODO: set it
} GridMessageMetadata;


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
  VARIABLE_KS,
  VARIABLE_POROSITY,
  VARIABLE_MANNING,
  VARIABLE_PERMEABILITY_X,
  VARIABLE_PERMEABILITY_Y,
  VARIABLE_PERMEABILITY_Z
} Variable;

typedef enum {
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


void SendActionMessage(fca_module mod, fca_port port, Action action, Variable variable,
                       void *parameter, size_t parameterSize);
void ParseMergedMessage(fca_port port,
                        size_t (*cb)(const void *buffer, size_t size, void *cbdata),
                        void *cbdata);
#endif
