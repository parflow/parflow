/*BHEADER*********************************************************************
 *  This file is part of Parflow. For details, see
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

#ifndef _MESSAGES_HEADER
#define _MESSAGES_HEADER

#include <fca.h>

/**
 * Struct used to transmit logs to logger modules
 */
typedef struct {
  const char *stamp_name;
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

/**
 * Variables that can be used for snapshots, to be defined as outports in contracts or
 * to be Steered
 */
typedef enum {
  // REM: to find the position of the variables in the memory the ProblemData struct is
  // helpful.
  VARIABLE_PRESSURE = 0,
  VARIABLE_SATURATION,
  VARIABLE_POROSITY,

  VARIABLE_MANNING,
  VARIABLE_PERMEABILITY_X,
  VARIABLE_PERMEABILITY_Y,
  VARIABLE_PERMEABILITY_Z,


  VARIABLE_LAST
} Variable;

/**
 * Actions that can be requested by action messages
 */
typedef enum {
  ACTION_GET_GRID_DEFINITION,   // send message with grid definition on snapshot port
  ACTION_TRIGGER_SNAPSHOT,      // request snapshot on snapshot port
  ACTION_SET,                   // set a variable to the attached grid message's content
  ACTION_ADD,                   // add attached grid message's content to a variable
  ACTION_MULTIPLY               // multiply a variable by attached grid message content
} Action;

/**
 * Conversion between variable names like "pressure" and the according constants like
 * VARIABLE_PRESSURE
 */
extern const char *VARIABLE_TO_NAME[VARIABLE_LAST];
Variable NameToVariable(const char *name);

/**
 * Data structures defining the message (header) format of FlowVR messages
 */
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

typedef struct {
  int nX;
  int nY;
  int nZ;
} GridDefinition;

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
  char run_name[128];
} GridMessageMetadata;

/**
 * Sends an action message
 *
 * @param mod FlowVR module to send the action message from
 * @param out the outport
 * @param action action to perform
 * @param variable variable on which to perform @param action
 */
extern void SendActionMessage(fca_module mod, fca_port port, Action action, Variable variable,
                              void *parameter, size_t parameter_size);

/**
 * Macro to simplify the declaration of callback functions for ParseMergedMessage
 *
 * @param buffer specifies where to read out data.
 * @param size specifies the overall of the merged message buffer is a part of
 * @param cbdata user data
 */
#define MergeMessageParser(function_name) \
  size_t function_name(const void *buffer, size_t size, void *cbdata)

/**
 * Gets the last message on inport @param port. Assuming it is a message produced by the FlowVR
 * merge Filter or by its derivatives. Calls @param cb with the current read out position as
 * @param buffer parameter to the callback function until the complete message is processed.
 * Thus buffer must always return how many bytes it read out.
 * @param cbdata can be used to transfer user data to the callback
 *
 * @param port message where the merged message arrived
 * @param cb call back to call with extracted message as parameter
 * @param buffer extracted message
 * @param size size of the overall merged message
 * @param cbdata user data
 */
extern void ParseMergedMessage(fca_port port,
                               size_t (*cb)(const void *buffer, size_t size, void *cbdata),
                               void *cbdata);

/**
 * Sends a steer message
 *
 * @param mod the FlowVR module to send the steer message from
 * @param out the outport
 * @param action the steer action to perform
 * @param variable the variable to steer
 * @param ix, iy, iz start indices of where to steer
 * @param data steer operand
 * @param nx, ny, nz the dimensiions of operand
 *
 *
 * Example:
 *
 * double operand[4] = {42, 42, 42, 42};
 * SendSteerMessage(mod, out, ACTION_SET, VARIABLE_PRESSURE, 0, 0, 0, operand, 2, 2, 1);
 *
 *  ==> ParFlow pressure = [[[42, 42, ...],
 *                           [42, 42, ...],
 *                           ...], ...]
 */
extern void SendSteerMessage(fca_module mod, fca_port out, const Action action,
                             const Variable variable,
                             int ix, int iy, int iz,
                             double *data, int nx, int ny, int nz);

/**
 * Sends a log message logging
 *
 * @param mod FlowVR module that will send the log
 * @param port outport on which the log shall be sent
 * @param n variables in  @param log
 * @param log[] array of variables to log
 */
extern void SendLogMessage(fca_module mod, fca_port port, StampLog log[], size_t n);

/**
 * Sends an empty message. Useful when analyzer need to send a message each timestep to
 * not deadlock the workflow.
 *
 * @param mod FlowVR module that will send the empty message
 * @param port outport on which the empty message shall be sent
 */
extern void SendEmptyMessage(fca_module mod, fca_port port);

// Some Fancy Reader code generation:
#define GenerateMessageReaderH(type) \
  typedef struct { \
    type ## MessageMetadata * m; \
    double *data; \
  } type ## Message; \
\
\
  extern inline type ## Message Read ## type ## Message(void const * const buffer)

#define GenerateMessageReaderC(type) \
  type ## Message Read ## type ## Message(void const * const buffer) \
  { \
    type ## Message res; \
    res.m = (type ## MessageMetadata*)buffer; \
    res.data = (double*)(res.m + 1); \
    return res; \
  }


/**
 * Call ReadGridMesssage(buffer), ReadSteerMessage(buffer) or ReadActionMessage(buffer)
 * on a buffer containing a message of the according type. This will create a struct
 * giving access to the message's meta data via the m member and to it's data via the
 * data member.
 *
 * Example:
 * {Grid|Steer|Action}Message res = Read{Grid|Steer|Action}Message(buffer);
 * res.m;     // pointer to message meta data
 * res.data;  // pointer to message data
 */

GenerateMessageReaderH(Grid);
GenerateMessageReaderH(Steer);
GenerateMessageReaderH(Action);

#endif
