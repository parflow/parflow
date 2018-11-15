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
#ifndef __PYPFANALYZER_H__
#define __PYPFANALYZER_H__

#include "messages.h"

// Python Wrapper with NumpyDataTypes:
// see https://docs.scipy.org/doc/numpy-1.13.0/reference/swig.interface-file.html

/**
 * Send a steer message.
 *
 * @param action the steer action to perform
 * @param variable the variable on which to perform the steer action
 * @param ix, iy, iz start indices of where to start applying the steer
 * @param IN_ARRAY3 the steer operand
 *
 * Python Example:
 * import pypfAnalyzer as pfa
 *
 * pfa.SendSteer(pfa.ACTION_SET, pfa.VARIABLE_PRESSURE, 0, 0, 0, [[[42, 42],
 *                                                                 [42, 42]]])
 *
 *  ==> ParFlow pressure = [[[42, 42, ...],
 *                           [42, 42, ...],
 *                           ...], ...]
 */
extern void SendSteer(const Action action, const Variable variable,
                      int ix, int iy, int iz,
                      double *IN_ARRAY3, int DIM1, int DIM2, int DIM3);

// needs to be called by the python code to run as a module!
extern void _run(char *logstamps[], size_t logstampsc);

/**
 * Send a log message logging the elements in @param slog
 *
 * @param slog[] the elements to log.
 */
extern void SendLog(StampLog slog[], size_t n);

/**
 * Send an empty steer action
 */
extern void SendEmpty();

/**
 * Aborts the FlowVR application
 */
extern void Abort();

#endif
