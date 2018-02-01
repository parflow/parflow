#ifndef __PYPFANALYZER_H__
#define __PYPFANALYZER_H__

#include "messages.h"

// Python Wrapper with NumpyDataTypes:
// see https://docs.scipy.org/doc/numpy-1.13.0/reference/swig.interface-file.html

/**
 * Send a steer message.
 *
 * \p ix, iy, iz start indices of where to steer
 * \p IN_ARRAY3 the steer operand
 */
extern void SendSteer(const Action action, const Variable variable,
                      int ix, int iy, int iz,
                      double *IN_ARRAY3, int DIM1, int DIM2, int DIM3);

// needs to be called by the python code to run as a module!
extern void _run(char *logstamps[], size_t logstampsc);

/**
 * Send a log message logging the elements in \p slog
 */
extern void SendLog(StampLog slog[], size_t n);

/**
 * Send an empty steer action
 */
extern void SendEmpty();

#endif
