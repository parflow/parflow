#ifndef __PYPFCOMMON_H__
#define __PYPFCOMMON_H__

#include "messages.h"

// Python Wrapper with NumpyDataTypes:


// TODO: for performance use maybe argout view arrays, see https://docs.scipy.org/doc/numpy-1.13.0/reference/swig.interface-file.html
extern void SendSteerMessage(const Action action, const Variable variable,
                             int ix, int iy, int iz,
                             double *IN_ARRAY3, int DIM1, int DIM2, int DIM3);

extern void _run(char *logstamps[], size_t logstampsc);  /// needs to be called by the python code to run as a module!

extern void SendLog(StampLog slog[], size_t n);

#endif
