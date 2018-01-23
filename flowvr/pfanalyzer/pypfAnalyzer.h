#ifndef __PYPFANALYZER_H__
#define __PYPFANALYZER_H__

#include "messages.h"

// Python Wrapper with NumpyDataTypes:


// see https://docs.scipy.org/doc/numpy-1.13.0/reference/swig.interface-file.html
extern void SendSteer(const Action action, const Variable variable,
                      int ix, int iy, int iz,
                      double *IN_ARRAY3, int DIM1, int DIM2, int DIM3);

extern void _run(char *logstamps[], size_t logstampsc);  /// needs to be called by the python code to run as a module!

extern void SendLog(StampLog slog[], size_t n);

#endif
