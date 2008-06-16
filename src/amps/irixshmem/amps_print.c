/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include <stdarg.h>

#include "amps.h"


void amps_Printf(char *fmt, ...)
{
   va_list argp;
   fprintf(stdout, "Node %d: ", amps_rank);
   va_start(argp, fmt);

   vfprintf(stdout, fmt, argp);

   fflush(NULL);

   va_end(argp);
}

