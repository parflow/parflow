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

/*===========================================================================*/
/**

This routine is used to print information to the standard output device.
Where the output actually ends up is dependent on the underlying message
passing system.  On some systems it will appear on the console where the
program was run; on others it will be placed in a file.  The arguments
are the same as for the standard C {\bf printf} function.

{\large Example:}
\begin{verbatim}
int main( int argc, char *argv)
{
   amps_Init(argc, argv);
   
   amps_Printf("Hello World");

   amps_Finalize();
}
\end{verbatim}

{\large Notes:}

Where the output ends up and the order of the output 
is dependent on the underlying message passing system.

@memo Print to stdout
@param fmt Output format string [IN]
@param ... Output optional variables
@return void
*/

void amps_Printf(char *fmt, ...)
{
   va_list argp;
   fprintf(stdout, "Node %d: ", amps_rank);
   va_start(argp, fmt);

#ifdef SGS_LAM
   /* In order to capture output in tcl can't have output kept
      open so direct to stderr and send stdout to /dev/null
      LAM is doing something with stdout */
   vfprintf(stderr, fmt, argp);
#else
   vfprintf(stdout, fmt, argp);
#endif

   va_end(argp);
}

