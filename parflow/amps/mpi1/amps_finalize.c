/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

/*===========================================================================*/
/**

Every {\em AMPS} program must call this function to exit from the
message passing environment.  This must be the last call to an
{\em AMPS} routine.  \Ref{amps_Finalize} might synchronize the 
of the node programs;  this might be necessary to correctly free up
memory resources and communication structures.

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


There is currently no way to forcibly kill another node.  Exiting do
to an error condition is problematic.

@memo Exit AMPS environment
@return Error code
*/

int amps_Finalize()
{

   MPI_Finalize();

#ifdef AMPS_MALLOC_DEBUG
  /* check out the heap and shut everything down if we are in debug mode */
#if 0
   dmalloc_verify(NULL);
   dmalloc_shutdown();
#endif
#endif
   return 0;
}
