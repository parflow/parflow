/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "amps.h"

/**

\Ref{amps_Wait} is used to block until the communication initiated by
them has completed.  {\bf handle} is the communications handle that
was returned by the \Ref{amps_ISend}, \Ref{amps_IRecv}, or
\Ref{amps_IExchangePackage} commands.  You must always do an
\Ref{amps_Wait} on to finalize an initiated non-blocking
communication.

{\large Example:}
\begin{verbatim}
amps_Invoice invoice;
amps_Handle  handle;
int me, i;
double d;

me = amps_Rank(amps_CommWorld);

invoice = amps_NewInvoice("%i%d", &i, &d);

handle = amps_ISend(amps_CommWorld, me+1, invoice);

// do some work 

amps_Wait(handle);

handle = amps_IRecv(amps_CommWorld, me+1, invoice);

while(amps_Test(handle))
{
        // do more work 
}
amps_Wait(handle);

amps_FreeInvoice(invoice);
\end{verbatim}

{\large Notes:}

The requirement for to always finish an initiated communication with
\Ref{amps_Wait} is under consideration.

@memo Wait for initialized non-blocking communication to finish
@param handle communication handle
@return Error code
*/
#ifndef AMPS_WAIT_SPECIALIZED
int amps_Wait(amps_Handle handle)
{
   if(handle)
   {
      if(handle -> type)
	 amps_Recv(handle -> comm, handle -> id, handle -> invoice);
      else
	 _amps_wait_exchange(handle);

      amps_FreeHandle(handle);
      handle = NULL;
   }

   return 0;
}
#endif
