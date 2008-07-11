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

\Ref{amps_Test} is used to test if a non-blocking communication is
completed.  A non-zero return value indicates success, zero indicates
that the operation has not completed.  {\bf handle} is the
communications handle that was returned by the \Ref{amps_ISend} or
\Ref{amps_IRecv} commands.  Do not use \Ref{amps_Test} in a busy
loop, \Ref{amps_Wait} is used for this purpose.

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

@memo Test if communication has completed
@param handle Communication request handle [IN]
@return True if completed, false if not
*/
int amps_Test(amps_Handle handle)
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

   return 1;
}
