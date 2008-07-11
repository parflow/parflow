/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "amps.h"

int amps_xsend(comm, dest, buffer, size)
amps_Comm comm;
int dest;
char *buffer;
int size;
{

   MPI_Send(buffer, size, MPI_BYTE, dest, 0, MPI_COMM_WORLD);

   return 0;
}


/*===========================================================================*/
/**

Sends a message to the node with {\bf rank} in the {\bf comm} context.
The contents of the message are described by the {\bf invoice} as
indicated in \Ref{amps_NewInvoice}.  After a return
from send it is invalid to access any overlaid variables.  This
version of send is possibly blocking.  

{\large Example:}
\begin{verbatim}
amps_Invoice invoice;
int me, i;
double d;

me = amps_Rank(amps_CommWorld);

invoice = amps_NewInvoice("%i%d", &i, &d);

amps_Send(amps_CommWorld, me+1, invoice);

amps_Recv(amps_CommWorld, me-1, invoice);

amps_FreeInvoice(invoice);
\end{verbatim}

{\large Notes:}

Since {\em AMPS} is layered on other message passing systems there is
some loosely defined behavior.  \Ref{amps_Send} is to be implemented
in a way that starts and completes a send operation on the source
side.  Since some systems have synchronization or do queuing on the
sending side this function can block until a receive is posted on the
node specified by {\bf rank}.

@memo Blocking send
@param comm Communication context [IN]
@param dest Rank of destination node [IN]
@param invoice Data to communicate [IN]
@return Error code
*/
int amps_Send(amps_Comm comm, int dest, amps_Invoice invoice)
{

   char *buffer;
   int   size;

   size = amps_pack(comm, invoice, &buffer);

/*
   amps_Printf("amps_Send: %d bytes to node %d from buffer %x\n", size, dest, buffer);
*/

   MPI_Send(buffer, size, MPI_BYTE, dest, 0, MPI_COMM_WORLD); 

   amps_ClearInvoice(invoice);

   return 0;

}
