/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "amps.h"

char *amps_recvb(src, size)
int src;
int *size;
{
    char *buf;

    MPI_Status status;

    MPI_Probe(src, 0, MPI_COMM_WORLD, &status);

    MPI_Get_count(&status, MPI_BYTE, size);

    buf = malloc(*size);

    MPI_Recv(buf, *size, MPI_BYTE, src, 0, MPI_COMM_WORLD, &status);

    return buf;
}

/*===========================================================================*/
/**

\Ref{amps_Recv} is a blocking receive operation.  It receives a message
from the node with {\bf rank} within the {\bf comm} context.  This
operation will not return until the receive operation has been
completed.  The received data is unpacked into the the data locations
specified in the {\bf invoice}.  After the return it is legal to access
overlayed variables (which must be freed with \Ref{amps_Clear}).

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

@memo Blocking receive
@param comm Communication context [IN]
@param source Node rank to receive from [IN]
@param invoice Data to receive [IN/OUT]
@return Error code
*/

int amps_Recv(amps_Comm comm, int source, amps_Invoice invoice)
{
   char *buffer;
   int size;
   MPI_Status status;

   AMPS_CLEAR_INVOICE(invoice);
   
   MPI_Probe(source, 0, MPI_COMM_WORLD, &status);

   MPI_Get_count(&status, MPI_BYTE, &size);

   buffer = malloc(size);

   MPI_Recv(buffer, size, MPI_BYTE, source, 0, MPI_COMM_WORLD, &status);

   amps_unpack(comm, invoice, buffer, size);

   AMPS_PACK_FREE_LETTER(comm, invoice, buffer);

   return 0;
}
