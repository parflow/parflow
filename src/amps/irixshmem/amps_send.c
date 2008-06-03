/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

int amps_xsend(buffer, dest)
char *buffer;
int dest;
{
    int *src_ptr;
    AMPS_ShMemBuffer *buf;
    
    src_ptr = (int *)(buffer - sizeof(double));
    *src_ptr = amps_rank;

    /* Set up the new buffer */
    if ( (buf = (AMPS_ShMemBuffer *)usmalloc( sizeof(AMPS_ShMemBuffer), amps_arena))
	== NULL)
    {
       printf("AMPS Error: AMPS_ShMemBuffer allocation failed\n");
       exit(1);
    }

    buf -> data = (char *)src_ptr;
    buf -> next = NULL;  
    
    /* place the buffer on the message queue */

    ussetlock(amps_shmem_info -> locks[dest]);

    buf -> prev = amps_shmem_info -> buffers_end[dest];

    if( amps_shmem_info -> buffers_end[dest] )
       amps_shmem_info -> buffers_end[dest] -> next = buf;

    amps_shmem_info -> buffers_end[dest] = buf;

    if ( !amps_shmem_info -> buffers[dest] )
       amps_shmem_info -> buffers[dest] = buf;

    usunsetlock(amps_shmem_info -> locks[dest]);
    
    /* increment semaphore so dest knows there is another message
       on the queue */
    usvsema(amps_shmem_info -> sema[dest]);

    return 0;
}

int amps_Send(comm, dest, invoice)
amps_Comm comm;
int dest;
amps_Invoice invoice;
{
   char *buffer;

   amps_pack(comm, invoice, &buffer);

   amps_xsend(buffer, dest);

   AMPS_CLEAR_INVOICE(invoice);
   return 0;
}
