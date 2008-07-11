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

#if 0
  printf("%d AMPS sending to %d\n", amps_rank, dest);
#endif

  /* Set up the new buffer */
  if ( (buf = (AMPS_ShMemBuffer *)malloc( sizeof(AMPS_ShMemBuffer)))
      == NULL)
    {
      printf("AMPS Error: AMPS_ShMemBuffer allocation failed\n");
      exit(1);
    }

  buf -> data = (char *)src_ptr;
  buf -> next = NULL;  
    
  /* place the buffer on the message queue */

  WaitForSingleObject(locks[dest], INFINITE);

  buf -> prev = buffers_end[dest];

  if( buffers_end[dest] )
    buffers_end[dest] -> next = buf;

  buffers_end[dest] = buf;

  if ( !buffers[dest] )
    buffers[dest] = buf;


  /* increment semaphore so dest knows there is another message
     on the queue */

#if 0
  printf("%d AMPS semaphoring %d\n", amps_rank, dest);
#endif
  ReleaseSemaphore(sema[dest], 1, NULL);

  ReleaseMutex(locks[dest]);
    

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
