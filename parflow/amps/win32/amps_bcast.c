/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

int 
amps_BCast (comm, source, invoice)
     amps_Comm comm;
     int source;
     amps_Invoice invoice;
{
  
  int not_found = 1;
  int *src_ptr;
  char *ptr;
  int dest;
  int size;
  
  AMPS_ShMemBuffer *buf;
  
  /* if running single node we don't need to do anything */
  if (amps_size > 1)
    if (source == amps_rank)
      {
	
	size = amps_pack (comm, invoice, &ptr) + sizeof (double);
	
	src_ptr = (int *) (ptr - sizeof (double));
	*src_ptr = amps_rank;
	
	/* Only allow one bcast at a time */
	WaitForSingleObject (bcast_lock, INFINITE);
	
	/* place the buffer on the bcast queues for all procs */
	for (dest = 0; dest < amps_size; dest++)
	  if (dest != source)
	    {
	      /* Set up the new buffer */
	      if ((buf = (AMPS_ShMemBuffer *) malloc (sizeof (AMPS_ShMemBuffer)))
		  == NULL)
		{
		  printf ("AMPS Error: AMPS_ShMemBuffer allocation failed\n");
		  exit (1);
		}

	      if ((buf->data = (char *) malloc (size)) == NULL)
		{
		  printf ("AMPS Error: AMPS_ShMemBuffer allocation failed\n");
		  exit (1);
		}
	      memcpy (buf->data, src_ptr, size);

	      buf->next = NULL;
	      buf->count = 0;

	      WaitForSingleObject (bcast_locks[dest], INFINITE);

	      buf->prev = bcast_buffers_end[dest];

	      if (bcast_buffers_end[dest])
		bcast_buffers_end[dest]->next = buf;

	      bcast_buffers_end[dest] = buf;

	      if (!bcast_buffers[dest])
		bcast_buffers[dest] = buf;

	      /* increment semaphore so dest knows there is another message
		  on the queue */
	      ReleaseSemaphore (bcast_sema[dest], 1, NULL);

	      ReleaseMutex (bcast_locks[dest]);

	    }

	ReleaseMutex (bcast_lock);


	(invoice)->combuf_flags &= ~AMPS_INVOICE_ALLOCATED;
	free (src_ptr);
      }
    else
      {
	/* not the source so receive */

	amps_ClearInvoice (invoice);

	/* first look on the local queue */
	buf = bcast_buffers_local[amps_rank];
	while (buf)
	  {
	    if (*(int *) (buf->data) == source)
	      {
		/* remove the found buffer from the list */
		if (buf->prev)
		  buf->prev->next = buf->next;
		else
		  /* start of list */
		  bcast_buffers_local[amps_rank]
		    = buf->next;

		if (buf->next)
		  buf->next->prev = buf->prev;
		else
		  bcast_buffers_local_end[amps_rank] =
		    buf->prev;

		/* return the buffer that was found */
		not_found = 0;
		break;
	      }
	    buf = buf->next;
	  }

	while (not_found)
	  {
	    /* wait until someone puts another buffer on queue */
	    WaitForSingleObject (bcast_sema[amps_rank], INFINITE);

	    WaitForSingleObject (bcast_locks[amps_rank], INFINITE);

	    buf = bcast_buffers[amps_rank];

	    /* start of list */
	    bcast_buffers[amps_rank] = buf->next;

	    if (buf->next)
	      buf->next->prev = NULL;
	    else
	      bcast_buffers_end[amps_rank] = NULL;

	    ReleaseMutex (bcast_locks[amps_rank]);

	    /* if this is the one we are looking for then we are done */
	    if (*(int *) (buf->data) == source)
	      not_found = 0;
	    else
	      {
		/* not the buffer we are looking for add to the local buffer
		  list */
		buf->next = NULL;

		buf->prev = bcast_buffers_local_end[amps_rank];

		if (bcast_buffers_local_end[amps_rank])
		  bcast_buffers_end[amps_rank]->next = buf;

		bcast_buffers_local_end[amps_rank] = buf;

		if (!bcast_buffers_local[amps_rank])
		  bcast_buffers_local[amps_rank] = buf;

	      }
	  }

	amps_unpack (comm, invoice, (buf->data) + sizeof (double));

	if ((invoice)->combuf_flags & AMPS_INVOICE_OVERLAYED)
	  (invoice)->combuf_flags |= AMPS_INVOICE_ALLOCATED;
	else
	  {
	    (invoice)->combuf_flags &= ~AMPS_INVOICE_ALLOCATED;
#if 0
	    printf ("%d: freeing data\n", amps_rank);
#endif
	    free (buf->data);
	  }

#if 0
	printf ("%d: freeing buffer\n", amps_rank);
#endif
	free (buf);

      }

  return 0;

}

