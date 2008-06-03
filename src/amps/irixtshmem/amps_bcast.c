/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

int amps_BCast(comm, source, invoice) 
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
   AMPS_ShMemBuffer *last_buf;

   /* if running single node we don't need to do anything */
   if(amps_size > 1)
      if( source == amps_rank )
      {
	 
	 size = amps_pack(comm, invoice, &ptr) +sizeof(double);
	 
	 src_ptr = (int *)(ptr - sizeof(double));
	 *src_ptr = amps_rank;
	 
	 /* Only allow one bcast at a time */
	 ussetlock(amps_shmem_info -> bcast_lock);
	 
	 /* place the buffer on the bcast queues for all procs */
	 for(dest = 0; dest < amps_size; dest++)
	    if( dest != source)
	    {
	       /* Set up the new buffer */
	       if ( (buf = (AMPS_ShMemBuffer *)usmalloc( sizeof(AMPS_ShMemBuffer), 
							amps_arena)) == NULL)
	       {
		  printf("AMPS Error: AMPS_ShMemBuffer allocation failed\n");
		  exit(1);
	       }
	       
	       if ( (buf -> data = (char *)usmalloc( size, amps_arena)) == NULL)
	       {
		  printf("AMPS Error: AMPS_ShMemBuffer allocation failed\n");
		  exit(1);
	       }
	       memcpy(buf->data, src_ptr, size);
	       
	       buf -> next = NULL;    
	       buf -> count = 0;
	       
	       ussetlock(amps_shmem_info -> bcast_locks[dest]);
	       
	       buf -> prev = amps_shmem_info -> bcast_buffers_end[dest];
	       
	       if( amps_shmem_info -> bcast_buffers_end[dest] )
		  amps_shmem_info -> bcast_buffers_end[dest] -> next = buf;
	       
	       amps_shmem_info -> bcast_buffers_end[dest] = buf;
	       
	       if ( !amps_shmem_info -> bcast_buffers[dest] )
		  amps_shmem_info -> bcast_buffers[dest] = buf;

	       /* increment semaphore so dest knows there is another message
		  on the queue */
	       usvsema(amps_shmem_info -> bcast_sema[dest]);
	       
	       usunsetlock(amps_shmem_info -> bcast_locks[dest]);

	    }
	 
	 usunsetlock(amps_shmem_info -> bcast_lock);
	 
	 
	 (invoice) -> combuf_flags &= ~AMPS_INVOICE_ALLOCATED; 
	 usfree(src_ptr, amps_arena);
      }
      else
      {
	 /* not the source so receive */
	 
	 amps_ClearInvoice(invoice);
	 
	 /* first look on the local queue */
	 buf = amps_shmem_info -> bcast_buffers_local[amps_rank];
	 while(buf)
	 {
	    if( *(int *)(buf -> data) == source)
	    {
	       /* remove the found buffer from the list */
	       if(buf -> prev)
		  buf -> prev -> next = buf -> next;
	       else
		  /* start of list */
		  amps_shmem_info -> bcast_buffers_local[amps_rank]
		     = buf -> next;
	       
	       if (buf -> next )
		  buf -> next -> prev = buf -> prev;
	       else
		  amps_shmem_info -> bcast_buffers_local_end[amps_rank] = 
		     buf -> prev;
	       
	       /* return the buffer that was found */
	       not_found = 0;
	       break;
	    }
	    buf = buf -> next;
	 }
	 
	 while(not_found)
	 {
	    /* wait until someone puts another buffer on queue */
	    uspsema(amps_shmem_info -> bcast_sema[amps_rank]);

	    ussetlock(amps_shmem_info -> bcast_locks[amps_rank]);
	    
	    buf = amps_shmem_info -> bcast_buffers[amps_rank];
	    
	    /* start of list */
	    amps_shmem_info -> bcast_buffers[amps_rank] = buf -> next;
	    
	    if (buf -> next )
	       buf -> next -> prev = NULL;
	    else
	       amps_shmem_info -> bcast_buffers_end[amps_rank] = NULL;
	    
	    usunsetlock(amps_shmem_info -> bcast_locks[amps_rank]);
	    
	    /* if this is the one we are looking for then we are done */
	    if( *(int *)(buf -> data) == source)
	       not_found = 0;
	    else
	    {
	       /* not the buffer we are looking for add to the local buffer
		  list */
	       buf -> next = NULL;

	       buf -> prev = amps_shmem_info -> bcast_buffers_local_end[amps_rank];

	       if( amps_shmem_info -> bcast_buffers_local_end[amps_rank] )
		  amps_shmem_info -> bcast_buffers_end[amps_rank] -> next 
		     = buf;

	       amps_shmem_info -> bcast_buffers_local_end[amps_rank] = buf;
	       
	       if ( !amps_shmem_info -> bcast_buffers_local[amps_rank] )
		  amps_shmem_info -> bcast_buffers_local[amps_rank] = buf;

	    }
	 }
	 
	 amps_unpack(comm, invoice, (buf -> data) + sizeof(double));
	 
	 if( (invoice) -> combuf_flags & AMPS_INVOICE_OVERLAYED) 
	    (invoice) -> combuf_flags |= AMPS_INVOICE_ALLOCATED; 
	 else 
	 { 
	    (invoice) -> combuf_flags &= ~AMPS_INVOICE_ALLOCATED; 
#if 0
	    printf("%d: freeing data\n", amps_rank);
#endif
	    usfree(buf -> data, amps_arena);
	 } 
	
#if 0 
	 printf("%d: freeing buffer\n", amps_rank);
#endif
	 usfree(buf, amps_arena);
	 
      }

}

