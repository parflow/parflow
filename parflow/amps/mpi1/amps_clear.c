/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include <string.h>
#include <stdarg.h>

#include "amps.h"

/*===========================================================================*/
/**

This function is used to free overlayed variables in the
{\bf invoice}.  \Ref{amps_Clear} is used after a receive operation
and when you have finished manipulating the overlayed variables that
were received.  After the {\bf invoice} has been cleared it is illegal
to access the overlayed variables.  Overlayed variables are generally
used for temporary values that need to be received but don't need to
be kept for around for an extended period of time.


{\large Example:}
\begin{verbatim}
amps_Invoice invoice;
int me, i;
double *d;

me = amps_Rank(amps_CommWorld);

invoice = amps_NewInvoice("%*\\d", 10, &d);

amps_Recv(amps_CommWorld, me-1, invoice);

for(i=0; i<10; i++)
{
        do_work(d[i]);
}

amps_Clear(invoice);

// can't access d array after clear 

amps_FreeInvoice(invoice);

\end{verbatim}

{\large Notes:}

In message passing systems that use buffers, overlayed variables are
located in the buffer.  This eliminates a copy.

  @memo Free overlayed variables associated with an invoice.
  @param inv Invoice to clear [IN/OUT]
  @return none
*/
void amps_ClearInvoice(amps_Invoice inv)
{
   amps_InvoiceEntry *ptr;
   int stride;
   
   
   /* if allocated then we deallocate                                       */
   if( inv -> combuf_flags & AMPS_INVOICE_ALLOCATED )
      amps_free(comm, inv -> combuf);
   
   /* set flag to unpack so free will occur if strange things happen */
   inv -> flags &= ~AMPS_PACKED;
   
   if ( (inv -> combuf_flags & AMPS_INVOICE_OVERLAYED) 
     || (inv -> combuf_flags & AMPS_INVOICE_NON_OVERLAYED) )
   {
      /* for each entry in the invoice pack null out the pointer
	 if needed  and free up space if we malloced any          */
      ptr = inv -> list;
      while(ptr != NULL)
      {
	 if(ptr -> stride_type == AMPS_INVOICE_POINTER)
	    stride = *(ptr -> ptr_stride);
	 else
	    stride = ptr -> stride;
	 
	 /* check if we actually created any space */
	 if( ptr -> data_type == AMPS_INVOICE_POINTER)
	 {
	    free( *((void**)(ptr -> data)));
	    *((void **)(ptr -> data)) = NULL;
	 }
	 
	 ptr = ptr->next;
      }
      
	inv -> combuf_flags &= ~AMPS_INVOICE_OVERLAYED;
   }
   
   /* No longer have any malloced space associated with this invoice */
   inv -> combuf_flags &= ~AMPS_INVOICE_ALLOCATED;
   inv -> combuf = NULL;
   
   inv -> combuf_flags &= ~AMPS_INVOICE_NON_OVERLAYED;
   
   return;
}
