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

void amps_ClearInvoice(inv)
amps_Invoice inv;
{
   amps_InvoiceEntry *ptr;
   int stride;
   
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
	    if( inv -> combuf_flags & AMPS_INVOICE_NON_OVERLAYED )
	       free( *((void**)(ptr -> data)));
	    else
	       if ( stride != 1)
		  free( *((void**)(ptr -> data)));

	    *((void **)(ptr -> data)) = NULL;
	 }
	 
	 ptr = ptr->next;
      }
      
	inv -> combuf_flags &= ~AMPS_INVOICE_OVERLAYED;
   }
   
   /* No longer have any malloced space associated with this invoice */
   inv -> combuf_flags &= ~AMPS_INVOICE_ALLOCATED;
   
   inv -> combuf_flags &= ~AMPS_INVOICE_NON_OVERLAYED;
   
   return;
}
