/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"
#include <string.h>
#include <stdarg.h>

int amps_CreateInvoice(comm, inv)
amps_Comm comm;
amps_Invoice inv;
{
   amps_InvoiceEntry *ptr;
   int len, stride;
   
   /* if allocated then we deallocate                                       */
   amps_ClearInvoice(inv);
   
   /* set flag indicateing we have allocated the space                      */
   inv -> combuf_flags |= AMPS_INVOICE_ALLOCATED;
   inv -> combuf_flags |= AMPS_INVOICE_OVERLAYED;
   
   inv -> comm = comm;
   
   ptr = inv -> list;
   while(ptr != NULL)
   {
      if(ptr -> len_type == AMPS_INVOICE_POINTER)
	 len = *(ptr -> ptr_len);
      else
	 len = ptr ->len;
      
      if(ptr -> stride_type == AMPS_INVOICE_POINTER)
	 stride = *(ptr -> ptr_stride);
      else
	 stride = ptr -> stride;
      
      switch(ptr->type)
      {
      case AMPS_INVOICE_CHAR_CTYPE:
	 if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    *( (void **)(ptr -> data)) = (void *)malloc(sizeof(char)*len*stride);
	 break;
      case AMPS_INVOICE_SHORT_CTYPE:
	 if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    *( (void **)(ptr -> data)) = (void *)malloc(sizeof(short)*len*stride);
	 break;
      case AMPS_INVOICE_INT_CTYPE:
	 if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    *( (void **)(ptr -> data)) = (void *)malloc(sizeof(int)*len*stride);
	 break;
      case AMPS_INVOICE_LONG_CTYPE:
	 if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    *( (void **)(ptr -> data)) = (void *)malloc(sizeof(long)*len*stride);
	 break;
      case AMPS_INVOICE_FLOAT_CTYPE:
	 if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    *( (void **)(ptr -> data)) = (void *)malloc(sizeof(float)*len*stride);
	 break;
      case AMPS_INVOICE_DOUBLE_CTYPE:
	 if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    *( (void **)(ptr -> data)) = (void *)malloc(sizeof(double)*len*stride);
	 break;
      }
      ptr = ptr->next;
   }
   return 0;
}




