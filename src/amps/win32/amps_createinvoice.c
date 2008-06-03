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

int amps_CreateInvoice(comm, inv)
     amps_Comm comm;
     amps_Invoice inv;
{
  amps_InvoiceEntry *ptr;
  char *cur_pos;
  int size, len, stride;
   
  size = amps_sizeof_invoice(comm, inv);
   
  /* if allocated then we deallocate                                       */
  amps_ClearInvoice(inv);
   
  cur_pos = inv -> combuf = amps_new(mlr, size);
   
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
	  cur_pos += AMPS_CALL_CHAR_ALIGN(comm, NULL, cur_pos, len, stride);
	  if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    if ( stride == 1 )
	      *( (void **)(ptr -> data)) = cur_pos;
	    else
	      *( (void **)(ptr -> data)) = 
		malloc(sizeof(char)*len*stride);
	  cur_pos += AMPS_CALL_CHAR_SIZEOF(comm, cur_pos, NULL, len, stride);
	  break;
	case AMPS_INVOICE_SHORT_CTYPE:
	  cur_pos += AMPS_CALL_SHORT_ALIGN(comm, NULL, cur_pos, len, stride);
	  if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    if ( stride == 1 )
	      *( (void **)(ptr -> data)) = cur_pos;
	    else
	      *( (void **)(ptr -> data)) = 
		malloc(sizeof(short)*len*stride);
	  cur_pos += AMPS_CALL_SHORT_SIZEOF(comm, cur_pos, NULL, len, stride);
	  break;
	case AMPS_INVOICE_INT_CTYPE:
	  cur_pos += AMPS_CALL_INT_ALIGN(comm, NULL, cur_pos, len, stride);
	  if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    if ( stride == 1 )
	      *( (void **)(ptr -> data)) = cur_pos;
	    else
	      *( (void **)(ptr -> data)) = 
		malloc(sizeof(int)*len*stride);
	  cur_pos += AMPS_CALL_INT_SIZEOF(comm, cur_pos, NULL, len, stride);
	  break;
	case AMPS_INVOICE_LONG_CTYPE:
	  cur_pos += AMPS_CALL_LONG_ALIGN(comm, NULL, cur_pos, len, stride);
	  if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    if ( stride == 1 )
	      *( (void **)(ptr -> data)) = cur_pos;
	    else
	      *( (void **)(ptr -> data)) = 
		malloc(sizeof(long)*len*stride);
	  cur_pos += AMPS_CALL_LONG_SIZEOF(comm, cur_pos, NULL, len, stride);
	  break;
	case AMPS_INVOICE_FLOAT_CTYPE:
	  cur_pos += AMPS_CALL_FLOAT_ALIGN(comm, NULL, cur_pos, len, stride);
	  if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    if ( stride == 1 )
	      *( (void **)(ptr -> data)) = cur_pos;
	    else
	      *( (void **)(ptr -> data)) = 
		malloc(sizeof(float)*len*stride);
	  cur_pos += AMPS_CALL_FLOAT_SIZEOF(comm, cur_pos, NULL, len, stride);
	  break;
	case AMPS_INVOICE_DOUBLE_CTYPE:
	  cur_pos += AMPS_CALL_DOUBLE_ALIGN(comm, NULL, cur_pos, len, stride);
	  if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    if ( stride == 1 )
	      *( (void **)(ptr -> data)) = cur_pos;
	    else
	      *( (void **)(ptr -> data)) = 
		malloc(sizeof(double)*len*stride);
	  cur_pos += AMPS_CALL_DOUBLE_SIZEOF(comm, cur_pos, NULL, len, stride);
	  break;
	}
      ptr = ptr->next;
    }
  return size;
}




