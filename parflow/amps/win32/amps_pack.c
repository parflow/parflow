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

int amps_pack(comm, inv, buffer)
     amps_Comm comm;
     amps_Invoice inv;
     char **buffer;
{
  amps_InvoiceEntry *ptr;
  char *cur_pos;
  char *temp_pos;
  char *data;
  int size, stride;

  int len;
  int dim;

  size = amps_sizeof_invoice(comm, inv);
   
  inv -> flags |= AMPS_PACKED;
   
  /* check to see if this was already allocated                            */
  if( ( inv -> combuf_flags & AMPS_INVOICE_ALLOCATED) )
    {
      *buffer = inv -> combuf;
    }
  else
    {
      if( (*buffer = amps_new(comm, size)) == NULL)
	amps_Error("amps_pack", OUT_OF_MEMORY, "malloc of letter", HALT);
      else
	inv -> combuf_flags |= AMPS_INVOICE_ALLOCATED;
    }

  /* for each entry in the invoice pack that entry into the letter         */
  ptr = inv -> list;
  cur_pos = *buffer;
  while(ptr != NULL)
    {
      /* invoke the packing convert out for the entry */
      /* if user then call user ones */
      /* else switch on builtin type */
      if(ptr -> len_type == AMPS_INVOICE_POINTER)
	len = *(ptr -> ptr_len);
      else
	len = ptr ->len;
      
      if(ptr -> stride_type == AMPS_INVOICE_POINTER)
	stride = *(ptr -> ptr_stride);
      else
	stride = ptr -> stride;
      
      if( ptr -> data_type == AMPS_INVOICE_POINTER)
	data = *((char **)(ptr -> data));
      else
	data = ptr -> data;
 
      switch(ptr->type)
	{
	case AMPS_INVOICE_CHAR_CTYPE:

	  cur_pos += AMPS_CALL_CHAR_ALIGN(comm, NULL, cur_pos, len, stride);
	  if(!ptr->ignore)
	    AMPS_CALL_CHAR_OUT(comm, data, cur_pos, len, stride);
	  cur_pos += AMPS_CALL_CHAR_SIZEOF(comm, cur_pos, NULL, len, stride);
	  break;
	case AMPS_INVOICE_SHORT_CTYPE:
	  cur_pos += AMPS_CALL_SHORT_ALIGN(comm, NULL, cur_pos, len, stride);
	  if(!ptr->ignore)
	    AMPS_CALL_SHORT_OUT(comm, data, cur_pos, len, stride);
	  cur_pos += AMPS_CALL_SHORT_SIZEOF(comm, cur_pos, NULL, len, stride);
	  break;
	case AMPS_INVOICE_INT_CTYPE:
	  cur_pos += AMPS_CALL_INT_ALIGN(comm, NULL, cur_pos, len, stride);
	  if(!ptr->ignore)
	    AMPS_CALL_INT_OUT(comm, data, cur_pos, len, 
			      stride);
	  cur_pos += AMPS_CALL_INT_SIZEOF(comm, cur_pos, NULL, len, stride);
	  break;
	case AMPS_INVOICE_LONG_CTYPE:
	  cur_pos += AMPS_CALL_LONG_ALIGN(comm, NULL, cur_pos, len, stride);
	  if(!ptr->ignore)
	    AMPS_CALL_LONG_OUT(comm, data, cur_pos, len, 
			       stride);
	  cur_pos += AMPS_CALL_LONG_SIZEOF(comm, cur_pos, NULL, len, stride);
	  break;
	case AMPS_INVOICE_FLOAT_CTYPE:
	  cur_pos += AMPS_CALL_FLOAT_ALIGN(comm, NULL, cur_pos, len, stride);
	  if(!ptr->ignore)
	    AMPS_CALL_FLOAT_OUT(comm, data, cur_pos, len, 
				stride);
	  cur_pos += AMPS_CALL_FLOAT_SIZEOF(comm, cur_pos, NULL, len, stride);
	  break;
	case AMPS_INVOICE_DOUBLE_CTYPE:
	  cur_pos += AMPS_CALL_DOUBLE_ALIGN(comm, NULL, cur_pos, len, stride);
	  if(!ptr->ignore)
	    AMPS_CALL_DOUBLE_OUT(comm, data, cur_pos, len, 
				 stride);
	  cur_pos += AMPS_CALL_DOUBLE_SIZEOF(comm, cur_pos, NULL, len, stride);
	  break;
	default:
	  dim = ( ptr -> dim_type == AMPS_INVOICE_POINTER) ?
	    *(ptr -> ptr_dim) : ptr -> dim;
	  temp_pos = cur_pos;
	  cur_pos += amps_vector_align(comm, 
				       ptr  -> type - AMPS_INVOICE_LAST_CTYPE,
				       &data, &cur_pos, dim, 
				       ptr -> ptr_len, ptr -> ptr_stride);
	  temp_pos = cur_pos;
	  amps_vector_out( comm, ptr -> type - AMPS_INVOICE_LAST_CTYPE,
			  &data, &temp_pos, dim-1, ptr -> ptr_len, 
			  ptr -> ptr_stride);
	  temp_pos = cur_pos;
	  cur_pos += amps_vector_sizeof_buffer( comm, 
					       ptr -> type - AMPS_INVOICE_LAST_CTYPE,
					       &data, &temp_pos, dim, 
					       ptr -> ptr_len, ptr -> ptr_stride);
	}
      ptr = ptr->next;
    }

  return size;
}


