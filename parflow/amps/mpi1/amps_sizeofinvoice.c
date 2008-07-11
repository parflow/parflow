/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "amps.h"

long amps_sizeof_invoice(comm, inv)
amps_Comm comm;
amps_Invoice inv;
{
    amps_InvoiceEntry *ptr;
    char *cur_pos = 0;
    char *temp_pos = 0;
    int len, stride;
    char *data;

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

	if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    data = *((char **)(ptr -> data));
	else
	    data = ptr -> data;

	switch(ptr->type)
	{
	case AMPS_INVOICE_CHAR_CTYPE:
	    cur_pos += AMPS_CALL_CHAR_ALIGN(comm, data, cur_pos, len, stride);
	    cur_pos += AMPS_CALL_CHAR_SIZEOF(comm, data, cur_pos, 
				    len, stride);
	    break;
	case AMPS_INVOICE_SHORT_CTYPE:
	    cur_pos += AMPS_CALL_SHORT_ALIGN(comm, data, cur_pos, len, stride);
	    cur_pos += AMPS_CALL_SHORT_SIZEOF(comm, data, cur_pos, 
				     len, stride);
	    break;
	case AMPS_INVOICE_INT_CTYPE:
	    cur_pos += AMPS_CALL_INT_ALIGN(comm, data, cur_pos, len, stride);
	    cur_pos += AMPS_CALL_INT_SIZEOF(comm, data, cur_pos, 
				   len, stride);
	    break;
	case AMPS_INVOICE_LONG_CTYPE:
	    cur_pos += AMPS_CALL_LONG_ALIGN(comm, data, cur_pos, len, stride);
	    cur_pos += AMPS_CALL_LONG_SIZEOF(comm, data, cur_pos, 
				    len, stride);
	    break;
	case AMPS_INVOICE_FLOAT_CTYPE:
	    cur_pos += AMPS_CALL_FLOAT_ALIGN(comm, data, cur_pos, len, stride);
	    cur_pos += AMPS_CALL_FLOAT_SIZEOF(comm, data, cur_pos, 
				     len, stride);
	    break;
	case AMPS_INVOICE_DOUBLE_CTYPE:
	    cur_pos += AMPS_CALL_DOUBLE_ALIGN(comm, data, cur_pos, len, stride);
	    cur_pos += AMPS_CALL_DOUBLE_SIZEOF(comm, data, cur_pos, 
				      len, stride);
	    break;
	default:
	    temp_pos = cur_pos;
	    cur_pos += amps_vector_align(comm, 
					 ptr -> type - AMPS_INVOICE_LAST_CTYPE,
					 &data, &temp_pos, (int)ptr -> dim, 
					 ptr -> ptr_len, ptr -> ptr_stride);
	    temp_pos = cur_pos;
	    cur_pos += amps_vector_sizeof_buffer(comm, 
					 ptr -> type - AMPS_INVOICE_LAST_CTYPE,
					 &data, &temp_pos, (int)ptr -> dim, 
					 ptr -> ptr_len, ptr -> ptr_stride);
	}

	ptr = ptr->next;
    }
    return (long) cur_pos;
}
