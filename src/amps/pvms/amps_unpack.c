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

	
int amps_unpack(comm, inv)
amps_Comm comm;
amps_Invoice inv;
{
   amps_InvoiceEntry *ptr;
   int len, stride;
   int  malloced = FALSE;
   int size;
   char *data;
   int dim;

   /* we are unpacking so signal this operation */
   
   inv -> flags &= ~AMPS_PACKED;

   /* for each entry in the invoice pack that entry into the letter         */
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
	 {
	    *((void **)(ptr -> data)) = malloc(sizeof(char)
					       *len*stride);
	    malloced = TRUE;
	    pvm_upkbyte(*((void **)(ptr -> data)), len, stride);
	 }
	 else
	    pvm_upkbyte(ptr -> data, len, stride);
	 break;
      case AMPS_INVOICE_SHORT_CTYPE:
	 if( ptr -> data_type == AMPS_INVOICE_POINTER)
	 {
	    *((void **)(ptr -> data)) = malloc(sizeof(short)
						  *len*stride);
	    malloced = TRUE;
	    pvm_upkshort(*((void **)(ptr -> data)), len, stride);
	 }
	 else
	    pvm_upkshort(ptr -> data, len, stride);
	 break;
      case AMPS_INVOICE_INT_CTYPE:
	 if( ptr -> data_type == AMPS_INVOICE_POINTER)
	 {
	    *((void **)(ptr -> data)) = malloc(sizeof(int)
					       *len*stride);
	    malloced = TRUE;
	    pvm_upkint(*((void **)(ptr -> data)), len, stride);
	 }
	 else
	    pvm_upkint(ptr -> data, len, stride);
	 
	 break;
      case AMPS_INVOICE_LONG_CTYPE:
	 if( ptr -> data_type == AMPS_INVOICE_POINTER)
	 {
	    *((void **)(ptr -> data)) = malloc(sizeof(long)
					       *len*stride);
	    malloced = TRUE;
	    pvm_upklong(*((void **)(ptr -> data)), len, stride);
	 }
	 else
	    pvm_upklong(ptr -> data, len, stride);
	 
	 break;
      case AMPS_INVOICE_FLOAT_CTYPE:
	 if( ptr -> data_type == AMPS_INVOICE_POINTER)
	 {
	    *((void **)(ptr -> data)) = malloc(sizeof(float)
					       *len*stride);
	    malloced = TRUE;
	    pvm_upkfloat(*((void **)(ptr -> data)), len, stride);
	 }
	 else
	    pvm_upkfloat(ptr -> data, len, stride);
	 
	 break;
      case AMPS_INVOICE_DOUBLE_CTYPE:
	 if( ptr -> data_type == AMPS_INVOICE_POINTER)
	 {
	    *((void **)(ptr -> data)) = malloc(sizeof(double)
					       *len*stride);
	    malloced = TRUE;
	    pvm_upkdouble(*((void **)(ptr -> data)), len, stride);
	 }
	 else
	    pvm_upkdouble(ptr -> data, len, stride);
	 
	 break;
      default:
	 dim = ( ptr -> dim_type == AMPS_INVOICE_POINTER) ?
	    *(ptr -> ptr_dim) : ptr -> dim;

	 size = amps_vector_sizeof_local( comm, 
					 ptr -> type - AMPS_INVOICE_LAST_CTYPE,
					 NULL, dim, 
					 ptr -> ptr_len, ptr -> ptr_stride);

	 if( ptr -> data_type == AMPS_INVOICE_POINTER )
	    data = *(char **)(ptr -> data) = (char *)malloc(size);
	 else 
	    data = ptr -> data;

	 amps_vector_in( comm, ptr -> type - AMPS_INVOICE_LAST_CTYPE,
			&data, dim-1, ptr -> ptr_len, 
			ptr -> ptr_stride);
      }
      ptr = ptr -> next;
   }  

   if(malloced)
   {
      inv -> combuf_flags |= AMPS_INVOICE_OVERLAYED;
      inv -> combuf_flags |= AMPS_INVOICE_NON_OVERLAYED;
      inv -> comm = comm;
   }

   /* PVM always don't really allocate things */
   inv -> combuf_flags &= ~AMPS_INVOICE_ALLOCATED;
   return 0;
}
