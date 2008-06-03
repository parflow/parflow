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

int amps_pack(comm, inv)
amps_Comm comm;
amps_Invoice inv;
{
   amps_InvoiceEntry *ptr;
   int stride;
   char *data;
   int len;
   int dim;

   inv -> flags |= AMPS_PACKED;
   
   /* for each entry in the invoice pack that entry into the letter         */
   ptr = inv -> list;
   
   pvm_initsend(AMPS_ENCODING);

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
	 data = *((void **)(ptr -> data));
      else
	  data = ptr -> data;
 
      switch(ptr->type)
      {
      case AMPS_INVOICE_CHAR_CTYPE:
	 pvm_pkbyte((char*)data, len, stride);
	 break;
      case AMPS_INVOICE_SHORT_CTYPE:
	 pvm_pkshort((short*)data, len, stride);
	 break;
      case AMPS_INVOICE_INT_CTYPE:
	 pvm_pkint((int*)data, len, stride);
	 break;
      case AMPS_INVOICE_LONG_CTYPE:
	 pvm_pklong((long*)data, len, stride);
	 break;
       case AMPS_INVOICE_FLOAT_CTYPE:
	 pvm_pkfloat((float*)data, len, stride);
	 break;
      case AMPS_INVOICE_DOUBLE_CTYPE:
	 pvm_pkdouble((double*)data, len, stride);
	 break;
      default:
	 dim = ( ptr -> dim_type == AMPS_INVOICE_POINTER) ?
	    *ptr -> ptr_dim : ptr -> dim;
	 amps_vector_out( comm, ptr -> type - AMPS_INVOICE_LAST_CTYPE,
			 &data, dim-1, ptr -> ptr_len, 
			 ptr -> ptr_stride);
      }
      ptr = ptr->next;
   }

   return 1;

}


