#include "amps.h"

int amps_SFBCast(comm, file, invoice)
amps_Comm comm;
amps_File file;
amps_Invoice invoice;
{
   amps_InvoiceEntry *ptr;
   int stride, len;
   int malloced = 0;

   if(!amps_Rank(comm))   
   {
      amps_ClearInvoice(invoice);

      invoice -> combuf_flags |= AMPS_INVOICE_NON_OVERLAYED;

      invoice -> comm = comm;

      /* for each entry in the invoice read the value from the input file */
      ptr = invoice -> list;
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
	 
	 switch(ptr->type)
	 {
	 case AMPS_INVOICE_CHAR_CTYPE:
	    if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    {
	       *( (void **)(ptr -> data)) = (void *)malloc(sizeof(char)*len*stride);
	       amps_ScanChar(file, *( void **)(ptr -> data), len, stride);
	       malloced = TRUE;
	    }
	    else
	       amps_ScanChar(file, ptr -> data, len, stride);
	    break;
	 case AMPS_INVOICE_SHORT_CTYPE:
	    if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    {
	       *( (void **)(ptr -> data)) = (void *)malloc(sizeof(short)*len*stride);
	       amps_ScanShort(file, *( void **)(ptr -> data), len, stride);
	       malloced = TRUE;
	    }
	    else
	       amps_ScanShort(file, ptr -> data, len, stride);
	    
	    break;
	 case AMPS_INVOICE_INT_CTYPE:
	    if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    {
	       *( (void **)(ptr -> data)) = (void *)malloc(sizeof(int)*len*stride);
	       amps_ScanInt(file, *( void **)(ptr -> data), len, stride);
	       malloced = TRUE;
	    }
	    else
	       amps_ScanInt(file, ptr -> data, len, stride);
	    
	    break;
	 case AMPS_INVOICE_LONG_CTYPE:
	    if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    {
	       *( (void **)(ptr -> data)) = (void *)malloc(sizeof(long)*len*stride);
	       amps_ScanLong(file, *( void **)(ptr -> data), len, stride);
	       malloced = TRUE;
	    }
	    else
	       amps_ScanLong(file, ptr -> data, len, stride);
	    
	    break;
	 case AMPS_INVOICE_FLOAT_CTYPE:
	    if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    {
	       *( (void **)(ptr -> data)) = (void *)malloc(sizeof(float)*len*stride);
	       amps_ScanFloat(file, *( void **)(ptr -> data), len, stride);
	       malloced = TRUE;
	    }
	    else
	       amps_ScanFloat(file, ptr -> data, len, stride);
	    
	    break;
	 case AMPS_INVOICE_DOUBLE_CTYPE:
	    if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    {
	       *( (void **)(ptr -> data)) = (void *)malloc(sizeof(double)*len*stride);
	       amps_ScanDouble(file, *( void **)(ptr -> data), len, stride);
	       malloced = TRUE;
	    }
	    else
	       amps_ScanDouble(file, ptr -> data, len, stride);
	    
	    break;
	 }
	 ptr = ptr->next;
      }
   }

   return amps_BCast(comm, 0, invoice);
}


