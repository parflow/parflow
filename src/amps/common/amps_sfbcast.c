/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "amps.h"

/**

\Ref{amps_SFBCast} is used to read data from a shared file.  Note
that the input is described by an \Ref{amps_Invoice} rather than the
standard {\bf printf} syntax.  This is to allow a closer mapping to
the communication routines.  Due to this change be careful; items in
the input file must match what is in the invoice description.  As it's
name implies this function reads from a file and broadcasts the data
in the file to all the nodes who are in the {\bf comm} context.  Think
of it as doing an \Ref{amps_BCAST} with a file replacing the node as
the source.  The data is stored in ASCII format and read in using
the Standard C library function {\bf scanf} so it's formatting rules
apply.

{\large Example:}
\begin{verbatim}
amps_File file;
amps_Invoice invoice;

file = amps_SFopen(filename, "r");

amps_SFBCast(amps_CommWorld, file, invoice);

amps_SFclose(file);
\end{verbatim}

{\large Notes:}

@memo Broadcast from a shared file
@param comm Communication context [IN]
@param file Shared file handle [IN]
@param invoice Descriptions of data to read from file and distribute [IN/OUT]
@return Error code
*/
int amps_SFBCast(amps_Comm comm, amps_File file, amps_Invoice invoice)
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


