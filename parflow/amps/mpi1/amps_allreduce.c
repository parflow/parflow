/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "amps.h"

/*===========================================================================*/
/**
  The collective operation \Ref{amps_AllReduce} is used to take information
  from each node of a context, perform an operation on the data, and return
  the combined result to the all the nodes.  This operation is also called
  a combine in some message passing systems.  The supported operation are
  \Ref{amps_Max}, \Ref{amps_Min} and \Ref{amps_Add}.

  {\large Example:}
\begin{verbatim}
amps_Invoice invoice;
double       d;
int          i;

invoice = amps_NewInvoice("%i%d", &i, &d);

// find maximum of i and d on all nodes
amps_AllReduce(amps_CommWorld, invoice, amps_Max);

// find sum of i and d on all nodes
amps_AllReduce(amps_CommWorld, invoice, amps_Add);

amps_FreeInvoice(invoice);

\end{verbatim}

  @memo Reduction Operation
  @param comm communication context for the reduction [IN]
  @param invoice invoice to reduce [IN/OUT]
  @param operation reduction operation to perform [IN]
  @return Error code
 */
int amps_AllReduce(amps_Comm comm, amps_Invoice invoice, MPI_Op operation)
{
   amps_InvoiceEntry *ptr;

   int len;
   int stride;

   char *data;
   char *in_buffer;
   char *out_buffer;
   
   char *ptr_src;
   char *ptr_dest;

   MPI_Datatype mpi_type;
   int element_size;

   ptr = invoice -> list;

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
	 mpi_type = MPI_CHAR;
	 element_size = sizeof(char);
	 break;
      case AMPS_INVOICE_SHORT_CTYPE:
	 mpi_type = MPI_SHORT;
	 element_size = sizeof(short);
	 break;
      case AMPS_INVOICE_INT_CTYPE:
	 mpi_type = MPI_INT;
	 element_size = sizeof(int);
	 break;
      case AMPS_INVOICE_LONG_CTYPE:
	 mpi_type = MPI_LONG;
	 element_size = sizeof(long);
	 break;
      case AMPS_INVOICE_FLOAT_CTYPE:
	 mpi_type = MPI_FLOAT;
	 element_size = sizeof(float);
	 break;
      case AMPS_INVOICE_DOUBLE_CTYPE:
	 mpi_type = MPI_DOUBLE;
	 element_size = sizeof(double);
	 break;
      default:
	 printf("AMPS Operation not supported\n");
      }
      
      in_buffer = malloc(element_size*len);
      out_buffer = malloc(element_size*len);

      /* Copy into a contigous buffer */
      if(stride == 1) 
	 bcopy( data, in_buffer, len*element_size);
      else 
	 for(ptr_src = data, ptr_dest = in_buffer; 
	     ptr_src < data + len*stride*element_size;
	     ptr_src += stride*element_size, ptr_dest += element_size) 
		bcopy(ptr_src, ptr_dest, element_size); 

      MPI_Allreduce(in_buffer, out_buffer, len, mpi_type, operation, comm);
      
      /* Copy back into user variables */
      if(stride == 1) 
	 bcopy( out_buffer, data, len*element_size);
      else 
	 for(ptr_src = out_buffer, ptr_dest = data; 
	     ptr_src < out_buffer + len*element_size;
	     ptr_src += element_size, ptr_dest += stride*element_size) 
	    bcopy(ptr_src, ptr_dest, element_size); 

      free(in_buffer);
      free(out_buffer);

      ptr = ptr -> next;
   }

   return 0;
}

