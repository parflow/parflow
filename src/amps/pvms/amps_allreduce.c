/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

int amps_ReduceOperation(comm, invoice, buf_src, operation)
amps_Comm comm;
amps_Invoice invoice;
char *buf_src;
int operation;
{
   amps_InvoiceEntry *ptr;
   char *pos_dest, *pos_src;
   char *end_dest;
   int len;
   int stride;

   if(operation)
   {
      ptr = invoice -> list;
      pos_src = buf_src;
      
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
	    pos_dest = *((void **)(ptr -> data));
	 else
	    pos_dest = ptr -> data;
 
	 switch (operation)
	 {
	 case amps_Max:
	    switch(ptr->type)
	    {
	    case AMPS_INVOICE_CHAR_CTYPE:
	       pos_src += AMPS_ALIGN(char, pos_src);
	       pvm_upkbyte( (char*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(char)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(char)*stride, pos_src += sizeof(char))
		  *(char *)pos_dest = 
		     max(*(char *)pos_dest, *(char *)pos_src);
	       break; 

	    case AMPS_INVOICE_SHORT_CTYPE:
	       pos_src += AMPS_ALIGN(short, pos_src);
	       pvm_upkshort( (short*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(short)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(short)*stride, pos_src += sizeof(short))
		  *(short *)pos_dest = 
		     max(*(short *)pos_dest, *(short *)pos_src);
	       break; 

	    case AMPS_INVOICE_INT_CTYPE:
	       pos_src += AMPS_ALIGN(int, pos_src);
	       pvm_upkint( (int*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(int)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(int)*stride, pos_src += sizeof(int))
		  *(int *)pos_dest = 
		     max(*(int *)pos_dest, *(int *)pos_src);
	       break; 

	    case AMPS_INVOICE_LONG_CTYPE:
	       pos_src += AMPS_ALIGN(long, pos_src);
	       pvm_upklong( (long*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(long)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(long)*stride, pos_src += sizeof(long))
		  *(long *)pos_dest = 
		     max(*(long *)pos_dest, *(long *)pos_src);
	       break; 

	    case AMPS_INVOICE_FLOAT_CTYPE:
	       pos_src += AMPS_ALIGN(float, pos_src);
	       pvm_upkfloat( (float*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(float)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(float)*stride, pos_src += sizeof(float))
		  *(float *)pos_dest = 
		     max(*(float *)pos_dest, *(float *)pos_src);
	       break; 

	    case AMPS_INVOICE_DOUBLE_CTYPE:
	       pos_src += AMPS_ALIGN(double, pos_src);
	       pvm_upkdouble( (double*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(double)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(double)*stride, 
		                                   pos_src += sizeof(double))
		  *(double *)pos_dest = 
		     max(*(double *)pos_dest, *(double *)pos_src);
	       break; 
	    }
	    break;	 

	 case amps_Min:
	    switch(ptr->type)
	    {
	    case AMPS_INVOICE_CHAR_CTYPE:
	       pos_src += AMPS_ALIGN(char, pos_src);
	       pvm_upkbyte( (char*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(char)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(char)*stride, pos_src += sizeof(char))
		  *(char *)pos_dest = 
		     min(*(char *)pos_dest, *(char *)pos_src);
	       break; 

	    case AMPS_INVOICE_SHORT_CTYPE:
	       pos_src += AMPS_ALIGN(short, pos_src);
	       pvm_upkshort( (short*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(short)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(short)*stride, pos_src += sizeof(short))
		  *(short *)pos_dest = 
		     min(*(short *)pos_dest, *(short *)pos_src);
	       break; 

	    case AMPS_INVOICE_INT_CTYPE:
	       pos_src += AMPS_ALIGN(int, pos_src);
	       pvm_upkint( (int*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(int)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(int)*stride, pos_src += sizeof(int))
		  *(int *)pos_dest = 
		     min(*(int *)pos_dest, *(int *)pos_src);
	       break; 

	    case AMPS_INVOICE_LONG_CTYPE:
	       pos_src += AMPS_ALIGN(long, pos_src);
	       pvm_upklong( (long*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(long)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(long)*stride, pos_src += sizeof(long))
		  *(long *)pos_dest = 
		     min(*(long *)pos_dest, *(long *)pos_src);
	       break; 

	    case AMPS_INVOICE_FLOAT_CTYPE:
	       pos_src += AMPS_ALIGN(float, pos_src);
	       pvm_upkfloat( (float*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(float)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(float)*stride, pos_src += sizeof(float))
		  *(float *)pos_dest = 
		     min(*(float *)pos_dest, *(float *)pos_src);
	       break; 

	    case AMPS_INVOICE_DOUBLE_CTYPE:
	       pos_src += AMPS_ALIGN(double, pos_src);
	       pvm_upkdouble( (double*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(double)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(double)*stride, 
		                                   pos_src += sizeof(double))
		  *(double *)pos_dest = 
		     min(*(double *)pos_dest, *(double *)pos_src);
	       break; 

	    }
	    break;

	 case amps_Add:
	    switch(ptr->type)
	    {
	    case AMPS_INVOICE_CHAR_CTYPE:
	       pos_src += AMPS_ALIGN(char, pos_src);
	       pvm_upkbyte( (char*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(char)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(char)*stride, pos_src += sizeof(char))
		  *(char *)pos_dest += *(char *)pos_src;
	       break; 

	    case AMPS_INVOICE_SHORT_CTYPE:
	       pos_src += AMPS_ALIGN(short, pos_src);
	       pvm_upkshort( (short*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(short)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(short)*stride, pos_src += sizeof(short))
		  *(short *)pos_dest +=  *(short *)pos_src;
	       break; 

	    case AMPS_INVOICE_INT_CTYPE:
	       pos_src += AMPS_ALIGN(int, pos_src);
	       pvm_upkint( (int*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(int)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(int)*stride, pos_src += sizeof(int))
	       {
		  *(int *)pos_dest += *(int *)pos_src;
	       }
	       break; 

	    case AMPS_INVOICE_LONG_CTYPE:
	       pos_src += AMPS_ALIGN(long, pos_src);
	       pvm_upklong( (long*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(long)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(long)*stride, pos_src += sizeof(long))
		  *(long *)pos_dest +=  *(long *)pos_src;
	       break; 

	    case AMPS_INVOICE_FLOAT_CTYPE:
	       pos_src += AMPS_ALIGN(float, pos_src);
	       pvm_upkfloat( (float*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(float)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(float)*stride, pos_src += sizeof(float))
		  *(float *)pos_dest += *(float *)pos_src;
	       break; 

	    case AMPS_INVOICE_DOUBLE_CTYPE:
	       pos_src += AMPS_ALIGN(double, pos_src);
	       pvm_upkdouble( (double*)pos_src, len, 1);
	       for(end_dest = pos_dest + len*sizeof(double)*stride; 
		   pos_dest < end_dest; 
		   pos_dest += sizeof(double)*stride, 
		                                   pos_src += sizeof(double))
		  *(double *)pos_dest += *(double *)pos_src;
	       break; 
	    }
	    break;
	 default:
	    ;
	 }
	 ptr = ptr->next;
      }
      return 0;
   }
   else
      return 0;
}

int amps_AllReduce(comm, invoice, operation)
amps_Comm comm;
amps_Invoice invoice;
int operation;
{
   int n;
   int N;
   int d;
   int poft, log, npoft;
   int node;

   char *r_buffer;

   int size;

   N = amps_Size(comm);
   n = amps_Rank(comm);
   
   amps_FindPowers(N, &log, &npoft, &poft);

   /* nothing to do if only one node */
   if(N < 2)
      return 0;

   size = amps_sizeof_invoice(comm, invoice);

   if ( n < poft )
   {
      
      r_buffer = (char *)malloc(size);

      if( n < N - poft )
      {
	 node = poft + n;

	 pvm_recv(amps_gettid(comm, node), amps_MsgTag);

	 amps_ReduceOperation(comm, invoice, r_buffer, operation);
      }
      
      for(d = 1; d < poft; d <<= 1)
      {
	 node = ( n ^ d);


	 amps_Send(comm, node, invoice);

	 pvm_recv(amps_gettid(comm, node), amps_MsgTag);
	   
	 amps_ReduceOperation(comm, invoice, r_buffer, operation);
      }
       
      if ( n < N - poft)
      {
	 node = poft + n;

	 amps_Send(comm, node, invoice);

      }

      free(r_buffer);       
   }
   else
   {
      amps_Send(comm,  n - poft, invoice);
      amps_Recv(comm,  n - poft, invoice);
   }

   return 0;
}

